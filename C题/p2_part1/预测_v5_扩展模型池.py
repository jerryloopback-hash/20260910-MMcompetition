# -*- coding: utf-8 -*-
"""
2026 C题 问题2-预测部分 v5：扩展模型池 + EGD时变权重集成
============================================================
模型池（机制差异化，负载/光伏分别定制，剔除误差高相关的同质树模型）：
  负载(5): SeasonalNaive, HistMean, STL分量, kNN相似日, LightGBM
  光伏(5): Persistence, kNN天气形态, STL分量, FPCA曲线, LightGBM

方法学依据：
1. 滚动origin：Tashman(IJF 2000)、Petropoulos et al.(IJF 2022) — 扩展窗口(anchored)杜绝未来信息泄漏
2. 在线权重：OneNet(NeurIPS 2023) — EGD指数梯度下降 w_{t+1,i}=w_{t,i}exp(-ηℓ̃)/Z
3. K步重初始化：OneNet Prop.2 — 每K步重置缓解概念漂移"慢切换"
4. 局部核加权损失：TVJMA(J.Econometrics 2020) — 30天指数衰减，近期主导
5. STL分解：Cleveland(J.Official Stat. 1990) — 训练窗内合规分解，替代CEEMDAN/VMD的全局泄漏
6. FPCA：Wagner-Muns(IEEE T-ITS 2018) — 日曲线低秩重构，与逐点模型机制正交
7. kNN相似日：TESLA(Akyurek UCSD)、tsfknn(R Journal 2019) — 形态匹配非参数方法
8. HarmonicReg：TBATS(De Livera/Hyndman/Snyder JASA 2011)季节部分的等价快速实现
9. 冷启动：Graefe et al.(2015) — 高不确定性下简单方法优先

更新策略：扩展窗口训练ML(老数据不丢) + 30天核衰减损失(近期主导) = anchored与rolling折中
防过拟合：LGB强正则+早停、时序切分诊断、模型间误差相关性分析
"""
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
import os, warnings, time
warnings.filterwarnings('ignore')

DATA_PATH = r'C:\Users\adminl\Desktop\20260910-MMcompetition\C题\附件\附件2.xlsx'
OUT_DIR = r'D:\CUMCM2026Problems\C题\预测结果'
os.makedirs(OUT_DIR, exist_ok=True)

N_SLOT = 144
START_DAY = '2025-02-01'
END_DAY = '2025-12-31'
RETRAIN_EVERY = 7
COLD_START_DAYS = 35

# 负载模型池(7)
LOAD_MODELS = ['SeasonalNaive', 'HistMean', 'STL', 'kNN', 'LightGBM']
# 光伏模型池(5)
PV_MODELS = ['Persistence', 'kNN', 'STL', 'FPCA', 'LightGBM']

# EGD参数
EGD_ETA = 0.005
EGD_WINDOW = 30
EGD_LAMBDA = 0.95

# 时点列名：与附件2完全一致（00:10:00 → 23:50:00 + 0:00+1）
TIME_COLS = [datetime.time((t+1)*10//60 % 24, (t+1)*10 % 60) for t in range(N_SLOT-1)]
TIME_COLS.append('0:00+1')
DATE_COL = '日期\\时间'

# ti=0→00:10→hour=1/6, ti=143→24:00→hour=24.0
def ti_to_hour(ti_arr):
    return (ti_arr + 1) * 10.0 / 60.0

# ---------------- 1. 数据加载 ----------------
def load_data():
    df_l = pd.read_excel(DATA_PATH, sheet_name='小区负载')
    df_p = pd.read_excel(DATA_PATH, sheet_name='光伏发电实际功率')
    dates = pd.to_datetime(df_l.iloc[:, 0].values).values
    load_mat = df_l.iloc[:, 1:1+N_SLOT].astype(float).values
    pv_mat = df_p.iloc[:, 1:1+N_SLOT].astype(float).values
    return dates, load_mat, pv_mat

# ---------------- 2. 特征工程（LightGBM用） ----------------
def make_calendar_features(dates_arr, time_idx_arr):
    dts = pd.to_datetime(dates_arr)
    hour = ti_to_hour(time_idx_arr)
    dow = dts.dayofweek.values.astype(float)
    return pd.DataFrame({
        'doy': dts.dayofyear.values.astype(float),
        'dow': dow,
        'month': dts.month.values.astype(float),
        'weekend': (dow >= 5).astype(float),
        'hour': hour,
        'sin_t': np.sin(2*np.pi*hour/24.0),
        'cos_t': np.cos(2*np.pi*hour/24.0),
        'sin_dow': np.sin(2*np.pi*dow/7.0),
        'cos_dow': np.cos(2*np.pi*dow/7.0),
        'time_idx': time_idx_arr.astype(float),
    })

def add_lag_features(df, mat, date_to_idx):
    di = np.array([date_to_idx.get(pd.Timestamp(d), -1) for d in df['date'].values])
    ti = df['time_idx'].values.astype(int)
    valid = di >= 0
    for name, off in [('lag1',1), ('lag2',2), ('lag7',7), ('lag14',14), ('lag21',21), ('lag28',28)]:
        arr = np.full(len(df), np.nan)
        j = di - off
        m = valid & (j >= 0)
        arr[m] = mat[j[m], ti[m]]
        df[name] = arr
    arr = np.full(len(df), np.nan)
    acc = np.zeros(len(df)); cnt = np.zeros(len(df))
    for jj in (di-7, di-14, di-21, di-28):
        mm = valid & (jj >= 0)
        acc[mm] += mat[jj[mm], ti[mm]]
        cnt[mm] += 1
    m = cnt > 0
    arr[m] = acc[m] / cnt[m]
    df['lag7mean'] = arr
    arr = np.full(len(df), np.nan)
    j = di - 1
    m = valid & (j >= 0)
    arr[m] = mat[j[m], :].mean(axis=1)
    df['lag1dmean'] = arr
    return df

FEATS = ['doy','dow','month','weekend','hour','sin_t','cos_t','sin_dow','cos_dow','time_idx',
         'lag1','lag2','lag7','lag14','lag21','lag28','lag7mean','lag1dmean']

# ---------------- 3. 简单基线模型 ----------------
def persistence(mat, i_day):
    return mat[i_day-1].copy()

def seasonal_naive(mat, i_day):
    if i_day-7 >= 0:
        return mat[i_day-7].copy()
    return mat[i_day-1].copy()

def hist_mean(mat, i_day):
    acc = np.zeros(N_SLOT); cnt = 0
    for off in (7, 14, 21, 28):
        if i_day - off >= 0:
            acc += mat[i_day-off, :]
            cnt += 1
    if cnt > 0:
        return acc / cnt
    return mat[i_day-1].copy()

# ---------------- 4. STL分量预测（合规分解，替代CEEMDAN/VMD） ----------------
def stl_predict(mat, i_day, period=7):
    """对每个ti的历史日序列做STL(period=7)，趋势阻尼外推+季节相位外推"""
    pred = np.zeros(N_SLOT)
    for ti in range(N_SLOT):
        series = mat[:i_day, ti].astype(float)
        n = len(series)
        if n < 2*period + 3:
            pred[ti] = np.mean(series[-28:]) if n >= 7 else (series[-1] if n > 0 else 0.0)
            continue
        try:
            res = STL(series, period=period, seasonal=7, trend=15, robust=True).fit()
            trend = res.trend
            # 近7天平均斜率，阻尼因子0.8外推1步
            slope = (trend[-1] - trend[-8]) / 7.0
            trend_next = trend[-1] + 0.8 * slope
            # 季节分量取上一个同相位
            seasonal_next = res.seasonal[-period]
            pred[ti] = trend_next + seasonal_next
        except Exception:
            pred[ti] = np.mean(series[-28:])
    return pred

# ---------------- 5. FPCA日曲线低秩重构 ----------------
def fpca_predict_load(mat, i_day, dates, n_weeks=8, n_comp=3):
    """负载：同dow历史曲线SVD，剔除最远1条离群曲线后低秩重构取稳健均值（避免春节等异常日传播）"""
    pred_dow = pd.Timestamp(dates[i_day]).dayofweek
    candidates = []
    for off in range(7, 7*n_weeks+1, 7):
        j = i_day - off
        if j >= 0:
            candidates.append(j)
    if len(candidates) < 2:
        return hist_mean(mat, i_day)
    curves = mat[candidates].astype(float)
    mean = curves.mean(axis=0)
    # 剔除最远1条离群曲线（n>=5时）
    if len(curves) >= 5:
        dists = np.sqrt(((curves - mean)**2).sum(axis=1))
        keep = np.argsort(dists)[:-1]
        curves = curves[keep]
        mean = curves.mean(axis=0)
    centered = curves - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        K = min(n_comp, len(curves)-1)
        reconstructed = mean + (U[:, :K] * S[:K]) @ Vt[:K, :]
        return reconstructed.mean(axis=0)
    except Exception:
        return mean

def fpca_predict_pv(mat, i_day, n_days=14, n_comp=3, damping=0.4):
    """光伏：近n_days天曲线SVD，阻尼score persistence"""
    start = max(0, i_day - n_days)
    curves = mat[start:i_day].astype(float)
    if len(curves) < 3:
        return mat[i_day-1].copy()
    mean = curves.mean(axis=0)
    centered = curves - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        K = min(n_comp, len(curves)-1)
        scores = U[:, :K] * S[:K]
        pred_score = scores[-1]
        pred_curve = mean + damping * pred_score @ Vt[:K, :]
        return np.clip(pred_curve, 0, None)
    except Exception:
        return np.clip(mean, 0, None)

# ---------------- 6. kNN相似日 ----------------
def knn_predict(mat, i_day, dates, k=3, by_dow=True, n_history=60):
    """kNN相似日：查询=昨天曲线，在历史中找k条与昨天最相似的曲线，取其次日曲线加权平均"""
    if i_day < 8:
        return hist_mean(mat, i_day)
    query = mat[i_day-1].astype(float)
    # 候选日j应与昨天(i_day-1)同dow，这样j+1与预测日i_day同dow
    query_dow = pd.Timestamp(dates[i_day-1]).dayofweek
    lo = max(1, i_day - n_history)
    candidates = []
    cand_curves = []
    for j in range(lo, i_day - 1):
        if by_dow and pd.Timestamp(dates[j]).dayofweek != query_dow:
            continue
        candidates.append(j)
        cand_curves.append(mat[j].astype(float))
    if len(candidates) == 0:
        return hist_mean(mat, i_day)
    cand_curves = np.array(cand_curves)
    # 归一化曲线后算欧氏距离（消除整体量级差异）
    q_norm = query / (np.abs(query).max() + 1e-6)
    c_norm = cand_curves / (np.abs(cand_curves).max(axis=1, keepdims=True) + 1e-6)
    dist = np.sqrt(((c_norm - q_norm)**2).sum(axis=1))
    kk = min(k, len(candidates))
    idx = np.argsort(dist)[:kk]
    weights = 1.0 / (dist[idx] + 1e-6)
    weights = weights / weights.sum()
    pred = np.zeros(N_SLOT)
    for w, ii in zip(weights, idx):
        pred += w * mat[candidates[ii] + 1]  # 取候选日的次日曲线
    return pred

# ---------------- 7. HarmonicReg（TBATS多周期季节部分的等价快速实现） ----------------
def harmonic_reg_predict(mat, i_day, period=7, n_harm=2):
    """对每个ti的历史日序列，用[1, t, sin/cos谐波]做Ridge，外推1步"""
    pred = np.zeros(N_SLOT)
    for ti in range(N_SLOT):
        series = mat[:i_day, ti].astype(float)
        n = len(series)
        if n < 14:
            pred[ti] = series[-1] if n > 0 else 0.0
            continue
        t = np.arange(n, dtype=float)
        cols = [np.ones(n), t]
        for h in range(1, n_harm+1):
            cols.append(np.sin(2*np.pi*h*t/period))
            cols.append(np.cos(2*np.pi*h*t/period))
        X = np.column_stack(cols)
        model = Ridge(alpha=1.0)
        model.fit(X, series)
        t_next = np.array([float(n)])
        cols_next = [np.ones(1), t_next]
        for h in range(1, n_harm+1):
            cols_next.append(np.sin(2*np.pi*h*t_next/period))
            cols_next.append(np.cos(2*np.pi*h*t_next/period))
        X_next = np.column_stack(cols_next)
        pred[ti] = model.predict(X_next)[0]
    return pred

# ---------------- 8. LightGBM（唯一ML模型，强正则+早停） ----------------
def build_train_matrix(dates, mat, train_end_idx, date_to_idx, val_days=7):
    va_start = train_end_idx - val_days + 1
    tr_end = va_start - 1
    def _build(lo, hi):
        idxs = np.arange(lo, hi+1)
        if len(idxs) == 0:
            return None, None
        di = np.repeat(idxs, N_SLOT)
        ti = np.tile(np.arange(N_SLOT), len(idxs))
        df = make_calendar_features(dates[di], ti)
        df['date'] = pd.to_datetime(dates[di])
        df['time_idx'] = ti
        df = add_lag_features(df, mat, date_to_idx)
        df['target'] = mat[di, ti]
        msk = df[FEATS].notna().all(axis=1)
        return df.loc[msk, FEATS].values, df.loc[msk, 'target'].values
    X_tr, y_tr = _build(28, tr_end)
    X_va, y_va = _build(va_start, train_end_idx)
    return X_tr, y_tr, X_va, y_va

def build_pred_matrix(day, date_to_idx, mat):
    ti = np.arange(N_SLOT)
    dates_arr = np.full(N_SLOT, np.datetime64(day), dtype='datetime64[D]')
    df = make_calendar_features(dates_arr, ti)
    df['date'] = pd.to_datetime(dates_arr)
    df['time_idx'] = ti
    df = add_lag_features(df, mat, date_to_idx)
    return df

def train_lgb(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='mae',
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def lgb_predict(model, df_pred):
    msk = df_pred[FEATS].notna().all(axis=1)
    p = np.full(N_SLOT, np.nan)
    X = df_pred.loc[msk, FEATS].values
    p[msk.values] = model.predict(X)
    p = np.nan_to_num(p, nan=df_pred['lag1'].values)
    return p

# ---------------- 9. EGD在线权重（OneNet核心公式） ----------------
def init_weights_load(n_train_days):
    """负载5模型冷启动：简单/统计占0.85，LGB占0.15"""
    if n_train_days < COLD_START_DAYS:
        return np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    return np.ones(len(LOAD_MODELS)) / len(LOAD_MODELS)

def init_weights_pv(n_train_days):
    """光伏5模型冷启动：Persistence强基线占0.30"""
    if n_train_days < COLD_START_DAYS:
        return np.array([0.30, 0.20, 0.15, 0.15, 0.20])
    return np.ones(len(PV_MODELS)) / len(PV_MODELS)

def egd_update(w, losses_history):
    """OneNet EGD: w_{t+1,i}=w_{t,i}*exp(-η*ℓ̃)/Z，ℓ̃为TVJMA局部核加权损失"""
    if len(losses_history) == 0:
        return w
    recent = np.array(losses_history[-EGD_WINDOW:])
    if len(recent) == 1:
        weighted_loss = recent[0]
    else:
        weights = np.array([EGD_LAMBDA**(len(recent)-1-i) for i in range(len(recent))])
        weights = weights / weights.sum()
        weighted_loss = np.average(recent, axis=0, weights=weights)
    log_w = np.log(w + 1e-12) - EGD_ETA * weighted_loss
    log_w -= log_w.max()
    w_new = np.exp(log_w)
    return w_new / w_new.sum()

def k_step_reinit(w, n_train_days, init_fn):
    """OneNet Prop.2: K步重初始化，α随数据量递减"""
    w_init = init_fn(n_train_days)
    alpha = max(0.05, 0.5 * np.exp(-(n_train_days - COLD_START_DAYS) / 60.0))
    return alpha * w_init + (1 - alpha) * w

# ---------------- 10. 模型预测调度 ----------------
def predict_all_load(mat, i_day, dates, date_to_idx, lgb_model):
    """返回(5,144)，顺序=LOAD_MODELS (SeasonalNaive, HistMean, STL, kNN, LightGBM)"""
    p = np.zeros((len(LOAD_MODELS), N_SLOT))
    p[0] = seasonal_naive(mat, i_day)
    p[1] = hist_mean(mat, i_day)
    p[2] = stl_predict(mat, i_day)
    p[3] = knn_predict(mat, i_day, dates, by_dow=True)
    if lgb_model is not None:
        df_pred = build_pred_matrix(dates[i_day], date_to_idx, mat)
        p[4] = lgb_predict(lgb_model, df_pred)
    else:
        p[4] = p[1]  # 冷启动回退HistMean
    return p

def predict_all_pv(mat, i_day, dates, date_to_idx, lgb_model):
    """返回(5,144)，顺序=PV_MODELS"""
    p = np.zeros((len(PV_MODELS), N_SLOT))
    p[0] = persistence(mat, i_day)
    p[1] = knn_predict(mat, i_day, dates, by_dow=False)
    p[2] = stl_predict(mat, i_day)
    p[3] = fpca_predict_pv(mat, i_day)
    if lgb_model is not None:
        df_pred = build_pred_matrix(dates[i_day], date_to_idx, mat)
        p[4] = lgb_predict(lgb_model, df_pred)
    else:
        p[4] = p[0]  # 冷启动回退Persistence
    return p

# ---------------- 11. 滚动预测主流程 ----------------
def rolling_predict(dates, load_mat, pv_mat, start, end):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    pred_days = pd.date_range(start, end, freq='D')
    n_pred = len(pred_days)
    n_load = len(LOAD_MODELS)
    n_pv = len(PV_MODELS)

    load_pred = np.full((n_pred, N_SLOT), np.nan)
    pv_pred = np.full((n_pred, N_SLOT), np.nan)
    load_pred_models = np.full((n_pred, n_load, N_SLOT), np.nan)
    pv_pred_models = np.full((n_pred, n_pv, N_SLOT), np.nan)
    wrec = []

    w_load = init_weights_load(31)
    w_pv = init_weights_pv(31)
    losses_load = []
    losses_pv = []
    lgb_load = None
    lgb_pv = None

    t0 = time.time()
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end_idx = i_day - 1
        n_train = train_end_idx + 1

        # 每7天重训LGB + K步重初始化
        need_retrain = (k % RETRAIN_EVERY == 0) or (lgb_load is None)
        if need_retrain and train_end_idx >= COLD_START_DAYS:
            Xl, yl, Xlv, ylv = build_train_matrix(dates, load_mat, train_end_idx, date_to_idx)
            if Xl is not None and len(Xl) > 100:
                lgb_load = train_lgb(Xl, yl, Xlv, ylv)
            Xp, yp, Xpv, ypv = build_train_matrix(dates, pv_mat, train_end_idx, date_to_idx)
            if Xp is not None and len(Xp) > 100:
                lgb_pv = train_lgb(Xp, yp, Xpv, ypv)
            w_load = k_step_reinit(w_load, n_train, init_weights_load)
            w_pv = k_step_reinit(w_pv, n_train, init_weights_pv)

        # 全模型预测
        all_load = predict_all_load(load_mat, i_day, dates, date_to_idx, lgb_load)
        all_pv = predict_all_pv(pv_mat, i_day, dates, date_to_idx, lgb_pv)
        load_pred_models[k] = all_load
        pv_pred_models[k] = all_pv

        # EGD加权集成
        comb_load = np.dot(w_load, all_load)
        comb_pv = np.dot(w_pv, all_pv)
        night = (ti_to_hour(np.arange(N_SLOT)) <= 6.0) | (ti_to_hour(np.arange(N_SLOT)) >= 20.0)
        comb_pv[night] = 0.0
        comb_pv = np.clip(comb_pv, 0, None)
        comb_load = np.clip(comb_load, 0, None)

        load_pred[k] = comb_load
        pv_pred[k] = comb_pv
        wrec.append((day, w_load.copy(), w_pv.copy(), n_train))

        # EGD权重更新
        true_load = load_mat[i_day]
        true_pv = pv_mat[i_day]
        day_losses_load = np.array([mean_absolute_error(true_load, all_load[j]) for j in range(n_load)])
        day_losses_pv = np.array([mean_absolute_error(true_pv, all_pv[j]) for j in range(n_pv)])
        losses_load.append(day_losses_load)
        losses_pv.append(day_losses_pv)
        w_load = egd_update(w_load, losses_load)
        w_pv = egd_update(w_pv, losses_pv)

        if (k+1) % 20 == 0:
            el = mean_absolute_error(true_load, comb_load)
            ep = mean_absolute_error(true_pv, comb_pv)
            print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} 第{k+1}/{n_pred}天 | '
                  f'负载MAE={el:.1f} 光伏MAE={ep:.1f} | '
                  f'w_load={np.round(w_load,3)} | w_pv={np.round(w_pv,3)} | N={n_train}')

    print(f'滚动预测完成，总耗时 {time.time()-t0:.1f}s')
    return (load_pred, pv_pred, load_pred_models, pv_pred_models, pred_days, wrec)

# ---------------- 12. 保存与验证 ----------------
def pv_mape_daytime(t, p):
    hrs = ti_to_hour(np.arange(N_SLOT))
    day = (hrs > 6.0) & (hrs < 20.0)
    t = t[:, day]; p = p[:, day]
    m = t > 50.0
    if m.sum() == 0: return np.nan
    return 100*np.mean(np.abs(t[m]-p[m])/(np.abs(t[m])+1e-6))

def save_results(pred_days, load_pred, pv_pred, load_pred_models, pv_pred_models,
                 load_mat, pv_mat, dates, wrec, tag='v5'):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    # 保存预测Excel（带日期版，格式与附件2一致）
    df_load = pd.DataFrame(load_pred, columns=TIME_COLS)
    df_load.insert(0, DATE_COL, pred_days)
    df_pv = pd.DataFrame(pv_pred, columns=TIME_COLS)
    df_pv.insert(0, DATE_COL, pred_days)
    df_load.to_excel(os.path.join(OUT_DIR, f'小区负载预测_2025_02-12_带日期_{tag}.xlsx'), index=False)
    df_pv.to_excel(os.path.join(OUT_DIR, f'光伏功率预测_2025_02-12_带日期_{tag}.xlsx'), index=False)

    # 逐模型月度误差 + 全期对比
    recs = []
    for k, day in enumerate(pred_days):
        i = date_to_idx[pd.Timestamp(day)]
        recs.append({'date': day, 'month': pd.Timestamp(day).month,
                     'lt': load_mat[i], 'pt': pv_mat[i]})
    rec = pd.DataFrame(recs)

    summary = []
    for m, g in rec.groupby('month'):
        idxs = g.index.values
        lt = np.vstack(load_mat[[date_to_idx[pd.Timestamp(d)] for d in g['date']]])
        pt = np.vstack(pv_mat[[date_to_idx[pd.Timestamp(d)] for d in g['date']]])
        lp = load_pred[idxs]; pp = pv_pred[idxs]
        summary.append({'方案': 'EGD时变集成', '月份': f'{m}月', '天数': len(g),
                        '负载MAE': mean_absolute_error(lt, lp),
                        '负载RMSE': np.sqrt(mean_squared_error(lt, lp)),
                        '负载MAPE%': 100*np.mean(np.abs(lt-lp)/(np.abs(lt)+1e-6)),
                        '光伏MAE': mean_absolute_error(pt, pp),
                        '光伏RMSE': np.sqrt(mean_squared_error(pt, pp)),
                        '光伏白天MAPE%': pv_mape_daytime(pt, pp)})
    lt_all = np.vstack(load_mat[[date_to_idx[pd.Timestamp(d)] for d in rec['date']]])
    pt_all = np.vstack(pv_mat[[date_to_idx[pd.Timestamp(d)] for d in rec['date']]])
    summary.append({'方案': 'EGD时变集成', '月份': '全期2-12月', '天数': len(rec),
                    '负载MAE': mean_absolute_error(lt_all, load_pred),
                    '负载RMSE': np.sqrt(mean_squared_error(lt_all, load_pred)),
                    '负载MAPE%': 100*np.mean(np.abs(lt_all-load_pred)/(np.abs(lt_all)+1e-6)),
                    '光伏MAE': mean_absolute_error(pt_all, pv_pred),
                    '光伏RMSE': np.sqrt(mean_squared_error(pt_all, pv_pred)),
                    '光伏白天MAPE%': pv_mape_daytime(pt_all, pv_pred)})

    # 逐模型全期误差
    model_summary = []
    for j, name in enumerate(LOAD_MODELS):
        lp_j = load_pred_models[:, j, :]
        model_summary.append({'模型': name, '对象': '负载',
                              'MAE': mean_absolute_error(lt_all, lp_j),
                              'RMSE': np.sqrt(mean_squared_error(lt_all, lp_j)),
                              'MAPE%': 100*np.mean(np.abs(lt_all-lp_j)/(np.abs(lt_all)+1e-6))})
    for j, name in enumerate(PV_MODELS):
        pp_j = pv_pred_models[:, j, :]
        model_summary.append({'模型': name, '对象': '光伏',
                              'MAE': mean_absolute_error(pt_all, pp_j),
                              'RMSE': np.sqrt(mean_squared_error(pt_all, pp_j)),
                              '白天MAPE%': pv_mape_daytime(pt_all, pp_j)})
    model_summary.append({'模型': 'EGD时变集成', '对象': '负载',
                          'MAE': mean_absolute_error(lt_all, load_pred),
                          'RMSE': np.sqrt(mean_squared_error(lt_all, load_pred)),
                          'MAPE%': 100*np.mean(np.abs(lt_all-load_pred)/(np.abs(lt_all)+1e-6))})
    model_summary.append({'模型': 'EGD时变集成', '对象': '光伏',
                          'MAE': mean_absolute_error(pt_all, pv_pred),
                          'RMSE': np.sqrt(mean_squared_error(pt_all, pv_pred)),
                          '白天MAPE%': pv_mape_daytime(pt_all, pv_pred)})

    # 模型间误差相关性矩阵（证明机制差异化）
    load_err = np.array([mean_absolute_error(lt_all, load_pred_models[:,j,:], multioutput='raw_values') for j in range(len(LOAD_MODELS))])
    pv_err = np.array([mean_absolute_error(pt_all, pv_pred_models[:,j,:], multioutput='raw_values') for j in range(len(PV_MODELS))])
    load_corr = np.corrcoef(load_err)
    pv_corr = np.corrcoef(pv_err)
    df_lcorr = pd.DataFrame(load_corr, index=LOAD_MODELS, columns=LOAD_MODELS)
    df_pcorr = pd.DataFrame(pv_corr, index=PV_MODELS, columns=PV_MODELS)

    sdf = pd.DataFrame(summary)
    mdf = pd.DataFrame(model_summary)
    with pd.ExcelWriter(os.path.join(OUT_DIR, f'验证误差报告_按月_{tag}.xlsx')) as writer:
        sdf.to_excel(writer, sheet_name='集成月度误差', index=False)
        mdf.to_excel(writer, sheet_name='逐模型全期对比', index=False)
        df_lcorr.to_excel(writer, sheet_name='负载模型误差相关性')
        df_pcorr.to_excel(writer, sheet_name='光伏模型误差相关性')

    # 权重演化
    wr_rows = []
    for day, wl, wp, nt in wrec:
        row = {'date': day, 'n_train': nt}
        for j, name in enumerate(LOAD_MODELS):
            row[f'w_load_{name}'] = wl[j]
        for j, name in enumerate(PV_MODELS):
            row[f'w_pv_{name}'] = wp[j]
        wr_rows.append(row)
    pd.DataFrame(wr_rows).to_excel(os.path.join(OUT_DIR, f'权重演化_{tag}.xlsx'), index=False)

    print('\n===== 逐模型全期误差对比 =====')
    print(mdf.round(2).to_string(index=False))
    print('\n===== EGD集成月度误差 =====')
    print(sdf.round(2).to_string(index=False))
    print('\n===== 负载模型误差相关性 =====')
    print(df_lcorr.round(3).to_string())
    print('\n===== 光伏模型误差相关性 =====')
    print(df_pcorr.round(3).to_string())
    return sdf, mdf, df_lcorr, df_pcorr

if __name__ == '__main__':
    dates, load_mat, pv_mat = load_data()
    result = rolling_predict(dates, load_mat, pv_mat, START_DAY, END_DAY)
    load_pred, pv_pred, load_pred_models, pv_pred_models, pred_days, wrec = result
    np.save(os.path.join(OUT_DIR, 'load_pred_ens_v5.npy'), load_pred)
    np.save(os.path.join(OUT_DIR, 'pv_pred_ens_v5.npy'), pv_pred)
    np.save(os.path.join(OUT_DIR, 'load_pred_models_v5.npy'), load_pred_models)
    np.save(os.path.join(OUT_DIR, 'pv_pred_models_v5.npy'), pv_pred_models)
    sdf, mdf, lcorr, pcorr = save_results(pred_days, load_pred, pv_pred,
                                              load_pred_models, pv_pred_models,
                                              load_mat, pv_mat, dates, wrec, tag='v5')
    print('\n全部完成，结果已保存到', OUT_DIR)
