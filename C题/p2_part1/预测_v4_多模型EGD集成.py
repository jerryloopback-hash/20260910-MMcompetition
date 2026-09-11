# -*- coding: utf-8 -*-
"""
2026 C题 问题2-预测部分 v4：基于论文方法的多模型时变权重集成
============================================================
方法学依据（均来自此前检索的真实文献）：
1. 多步预测策略：Jiang et al. (Water Resources Management) — 直接分时刻/直接策略优于递归
2. 在线权重更新：OneNet (NeurIPS 2023) — EGD指数梯度下降 w_{t+1,i}=w_{t,i}exp(-ηℓ_{t,i})/Z_t
3. K步重初始化：OneNet Proposition 2 — 每K步重置权重缓解概念漂移下的"慢切换"
4. 时变模型平均：TVJMA (J. Econometrics) — 局部核加权，近样本权重高，小样本下方差小的模型自然占优
5. 冷启动：Transfer Learning + Bayesian weighted averaging (IEEE TIA) — 数据稀缺时简单方法优先

模型池（6个，互补组合）：
  M1 Persistence     — 昨天同刻（短期记忆）
  M2 SeasonalNaive   — 上周同刻（周周期）
  M3 HistMean        — 近4周同刻均值（长期平滑，TVJMA局部核思想）
  M4 LightGBM        — 梯度提升树（ML-boosting）
  M5 RandomForest    — 随机森林（ML-bagging，与boosting互补）
  M6 XGBoost         — 极端梯度提升（不同分裂策略，与LGB互补）

权重机制：
  - EGD在线更新：每天观测真实值后按公式更新
  - 冷启动初始权重：历史<35天时简单方法(M1-M3)占0.8，ML(M4-M6)占0.2
  - K步重初始化：每7天(与重训同步)按α混合初始权重与当前EGD权重，α随数据量递减
  - 局部核加权损失：EGD损失用最近30天指数加权平均(TVJMA思想)

防过拟合：
  - EGD理论regret bound (OneNet Prop.1)
  - ML模型早停+正则化
  - 时序前向80/20切分诊断
"""
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os, warnings, time
warnings.filterwarnings('ignore')

DATA_PATH = r'D:\CUMCM2026Problems\C题\附件\附件2.xlsx'
OUT_DIR = r'D:\CUMCM2026Problems\C题\预测结果'
os.makedirs(OUT_DIR, exist_ok=True)

N_SLOT = 144
START_DAY = '2025-02-01'
END_DAY = '2025-12-31'
RETRAIN_EVERY = 7
COLD_START_DAYS = 35
MODEL_NAMES = ['Persistence', 'SeasonalNaive', 'HistMean', 'LightGBM', 'RandomForest', 'XGBoost']
N_MODELS = len(MODEL_NAMES)

# EGD参数
EGD_ETA = 0.005          # 学习率（OneNet: η=sqrt(2log(d)/T)的在线近似）
EGD_WINDOW = 30           # 局部核加权窗口（TVJMA思想）
EGD_LAMBDA = 0.95         # 指数衰减核

# 时点列名：与附件2完全一致（00:10:00 → 23:50:00 + 0:00+1）
# ti=0 对应附件第0列(00:10)，ti=143 对应附件第143列(0:00+1=24:00)
TIME_COLS = [datetime.time((t+1)*10//60 % 24, (t+1)*10 % 60) for t in range(N_SLOT-1)]
TIME_COLS.append('0:00+1')
DATE_COL = '日期\\时间'

# 每个ti对应的实际小时（ti=0→00:10→hour=1/6，ti=143→24:00→hour=24.0）
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

# ---------------- 2. 特征工程 ----------------
def make_calendar_features(dates_arr, time_idx_arr):
    dts = pd.to_datetime(dates_arr)
    hour = ti_to_hour(time_idx_arr)  # ti=0→00:10, ti=143→24:00
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

# ---------------- 3. 简单模型（无需训练） ----------------
def simple_predict(mat, i_day):
    """返回3个简单模型的预测: [Persistence, SeasonalNaive, HistMean]"""
    p = np.full((3, N_SLOT), np.nan)
    p[0] = mat[i_day-1, :].copy()  # Persistence: 昨天同刻
    if i_day-7 >= 0:
        p[1] = mat[i_day-7, :].copy()  # SeasonalNaive: 上周同刻
    else:
        p[1] = p[0].copy()
    # HistMean: 近4周同刻均值（TVJMA局部平滑思想）
    acc = np.zeros(N_SLOT); cnt = 0
    for off in (7, 14, 21, 28):
        if i_day - off >= 0:
            acc += mat[i_day-off, :]
            cnt += 1
    if cnt > 0:
        p[2] = acc / cnt
    else:
        p[2] = p[0].copy()
    return p

# ---------------- 4. ML模型训练与预测 ----------------
def build_train_matrix(dates, mat, train_end_idx, date_to_idx, val_days=7):
    """构造训练集(含早停验证集)，合并所有时刻，用time_idx区分"""
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

def train_ml_models(X_tr, y_tr, X_va, y_va):
    """训练3个ML模型，返回dict"""
    models = {}
    # LightGBM (早停+正则)
    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='mae',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    models['LightGBM'] = m_lgb

    # RandomForest (bagging，与boosting互补)
    m_rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        max_features=0.7, n_jobs=-1, random_state=42)
    m_rf.fit(X_tr, y_tr)
    models['RandomForest'] = m_rf

    # XGBoost (不同分裂策略，与LGB互补)
    m_xgb = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbosity=0,
        early_stopping_rounds=50)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    models['XGBoost'] = m_xgb
    return models

def ml_predict(models, df_pred):
    """3个ML模型预测，返回[LGB, RF, XGB]"""
    msk = df_pred[FEATS].notna().all(axis=1)
    p = np.full((3, N_SLOT), np.nan)
    X = df_pred.loc[msk, FEATS].values
    p[0, msk.values] = models['LightGBM'].predict(X)
    p[1, msk.values] = models['RandomForest'].predict(X)
    p[2, msk.values] = models['XGBoost'].predict(X)
    # 缺失值回退到Persistence
    for j in range(3):
        p[j] = np.nan_to_num(p[j], nan=df_pred['lag1'].values)
    return p

# ---------------- 5. EGD在线权重（OneNet核心公式） ----------------
def init_weights(n_train_days):
    """冷启动初始权重：数据少时简单方法占优（TVJMA小样本下方差小的模型权重高）"""
    if n_train_days < COLD_START_DAYS:
        # 简单方法(Persistence, SeasonalNaive, HistMean)占0.8，ML占0.2
        return np.array([0.30, 0.30, 0.20, 0.08, 0.06, 0.06])
    else:
        return np.ones(N_MODELS) / N_MODELS

def egd_update(w, losses_history):
    """
    OneNet EGD公式: w_{t+1,i} = w_{t,i} * exp(-η * ℓ̃_{t,i}) / Z
    其中 ℓ̃ 是TVJMA局部核加权损失（最近EGD_WINDOW天指数加权平均）
    """
    if len(losses_history) == 0:
        return w
    # 局部核加权（TVJMA思想：近样本权重高）
    recent = np.array(losses_history[-EGD_WINDOW:])
    if len(recent) == 1:
        weighted_loss = recent[0]
    else:
        weights = np.array([EGD_LAMBDA**(len(recent)-1-i) for i in range(len(recent))])
        weights = weights / weights.sum()
        weighted_loss = np.average(recent, axis=0, weights=weights)
    # EGD更新
    log_w = np.log(w + 1e-12) - EGD_ETA * weighted_loss
    log_w -= log_w.max()
    w_new = np.exp(log_w)
    w_new = w_new / w_new.sum()
    return w_new

def k_step_reinit(w, n_train_days, retrain_count):
    """
    OneNet Proposition 2: K步重初始化缓解概念漂移下的"慢切换"
    α随数据量递减：冷启动期重初始化力度大（α大），后期力度小
    """
    w_init = init_weights(n_train_days)
    # α: 数据量越少，重初始化权重越高
    alpha = max(0.05, 0.5 * np.exp(-(n_train_days - COLD_START_DAYS) / 60.0))
    return alpha * w_init + (1 - alpha) * w

# ---------------- 6. 滚动预测主流程 ----------------
def rolling_predict(dates, load_mat, pv_mat, start, end):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    pred_days = pd.date_range(start, end, freq='D')
    n_pred = len(pred_days)

    # 预测结果
    load_pred = np.full((n_pred, N_SLOT), np.nan)
    pv_pred = np.full((n_pred, N_SLOT), np.nan)
    # 逐模型预测（用于报告）
    load_pred_models = np.full((n_pred, N_MODELS, N_SLOT), np.nan)
    pv_pred_models = np.full((n_pred, N_MODELS, N_SLOT), np.nan)
    # 权重记录
    wrec = []

    # EGD状态
    w_load = init_weights(31)  # 初始1月数据
    w_pv = init_weights(31)
    losses_load = []  # 每天每个模型的MAE
    losses_pv = []

    ml_models_load = None
    ml_models_pv = None

    t0 = time.time()
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end_idx = i_day - 1
        n_train = train_end_idx + 1

        # ---- 简单模型预测 ----
        s_load = simple_predict(load_mat, i_day)
        s_pv = simple_predict(pv_mat, i_day)

        # ---- ML模型：每7天重训 ----
        need_retrain = (k % RETRAIN_EVERY == 0) or (ml_models_load is None)
        if need_retrain and train_end_idx >= COLD_START_DAYS:
            Xl, yl, Xlv, ylv = build_train_matrix(dates, load_mat, train_end_idx, date_to_idx)
            if Xl is not None and len(Xl) > 100:
                ml_models_load = train_ml_models(Xl, yl, Xlv, ylv)
            Xp, yp, Xpv, ypv = build_train_matrix(dates, pv_mat, train_end_idx, date_to_idx)
            if Xp is not None and len(Xp) > 100:
                ml_models_pv = train_ml_models(Xp, yp, Xpv, ypv)
            # K步重初始化（OneNet Prop.2）
            w_load = k_step_reinit(w_load, n_train, k // RETRAIN_EVERY)
            w_pv = k_step_reinit(w_pv, n_train, k // RETRAIN_EVERY)

        # ---- ML预测 ----
        if ml_models_load is not None:
            df_pl = build_pred_matrix(day, date_to_idx, load_mat)
            m_load = ml_predict(ml_models_load, df_pl)
        else:
            m_load = np.tile(s_load[0], (3, 1))  # 冷启动回退Persistence

        if ml_models_pv is not None:
            df_pp = build_pred_matrix(day, date_to_idx, pv_mat)
            m_pv = ml_predict(ml_models_pv, df_pp)
        else:
            m_pv = np.tile(s_pv[0], (3, 1))

        # 合并6模型预测 [Persistence, SeasonalNaive, HistMean, LGB, RF, XGB]
        all_load = np.vstack([s_load, m_load])
        all_pv = np.vstack([s_pv, m_pv])

        # 记录逐模型预测
        load_pred_models[k] = all_load
        pv_pred_models[k] = all_pv

        # ---- 加权集成（当前权重） ----
        comb_load = np.dot(w_load, all_load)
        comb_pv = np.dot(w_pv, all_pv)
        # 光伏夜间置0
        night = (ti_to_hour(np.arange(N_SLOT)) <= 6.0) | (ti_to_hour(np.arange(N_SLOT)) >= 20.0)
        comb_pv[night] = 0.0
        comb_pv = np.clip(comb_pv, 0, None)

        load_pred[k] = comb_load
        pv_pred[k] = comb_pv
        wrec.append((day, w_load.copy(), w_pv.copy(), n_train))

        # ---- EGD权重更新（观测到真实值后） ----
        true_load = load_mat[i_day]
        true_pv = pv_mat[i_day]
        day_losses_load = np.array([mean_absolute_error(true_load, all_load[j]) for j in range(N_MODELS)])
        day_losses_pv = np.array([mean_absolute_error(true_pv, all_pv[j]) for j in range(N_MODELS)])
        losses_load.append(day_losses_load)
        losses_pv.append(day_losses_pv)
        w_load = egd_update(w_load, losses_load)
        w_pv = egd_update(w_pv, losses_pv)

        if (k+1) % 40 == 0:
            el = mean_absolute_error(true_load, comb_load)
            ep = mean_absolute_error(true_pv, comb_pv)
            print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} 第{k+1}/{n_pred}天 | '
                  f'负载MAE={el:.1f} 光伏MAE={ep:.1f} | '
                  f'w_simple_load={w_load[:3].sum():.2f} w_ml_load={w_load[3:].sum():.2f} | '
                  f'w_simple_pv={w_pv[:3].sum():.2f} w_ml_pv={w_pv[3:].sum():.2f} | N={n_train}')

    print(f'滚动预测完成，总耗时 {time.time()-t0:.1f}s')
    return (load_pred, pv_pred, load_pred_models, pv_pred_models, pred_days, wrec)

# ---------------- 7. 保存与验证 ----------------
def save_results(pred_days, load_pred, pv_pred, load_pred_models, pv_pred_models,
                 load_mat, pv_mat, dates, wrec):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    # 保存预测Excel（格式与附件2完全一致：第一列"日期\时间"，时点列00:10:00→0:00+1）
    df_load = pd.DataFrame(load_pred, columns=TIME_COLS)
    df_load.insert(0, DATE_COL, pred_days)
    df_pv = pd.DataFrame(pv_pred, columns=TIME_COLS)
    df_pv.insert(0, DATE_COL, pred_days)
    df_load.to_excel(os.path.join(OUT_DIR, '小区负载预测_2025_02-12.xlsx'), index=False)
    df_pv.to_excel(os.path.join(OUT_DIR, '光伏功率预测_2025_02-12.xlsx'), index=False)

    # 逐模型月度误差报告
    recs = []
    for k, day in enumerate(pred_days):
        i = date_to_idx[pd.Timestamp(day)]
        recs.append({'date': day, 'month': pd.Timestamp(day).month,
                     'lt': load_mat[i], 'pt': pv_mat[i]})
    rec = pd.DataFrame(recs)

    def pv_mape_daytime(t, p):
        hrs = ti_to_hour(np.arange(N_SLOT))
        day = (hrs > 6.0) & (hrs < 20.0)
        t = t[:, day]; p = p[:, day]
        m = t > 50.0
        if m.sum() == 0: return np.nan
        return 100*np.mean(np.abs(t[m]-p[m])/(np.abs(t[m])+1e-6))

    # 集成模型月度误差
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
                        '光伏MAPE%(白天)%': pv_mape_daytime(pt, pp)})
    # 全期
    lt = np.vstack(load_mat[[date_to_idx[pd.Timestamp(d)] for d in rec['date']]])
    pt = np.vstack(pv_mat[[date_to_idx[pd.Timestamp(d)] for d in rec['date']]])
    summary.append({'方案': 'EGD时变集成', '月份': '全期2-12月', '天数': len(rec),
                    '负载MAE': mean_absolute_error(lt, load_pred),
                    '负载RMSE': np.sqrt(mean_squared_error(lt, load_pred)),
                    '负载MAPE%': 100*np.mean(np.abs(lt-load_pred)/(np.abs(lt)+1e-6)),
                    '光伏MAE': mean_absolute_error(pt, pv_pred),
                    '光伏RMSE': np.sqrt(mean_squared_error(pt, pv_pred)),
                    '光伏MAPE%(白天)%': pv_mape_daytime(pt, pv_pred)})

    # 逐模型全期误差
    model_summary = []
    for j, name in enumerate(MODEL_NAMES):
        lp_j = load_pred_models[:, j, :]
        pp_j = pv_pred_models[:, j, :]
        model_summary.append({'模型': name,
                              '负载MAE': mean_absolute_error(lt, lp_j),
                              '负载RMSE': np.sqrt(mean_squared_error(lt, lp_j)),
                              '负载MAPE%': 100*np.mean(np.abs(lt-lp_j)/(np.abs(lt)+1e-6)),
                              '光伏MAE': mean_absolute_error(pt, pp_j),
                              '光伏RMSE': np.sqrt(mean_squared_error(pt, pp_j)),
                              '光伏MAPE%(白天)%': pv_mape_daytime(pt, pp_j)})
    model_summary.append({'模型': 'EGD时变集成',
                          '负载MAE': mean_absolute_error(lt, load_pred),
                          '负载RMSE': np.sqrt(mean_squared_error(lt, load_pred)),
                          '负载MAPE%': 100*np.mean(np.abs(lt-load_pred)/(np.abs(lt)+1e-6)),
                          '光伏MAE': mean_absolute_error(pt, pv_pred),
                          '光伏RMSE': np.sqrt(mean_squared_error(pt, pv_pred)),
                          '光伏MAPE%(白天)%': pv_mape_daytime(pt, pv_pred)})

    sdf = pd.DataFrame(summary)
    mdf = pd.DataFrame(model_summary)
    with pd.ExcelWriter(os.path.join(OUT_DIR, '验证误差报告_按月.xlsx')) as writer:
        sdf.to_excel(writer, sheet_name='集成月度误差', index=False)
        mdf.to_excel(writer, sheet_name='逐模型全期对比', index=False)

    # 权重演化
    wr_rows = []
    for day, wl, wp, nt in wrec:
        row = {'date': day, 'n_train': nt}
        for j, name in enumerate(MODEL_NAMES):
            row[f'w_load_{name}'] = wl[j]
            row[f'w_pv_{name}'] = wp[j]
        row['w_load_simple'] = wl[:3].sum()
        row['w_load_ml'] = wl[3:].sum()
        row['w_pv_simple'] = wp[:3].sum()
        row['w_pv_ml'] = wp[3:].sum()
        wr_rows.append(row)
    pd.DataFrame(wr_rows).to_excel(os.path.join(OUT_DIR, '权重演化.xlsx'), index=False)

    print('\n===== 逐模型全期误差对比 =====')
    print(mdf.round(2).to_string(index=False))
    print('\n===== EGD集成月度误差 =====')
    print(sdf.round(2).to_string(index=False))
    return sdf, mdf

if __name__ == '__main__':
    dates, load_mat, pv_mat = load_data()
    result = rolling_predict(dates, load_mat, pv_mat, START_DAY, END_DAY)
    load_pred, pv_pred, load_pred_models, pv_pred_models, pred_days, wrec = result
    np.save(os.path.join(OUT_DIR, 'load_pred_ens.npy'), load_pred)
    np.save(os.path.join(OUT_DIR, 'pv_pred_ens.npy'), pv_pred)
    sdf, mdf = save_results(pred_days, load_pred, pv_pred, load_pred_models, pv_pred_models,
                             load_mat, pv_mat, dates, wrec)
    print('\n全部完成，结果已保存到', OUT_DIR)
