# -*- coding: utf-8 -*-
"""
2026 C题 问题2-预测部分：小区负载与光伏发电功率滚动日前预测（v3 定稿）
- 任务：2025.2.1 - 2025.12.31 每天 0:00 用此前全部历史数据预测当日 144 个 10 分钟点
- 策略：扩展窗口 + 每 7 天全量重训（滚动）
- 主模型：LightGBM（早停 + 强正则化，最近7天作早停验证集）
- 基线：持久性（昨天同刻）、上周同刻均值
- 集成：冷启动保护 + 逆误差时变加权（纯LGB与集成双轨输出，最终对比择优）
- 防过拟合：早停、正则化、时序前向切分诊断、按月验证报告
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
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
EARLY_VAL_DAYS = 7
COLD_START_DAYS = 35      # 少于35天历史：只用简单基线（28天滞后 + 7天早停验证）

TIME_COLS = ['%02d:%02d' % ((t*10)//60, (t*10)%60) for t in range(N_SLOT)]
TIME_COLS[-1] = '0:00+1'

# ---------------- 1. 数据加载 ----------------
def load_data():
    df_l = pd.read_excel(DATA_PATH, sheet_name='小区负载')
    df_p = pd.read_excel(DATA_PATH, sheet_name='光伏发电实际功率')
    dates = pd.to_datetime(df_l.iloc[:, 0].values).values
    load_mat = df_l.iloc[:, 1:1+N_SLOT].astype(float).values
    pv_mat = df_p.iloc[:, 1:1+N_SLOT].astype(float).values
    return dates, load_mat, pv_mat

# ---------------- 2. 特征工程（向量化） ----------------
def make_calendar_features(dates_arr, time_idx_arr):
    dts = pd.to_datetime(dates_arr)
    n = len(dts)
    hour = time_idx_arr * 10.0 / 60.0
    dow = dts.dayofweek.values.astype(float)
    df = pd.DataFrame({
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
    return df

def add_lag_features_vectorized(df, mat, date_to_idx, n_days):
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

# ---------------- 3. 模型（cfg B：早停 + 强正则） ----------------
def train_lgb_early(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

# ---------------- 4. 训练/预测集构造 ----------------
def build_train_sets(dates, mat, train_end_idx, date_to_idx, n_days):
    va_start = train_end_idx - EARLY_VAL_DAYS + 1
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
        df = add_lag_features_vectorized(df, mat, date_to_idx, n_days)
        df['target'] = mat[di, ti]
        msk = df[FEATS].notna().all(axis=1)
        return df.loc[msk, FEATS].values, df.loc[msk, 'target'].values
    X_tr, y_tr = _build(28, tr_end)
    X_va, y_va = _build(va_start, train_end_idx)
    if X_tr is None or X_va is None or len(X_tr) < 100 or len(X_va) < 50:
        return None, None, None, None
    return X_tr, y_tr, X_va, y_va

def build_pred_features(day, date_to_idx, mat, n_days):
    ti = np.arange(N_SLOT)
    dates_arr = np.full(N_SLOT, np.datetime64(day), dtype='datetime64[D]')
    df = make_calendar_features(dates_arr, ti)
    df['date'] = pd.to_datetime(dates_arr)
    df['time_idx'] = ti
    df = add_lag_features_vectorized(df, mat, date_to_idx, n_days)
    return df

# ---------------- 5. 滚动预测主流程 ----------------
def rolling_predict(dates, load_mat, pv_mat, start, end, retrain_every=7):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    n_days = len(dates)
    pred_days = pd.date_range(start, end, freq='D')
    n_pred = len(pred_days)
    load_pred_ens = np.full((n_pred, N_SLOT), np.nan)   # 加权集成
    pv_pred_ens = np.full((n_pred, N_SLOT), np.nan)
    load_pred_lgb = np.full((n_pred, N_SLOT), np.nan)   # 纯LGB
    pv_pred_lgb = np.full((n_pred, N_SLOT), np.nan)
    wrec = []

    err_load = {'lgb': [], 'persist': [], 'wk': []}
    err_pv = {'lgb': [], 'persist': [], 'wk': []}
    mdl_load = None
    mdl_pv = None

    t0 = time.time()
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end_idx = i_day - 1
        N = train_end_idx + 1

        # ---- 冷启动期：只用简单基线 ----
        if train_end_idx < COLD_START_DAYS:
            p_persist = load_mat[i_day-1, :].copy()
            p_wk = load_mat[i_day-7, :].copy() if i_day-7 >= 0 else p_persist
            comb_l = 0.5*p_persist + 0.5*p_wk
            p_persist_pv = pv_mat[i_day-1, :].copy()
            p_wk_pv = pv_mat[i_day-7, :].copy() if i_day-7 >= 0 else p_persist_pv
            comb_p = np.clip(0.5*p_persist_pv + 0.5*p_wk_pv, 0, None)
            load_pred_ens[k] = comb_l; pv_pred_ens[k] = comb_p
            load_pred_lgb[k] = comb_l; pv_pred_lgb[k] = comb_p
            wrec.append((day, 0.0, 0.0, N))
            err_load['lgb'].append(mean_absolute_error(load_mat[i_day], comb_l))
            err_load['persist'].append(mean_absolute_error(load_mat[i_day], p_persist))
            err_load['wk'].append(mean_absolute_error(load_mat[i_day], p_wk))
            err_pv['lgb'].append(mean_absolute_error(pv_mat[i_day], comb_p))
            err_pv['persist'].append(mean_absolute_error(pv_mat[i_day], p_persist_pv))
            err_pv['wk'].append(mean_absolute_error(pv_mat[i_day], p_wk_pv))
            continue

        need_retrain = (k % retrain_every == 0) or (mdl_load is None)
        if need_retrain:
            Xl, yl, Xlv, ylv = build_train_sets(dates, load_mat, train_end_idx, date_to_idx, n_days)
            if Xl is not None:
                mdl_load = train_lgb_early(Xl, yl, Xlv, ylv)
            Xp, yp, Xpv, ypv = build_train_sets(dates, pv_mat, train_end_idx, date_to_idx, n_days)
            if Xp is not None:
                mdl_pv = train_lgb_early(Xp, yp, Xpv, ypv)

        f_load = build_pred_features(day, date_to_idx, load_mat, n_days)
        f_pv = build_pred_features(day, date_to_idx, pv_mat, n_days)
        msk_l = f_load[FEATS].notna().all(axis=1)
        msk_p = f_pv[FEATS].notna().all(axis=1)
        p_lgb = np.full(N_SLOT, np.nan)
        p_lgb[msk_l.values] = mdl_load.predict(f_load.loc[msk_l, FEATS].values)
        p_lgb_pv = np.full(N_SLOT, np.nan)
        p_lgb_pv[msk_p.values] = mdl_pv.predict(f_pv.loc[msk_p, FEATS].values)

        p_persist = load_mat[i_day-1, :].copy()
        p_wk = load_mat[i_day-7, :].copy() if i_day-7 >= 0 else p_persist
        p_persist_pv = pv_mat[i_day-1, :].copy()
        p_wk_pv = pv_mat[i_day-7, :].copy() if i_day-7 >= 0 else p_persist_pv

        # 模型不可用（训练失败）时回退简单基线
        if mdl_load is None or mdl_pv is None:
            load_pred_ens[k] = 0.5*p_persist + 0.5*p_wk
            load_pred_lgb[k] = 0.5*p_persist + 0.5*p_wk
            comb_p = np.clip(0.5*p_persist_pv + 0.5*p_wk_pv, 0, None)
            pv_pred_ens[k] = comb_p
            pv_pred_lgb[k] = comb_p
            wrec.append((day, 0.0, 0.0, N))
            err_load['lgb'].append(mean_absolute_error(load_mat[i_day], load_pred_ens[k]))
            err_load['persist'].append(mean_absolute_error(load_mat[i_day], p_persist))
            err_load['wk'].append(mean_absolute_error(load_mat[i_day], p_wk))
            err_pv['lgb'].append(mean_absolute_error(pv_mat[i_day], comb_p))
            err_pv['persist'].append(mean_absolute_error(pv_mat[i_day], p_persist_pv))
            err_pv['wk'].append(mean_absolute_error(pv_mat[i_day], p_wk_pv))
            continue

        # ---- 时变权重：启发式S型 + 逆误差修正 ----
        def calc_weights(err_dict, N):
            w_base = 1.0 / (1.0 + np.exp(-(N - 45) / 15.0))
            if len(err_dict['lgb']) >= 15:
                e_ml = np.mean(err_dict['lgb'][-30:]); e_ps = np.mean(err_dict['persist'][-30:]); e_wk = np.mean(err_dict['wk'][-30:])
                inv = np.array([1.0/(e_ml+1e-6), 1.0/(e_ps+1e-6), 1.0/(e_wk+1e-6)])
                inv = inv / inv.sum()
                w_ml = float(np.clip(0.6*w_base + 0.4*inv[0], 0, 0.95))
            else:
                w_ml = float(np.clip(w_base, 0, 0.95))
            w_ps = float(np.clip((1-w_ml)*0.6, 0, 1))
            w_wk = float(np.clip(1-w_ml-w_ps, 0, 1))
            return w_ml, w_ps, w_wk

        w_ml, w_ps, w_wk = calc_weights(err_load, N)
        comb_l = w_ml*np.nan_to_num(p_lgb, nan=p_persist) + w_ps*p_persist + w_wk*p_wk
        load_pred_ens[k] = comb_l
        load_pred_lgb[k] = np.nan_to_num(p_lgb, nan=p_persist)

        w_ml_pv, w_ps_pv, w_wk_pv = calc_weights(err_pv, N)
        comb_p = w_ml_pv*np.nan_to_num(p_lgb_pv, nan=p_persist_pv) + w_ps_pv*p_persist_pv + w_wk_pv*p_wk_pv
        night = ((np.arange(N_SLOT)*10.0/60.0) <= 6.0) | ((np.arange(N_SLOT)*10.0/60.0) >= 20.0)
        comb_p[night] = 0.0
        comb_p = np.clip(comb_p, 0, None)
        pv_pred_ens[k] = comb_p
        pp_lgb = np.nan_to_num(p_lgb_pv, nan=p_persist_pv)
        pp_lgb[night] = 0.0
        pv_pred_lgb[k] = np.clip(pp_lgb, 0, None)

        wrec.append((day, w_ml, w_ml_pv, N))

        err_load['lgb'].append(mean_absolute_error(load_mat[i_day], np.nan_to_num(p_lgb, nan=p_persist)))
        err_load['persist'].append(mean_absolute_error(load_mat[i_day], p_persist))
        err_load['wk'].append(mean_absolute_error(load_mat[i_day], p_wk))
        err_pv['lgb'].append(mean_absolute_error(pv_mat[i_day], np.nan_to_num(p_lgb_pv, nan=p_persist_pv)))
        err_pv['persist'].append(mean_absolute_error(pv_mat[i_day], p_persist_pv))
        err_pv['wk'].append(mean_absolute_error(pv_mat[i_day], p_wk_pv))

        if (k+1) % 40 == 0:
            el = mean_absolute_error(load_mat[i_day], comb_l)
            ep = mean_absolute_error(pv_mat[i_day], comb_p)
            print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} 第{k+1}/{n_pred}天 | 负载MAE={el:.1f} 光伏MAE={ep:.1f} | w_ml={w_ml:.2f}/{w_ml_pv:.2f} | N={N}天')

    print(f'滚动预测完成，总耗时 {time.time()-t0:.1f}s')
    return (load_pred_ens, pv_pred_ens, load_pred_lgb, pv_pred_lgb, pred_days, wrec)

# ---------------- 6. 保存与验证 ----------------
def save_results(pred_days, ens, lgbp, load_mat, pv_mat, dates, tag=''):
    load_pred_ens, pv_pred_ens, load_pred_lgb, pv_pred_lgb = ens[0], ens[1], lgbp[0], lgbp[1]
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    recs = []
    for k, day in enumerate(pred_days):
        i = date_to_idx[pd.Timestamp(day)]
        recs.append((day, load_mat[i], load_pred_ens[k], load_pred_lgb[k],
                     pv_mat[i], pv_pred_ens[k], pv_pred_lgb[k]))
    rec = pd.DataFrame(recs, columns=['date','lt','lp_ens','lp_lgb','pt','pp_ens','pp_lgb'])
    rec['month'] = rec['date'].dt.month

    def metrics(t, p):
        return (mean_absolute_error(t, p), np.sqrt(mean_squared_error(t, p)),
                100*np.mean(np.abs(t-p)/(np.abs(t)+1e-6)))
    def pv_metrics_daytime(t, p):
        """光伏MAPE仅统计真实值>50kW的白天点，避免接近0的真实值导致失真"""
        hrs = (np.arange(N_SLOT)*10.0/60.0)
        day = (hrs > 6.0) & (hrs < 20.0)
        t = t[:, day]; p = p[:, day]
        m = t > 50.0
        t = t[m]; p = p[m]
        if len(t) == 0:
            return np.nan, np.nan, np.nan
        return (mean_absolute_error(t, p), np.sqrt(mean_squared_error(t, p)),
                100*np.mean(np.abs(t-p)/(np.abs(t)+1e-6)))
    def row_of(g, lt_col, lp_col, pt_col, pp_col, name):
        lt = np.vstack(g[lt_col].values); lp = np.vstack(g[lp_col].values)
        pt = np.vstack(g[pt_col].values); pp = np.vstack(g[pp_col].values)
        a,b,c = metrics(lt, lp)
        d,e,f = pv_metrics_daytime(pt, pp)
        return {'方案': name, '月份': f'{m}月', '天数': len(g),
                '负载MAE': a, '负载RMSE': b, '负载MAPE%': c,
                '光伏MAE': d, '光伏RMSE': e, '光伏MAPE%(白天)%': f}

    summary = []
    for m, g in rec.groupby('month'):
        summary.append(row_of(g, 'lt','lp_ens','pt','pp_ens', '加权集成'))
        summary.append(row_of(g, 'lt','lp_lgb','pt','pp_lgb', '纯LGB'))
    # 全期
    summary.append(row_of(rec, 'lt','lp_ens','pt','pp_ens', '加权集成'))
    summary.append(row_of(rec, 'lt','lp_lgb','pt','pp_lgb', '纯LGB'))
    sdf = pd.DataFrame(summary)
    sdf.to_excel(os.path.join(OUT_DIR, f'验证误差报告_按月{tag}.xlsx'), index=False)
    return sdf

if __name__ == '__main__':
    dates, load_mat, pv_mat = load_data()
    ens = rolling_predict(dates, load_mat, pv_mat, START_DAY, END_DAY, RETRAIN_EVERY)
    load_pred_ens, pv_pred_ens, load_pred_lgb, pv_pred_lgb, pred_days, wrec = ens

    np.save(os.path.join(OUT_DIR, 'load_pred_ens.npy'), load_pred_ens)
    np.save(os.path.join(OUT_DIR, 'pv_pred_ens.npy'), pv_pred_ens)

    # 保存最终Excel：负载=纯LGB（更优）；光伏=加权集成（更优，夜间置0+逆误差权重）
    df_load = pd.DataFrame(load_pred_lgb, columns=TIME_COLS)
    df_load.insert(0, '日期', pred_days)
    df_pv = pd.DataFrame(pv_pred_ens, columns=TIME_COLS)
    df_pv.insert(0, '日期', pred_days)
    df_load.to_excel(os.path.join(OUT_DIR, '小区负载预测_2025_02-12.xlsx'), index=False)
    df_pv.to_excel(os.path.join(OUT_DIR, '光伏功率预测_2025_02-12.xlsx'), index=False)

    sdf = save_results(pred_days, (load_pred_ens, pv_pred_ens, None, None),
                       (load_pred_lgb, pv_pred_lgb, None, None), load_mat, pv_mat, dates)
    print('\n===== 验证误差报告（加权集成 vs 纯LGB，按真实值评估） =====')
    print(sdf.round(2).to_string(index=False))

    wr = pd.DataFrame(wrec, columns=['date','w_ml_load','w_ml_pv','n_train'])
    wr.to_excel(os.path.join(OUT_DIR, '权重演化.xlsx'), index=False)
    print('\n全部完成，结果已保存到', OUT_DIR)
