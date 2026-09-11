# -*- coding: utf-8 -*-
"""参数敏感性测试：三种模型配置在2-3月滚动场景下的实际误差对比"""
import sys, time
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 载入基础函数
src = open(r'D:\CUMCM2026Problems\C题\预测_小区负载与光伏.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(src, globals())

import lightgbm as lgb

def train_cfg(X_tr, y_tr, X_va, y_va, cfg):
    if cfg == 'A':  # v1 原版：高容量无早停
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr)
    elif cfg == 'B':  # v2 强正则+早停
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='mae',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    elif cfg == 'C':  # 折中：中等容量+早停
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='mae',
              callbacks=[lgb.early_stopping(80, verbose=False)])
    return m

def rolling_cfg(cfg):
    dates, load_mat, pv_mat = load_data()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    n_days = len(dates)
    pred_days = pd.date_range('2025-02-01', '2025-03-31', freq='D')
    load_pred = np.full((len(pred_days), N_SLOT), np.nan)
    pv_pred = np.full((len(pred_days), N_SLOT), np.nan)
    err_load = {'lgb': [], 'persist': [], 'wk': []}
    err_pv = {'lgb': [], 'persist': [], 'wk': []}
    mdl_load = mdl_pv = None
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end_idx = i_day - 1
        if train_end_idx < 35:
            load_pred[k] = load_mat[i_day-1]
            pv_pred[k] = pv_mat[i_day-1]
            err_load['lgb'].append(mean_absolute_error(load_mat[i_day], load_pred[k]))
            err_load['persist'].append(err_load['lgb'][-1]); err_load['wk'].append(err_load['lgb'][-1])
            err_pv['lgb'].append(mean_absolute_error(pv_mat[i_day], pv_pred[k]))
            err_pv['persist'].append(err_pv['lgb'][-1]); err_pv['wk'].append(err_pv['lgb'][-1])
            continue
        if (k % 7 == 0) or (mdl_load is None):
            Xl, yl, Xlv, ylv = build_train_sets(dates, load_mat, train_end_idx, date_to_idx, n_days)
            mdl_load = train_cfg(Xl, yl, Xlv, ylv, cfg)
            Xp, yp, Xpv, ypv = build_train_sets(dates, pv_mat, train_end_idx, date_to_idx, n_days)
            mdl_pv = train_cfg(Xp, yp, Xpv, ypv, cfg)
        f_load = build_pred_features(day, date_to_idx, load_mat, n_days)
        f_pv = build_pred_features(day, date_to_idx, pv_mat, n_days)
        msk_l = f_load[FEATS].notna().all(axis=1); msk_p = f_pv[FEATS].notna().all(axis=1)
        p_lgb = np.full(N_SLOT, np.nan); p_lgb[msk_l.values] = mdl_load.predict(f_load.loc[msk_l, FEATS].values)
        p_lgb_pv = np.full(N_SLOT, np.nan); p_lgb_pv[msk_p.values] = mdl_pv.predict(f_pv.loc[msk_p, FEATS].values)
        # 简化：直接输出LGB预测（不含权重集成，纯比模型）
        load_pred[k] = np.nan_to_num(p_lgb, nan=load_mat[i_day-1])
        pv_pred[k] = np.clip(np.nan_to_num(p_lgb_pv, nan=pv_mat[i_day-1]), 0, None)
        night = ((np.arange(N_SLOT)*10.0/60.0) <= 6.0) | ((np.arange(N_SLOT)*10.0/60.0) >= 20.0)
        pv_pred[k, night] = 0.0
        err_load['lgb'].append(mean_absolute_error(load_mat[i_day], load_pred[k]))
        err_pv['lgb'].append(mean_absolute_error(pv_mat[i_day], pv_pred[k]))
    lt = np.vstack([load_mat[date_to_idx[pd.Timestamp(d)]] for d in pred_days])
    pt = np.vstack([pv_mat[date_to_idx[pd.Timestamp(d)]] for d in pred_days])
    return (mean_absolute_error(lt, load_pred), np.sqrt(mean_squared_error(lt, load_pred)),
            mean_absolute_error(pt, pv_pred), np.sqrt(mean_squared_error(pt, pv_pred)))

for cfg in ['A', 'B', 'C']:
    t0 = time.time()
    l_mae, l_rmse, p_mae, p_rmse = rolling_cfg(cfg)
    print(f'cfg {cfg}: 负载MAE={l_mae:.1f} RMSE={l_rmse:.1f} | 光伏MAE={p_mae:.1f} RMSE={p_rmse:.1f} | {time.time()-t0:.1f}s')
