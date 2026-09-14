# -*- coding: utf-8 -*-
"""
XGBoost单模型全量预测（对照组）
复用v5的特征工程，XGBoost强正则，扩展窗口+每7天重训+直接策略
与v5 EGD集成、LSTM baseline形成三组对照
"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 复用v5的数据加载、特征工程（exec到LightGBM定义之前）
exec(open(r'D:\CUMCM2026Problems\C题\P2a_forecast.py', encoding='utf-8').read().split('# ---------------- 9.')[0])

OUT = r'D:\CUMCM2026Problems\C题\预测结果'
RETRAIN_EVERY = 7

def train_xgb(X_tr, y_tr, X_va, y_va):
    """XGBoost强正则训练，早停50轮"""
    m = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        min_child_weight=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42,
        eval_metric='mae', early_stopping_rounds=50, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m

def rolling_xgb(mat, dates, date_to_idx, pred_days, is_pv=False):
    """滚动XGBoost预测：扩展窗口，每7天重训，直接策略"""
    n_pred = len(pred_days)
    preds = np.full((n_pred, N_SLOT), np.nan)
    model = None
    t0 = time.time()
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end = i_day - 1
        need_retrain = (k % RETRAIN_EVERY == 0) or (model is None)
        if need_retrain and train_end >= 35:
            X_tr, y_tr, X_va, y_va = build_train_matrix(dates, mat, train_end, date_to_idx)
            if X_tr is not None and len(X_tr) > 100:
                model = train_xgb(X_tr, y_tr, X_va, y_va)
        if model is not None:
            df_pred = build_pred_matrix(dates[i_day], date_to_idx, mat)
            p = lgb_predict(model, df_pred)  # 复用lgb_predict的预测逻辑（接口一致）
            if is_pv:
                hrs = ti_to_hour(np.arange(N_SLOT))
                p[(hrs <= 6.0) | (hrs >= 20.0)] = 0.0
            preds[k] = np.clip(p, 0, None)
        else:
            preds[k] = mat[i_day-1]  # 冷启动回退persistence
        if (k+1) % 50 == 0:
            print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} 第{k+1}/{n_pred}天 XGB MAE={mean_absolute_error(mat[i_day], preds[k]):.1f}')
    print(f'XGB滚动完成，耗时{time.time()-t0:.1f}s')
    return preds

def save_excel(preds, pred_days, name):
    """保存预测结果Excel，格式与v5一致（日期+144时点）"""
    time_cols = [f'{ti_to_hour(ti):.2f}h' for ti in range(N_SLOT)]
    df = pd.DataFrame(preds, columns=time_cols)
    df.insert(0, '日期', [pd.Timestamp(d).strftime('%Y-%m-%d') for d in pred_days])
    df.to_excel(OUT + f'\\{name}.xlsx', index=False)
    print(f'已保存 {name}.xlsx')

# ==================== 主流程 ====================
print('加载数据...')
dates, load_mat, pv_mat = load_data()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
pred_days = pd.date_range('2025-02-01', '2025-12-31', freq='D')
idx = [date_to_idx[pd.Timestamp(d)] for d in pred_days]
lt = load_mat[idx]; pt = pv_mat[idx]

print('\n===== XGBoost 负载全量预测 =====')
load_xgb = rolling_xgb(load_mat, dates, date_to_idx, pred_days, is_pv=False)
np.save(OUT + r'\load_pred_xgb.npy', load_xgb)
save_excel(load_xgb, pred_days, '小区负载预测_XGBoost对照组')

print('\n===== XGBoost 光伏全量预测 =====')
pv_xgb = rolling_xgb(pv_mat, dates, date_to_idx, pred_days, is_pv=True)
np.save(OUT + r'\pv_pred_xgb.npy', pv_xgb)
save_excel(pv_xgb, pred_days, '光伏功率预测_XGBoost对照组')

# ==================== 误差对比 ====================
print('\n===== 三组对照误差对比（全期334天） =====')

# LSTM结果
lstm_load = np.load(OUT + r'\lstm_load_pred.npy')
lstm_pv = np.load(OUT + r'\lstm_pv_pred.npy')

# v5 EGD集成
ens_load = np.load(OUT + r'\load_pred_ens_v5.npy')
ens_pv = np.load(OUT + r'\pv_pred_ens_v5.npy')

def metrics(t, p, name, is_pv=False):
    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    if is_pv:
        hrs = ti_to_hour(np.arange(N_SLOT))
        day = (hrs > 6.0) & (hrs < 20.0)
        td = t[:, day]; pd_ = p[:, day]
        m = td > 50.0
        mape = 100*np.mean(np.abs(td[m]-pd_[m])/(np.abs(td[m])+1e-6)) if m.sum() > 0 else np.nan
    else:
        mape = 100*np.mean(np.abs(t-p)/(np.abs(t)+1e-6))
    print(f'  {name:20s} MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:.2f}%')
    return mae, rmse, mape

print('\n负载：')
metrics(lt, lstm_load, 'LSTM')
metrics(lt, load_xgb, 'XGBoost')
metrics(lt, ens_load, 'v5 EGD集成(本文)')

print('\n光伏：')
metrics(pt, lstm_pv, 'LSTM', is_pv=True)
metrics(pt, pv_xgb, 'XGBoost', is_pv=True)
metrics(pt, ens_pv, 'v5 EGD集成(本文)', is_pv=True)

# 保存对比表
rows = []
for name, lp, pp in [('LSTM', lstm_load, lstm_pv), ('XGBoost', load_xgb, pv_xgb), ('v5 EGD集成(本文)', ens_load, ens_pv)]:
    lmae, lrmse, lmape = metrics(lt, lp, name)
    pmae, prmse, pmape = metrics(pt, pp, name, is_pv=True)
    rows.append({'模型': name, '负载MAE': round(lmae,2), '负载RMSE': round(lrmse,2), '负载MAPE%': round(lmape,2),
                  '光伏MAE': round(pmae,2), '光伏RMSE': round(prmse,2), '光伏白天MAPE%': round(pmape,2)})
pd.DataFrame(rows).to_excel(OUT + r'\三组对照误差对比.xlsx', index=False)
print('\n对比表已保存到 三组对照误差对比.xlsx')
print('\n全部完成。')
