# -*- coding: utf-8 -*-
"""
问题4电价预测实验v2：5成员池 [SeasonalNaive, HistMean, STL(限窗), kNN, ShapeLevel(lag7)]
EGD η 按电价损失量级重标定(15)，验证集成增益与权重演化。
"""
import importlib.util, sys, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')
spec = importlib.util.spec_from_file_location('v5', r'D:\CUMCM2026Problems\C题\预测_v5_扩展模型池.py')
v5 = importlib.util.module_from_spec(spec); sys.modules['v5'] = v5; spec.loader.exec_module(v5)

OUT = r'D:\CUMCM2026Problems\C题\p4'
a4 = pd.read_excel(r'D:\CUMCM2026Problems\C题\附件\附件4.xlsx', index_col=0)
a4.columns = range(v5.N_SLOT)
price_mat = a4.values.astype(float)
dates = pd.to_datetime(a4.index).values.astype('datetime64[D]')
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
a1 = pd.read_excel(r'D:\CUMCM2026Problems\C题\附件\附件1.xlsx')
shape144 = a1['电价'].values.astype(float); shape_mean = shape144.mean()
lev = price_mat.mean(axis=1)

n_pred_days = pd.date_range('2025-02-01', '2025-12-31', freq='D')
n_pred = len(n_pred_days)
POOL = ['SeasonalNaive', 'HistMean', 'STL', 'kNN', 'ShapeLevel']
ETA = 15.0          # 电价损失~0.045，η×loss≈0.7/天，与v5负载口径(0.005×150)等效
STL_WIN = 180       # STL限窗加速（v5为全历史）

def stl_predict_win(mat, i_day, period=7):
    pred = np.zeros(v5.N_SLOT)
    lo = max(0, i_day - STL_WIN)
    for ti in range(v5.N_SLOT):
        series = mat[lo:i_day, ti].astype(float)
        n = len(series)
        if n < 2*period + 3:
            pred[ti] = np.mean(series[-28:]) if n >= 7 else (series[-1] if n > 0 else 0.0)
            continue
        try:
            res = STL(series, period=period, seasonal=7, trend=15, robust=True).fit()
            trend = res.trend
            slope = (trend[-1] - trend[-8]) / 7.0
            pred[ti] = trend[-1] + 0.8*slope + res.seasonal[-period]
        except Exception:
            pred[ti] = np.mean(series[-28:])
    return pred

def shape_level_lag7(i_day):
    if i_day < 7: return shape144
    return shape144 * (lev[i_day-7] / shape_mean)

def predict_pool(i_day):
    p = np.zeros((len(POOL), v5.N_SLOT))
    p[0] = v5.seasonal_naive(price_mat, i_day)
    p[1] = v5.hist_mean(price_mat, i_day)
    p[2] = stl_predict_win(price_mat, i_day)
    p[3] = v5.knn_predict(price_mat, i_day, dates, by_dow=True)
    p[4] = shape_level_lag7(i_day)
    return np.clip(p, 0.01, None)

w = np.ones(len(POOL)) / len(POOL)
losses = []
pred_pool = np.full((n_pred, len(POOL), v5.N_SLOT), np.nan)
pred_ens = np.full((n_pred, v5.N_SLOT), np.nan)
wrec = []
t0 = time.time()
for k, day in enumerate(n_pred_days):
    i_day = date_to_idx[pd.Timestamp(day)]
    allp = predict_pool(i_day)
    pred_pool[k] = allp
    pred_ens[k] = np.dot(w, allp)
    wrec.append(w.copy())
    true = price_mat[i_day]
    losses.append(np.array([mean_absolute_error(true, allp[j]) for j in range(len(POOL))]))
    w = v5.egd_update(w, losses)
    if (k+1) % 50 == 0:
        print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} {k+1}/{n_pred} '
              f'MAE={mean_absolute_error(true, pred_ens[k])*1e3:.2f}e-3 | w={np.round(w,3)}', flush=True)
print(f'滚动完成 {time.time()-t0:.1f}s', flush=True)

true_all = price_mat[31:]
idx31 = pd.to_datetime(a4.index)[31:]
rows = []
for j, name in enumerate(POOL):
    p = pred_pool[:, j, :]
    rows.append((f'成员-{name}', np.abs(p-true_all).mean()*1e3,
                 (np.abs(p-true_all)/true_all).mean()*100))
rows.append(('EGD集成(η=15)', np.abs(pred_ens-true_all).mean()*1e3,
             (np.abs(pred_ens-true_all)/true_all).mean()*100))
old = np.load(os.path.join(OUT, 'p4_price_pred_ens.npy'))
rows.append(('EGD集成(旧η=0.005,无ShapeLevel)', np.abs(old-true_all).mean()*1e3,
             (np.abs(old-true_all)/true_all).mean()*100))
res = pd.DataFrame(rows, columns=['方法', 'MAE(×1e-3元)', 'MAPE(%)']).sort_values('MAE(×1e-3元)').reset_index(drop=True)
pd.set_option('display.width', 200)
print(); print('===== 电价池v2 全期对比 ====='); print(res.to_string(float_format=lambda x: f'{x:.2f}'))

wdf = pd.DataFrame(wrec, columns=POOL, index=idx31)
print(); print('EGD权重终值:', np.round(wdf.iloc[-1].values, 3))
print('权重演化(每50天采样):'); print(wdf.iloc[::50].round(3).to_string())

m = pd.Series(np.abs(pred_ens-true_all).mean(axis=1), index=idx31)
print(); print('月度MAE(×1e-3):'); print(m.groupby(lambda d: d.month).mean().mul(1e3).round(2).to_string())

np.save(os.path.join(OUT, 'p4_price_pred_ens_v2.npy'), pred_ens)
np.save(os.path.join(OUT, 'p4_price_pred_pool_v2.npy'), pred_pool)
wdf.to_excel(os.path.join(OUT, 'p4_电价EGD权重演化_v2.xlsx'))
res.to_excel(os.path.join(OUT, 'p4_电价预测对比_v2.xlsx'), index=False)
print('已保存v2')
