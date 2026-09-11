# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
src = open(r'D:\CUMCM2026Problems\C题\预测_v4_多模型EGD集成.py', encoding='utf-8').read()
exec(src.split('if __name__')[0])
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error

dates, load_mat, pv_mat = load_data()
result = rolling_predict(dates, load_mat, pv_mat, '2025-02-01', '2025-03-31')
load_pred, pv_pred, lpm, ppm, pd_days, wrec = result
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
lt = np.vstack([load_mat[date_to_idx[pd.Timestamp(d)]] for d in pd_days])
pt = np.vstack([pv_mat[date_to_idx[pd.Timestamp(d)]] for d in pd_days])
print(f'\n2-3月 负载MAE={mean_absolute_error(lt, load_pred):.1f} 光伏MAE={mean_absolute_error(pt, pv_pred):.1f}')
print('最终权重(负载):', dict(zip(MODEL_NAMES, np.round(wrec[-1][1], 3))))
print('最终权重(光伏):', dict(zip(MODEL_NAMES, np.round(wrec[-1][2], 3))))
# 逐模型2-3月误差
for j, name in enumerate(MODEL_NAMES):
    print(f'  {name}: 负载MAE={mean_absolute_error(lt, lpm[:,j,:]):.1f} 光伏MAE={mean_absolute_error(pt, ppm[:,j,:]):.1f}')
