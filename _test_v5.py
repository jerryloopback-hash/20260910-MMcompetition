# -*- coding: utf-8 -*-
"""v5 快速测试：2-3月，验证新模型池各模型单独误差与EGD集成效果"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
src = open(r'D:\CUMCM2026Problems\C题\预测_v5_扩展模型池.py', encoding='utf-8').read()
exec(src.split('if __name__')[0])
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dates, load_mat, pv_mat = load_data()
print('开始2-3月快速测试...')
result = rolling_predict(dates, load_mat, pv_mat, '2025-02-01', '2025-03-31')
load_pred, pv_pred, lpm, ppm, pd_days, wrec = result
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
lt = np.vstack([load_mat[date_to_idx[pd.Timestamp(d)]] for d in pd_days])
pt = np.vstack([pv_mat[date_to_idx[pd.Timestamp(d)]] for d in pd_days])

print(f'\n===== 2-3月 EGD集成 =====')
print(f'负载 MAE={mean_absolute_error(lt, load_pred):.1f}  RMSE={np.sqrt(mean_squared_error(lt, load_pred)):.1f}  MAPE={100*np.mean(np.abs(lt-load_pred)/(np.abs(lt)+1e-6)):.2f}%')
print(f'光伏 MAE={mean_absolute_error(pt, pv_pred):.1f}  RMSE={np.sqrt(mean_squared_error(pt, pv_pred)):.1f}')

print(f'\n===== 负载逐模型误差(2-3月) =====')
for j, name in enumerate(LOAD_MODELS):
    lp_j = lpm[:, j, :]
    print(f'  {name:15s} MAE={mean_absolute_error(lt, lp_j):7.1f}  RMSE={np.sqrt(mean_squared_error(lt, lp_j)):7.1f}')

print(f'\n===== 光伏逐模型误差(2-3月) =====')
for j, name in enumerate(PV_MODELS):
    pp_j = ppm[:, j, :]
    print(f'  {name:15s} MAE={mean_absolute_error(pt, pp_j):7.1f}  RMSE={np.sqrt(mean_squared_error(pt, pp_j)):7.1f}')

print(f'\n最终权重(负载):', dict(zip(LOAD_MODELS, np.round(wrec[-1][1], 3))))
print(f'最终权重(光伏):', dict(zip(PV_MODELS, np.round(wrec[-1][2], 3))))

# 模型间误差相关性（快速版：用日MAE序列算相关）
print(f'\n===== 负载模型日MAE相关性 =====')
load_day_mae = np.array([[mean_absolute_error(lt[k], lpm[k,j,:]) for j in range(len(LOAD_MODELS))] for k in range(len(lt))])
print(pd.DataFrame(np.corrcoef(load_day_mae.T), index=LOAD_MODELS, columns=LOAD_MODELS).round(3).to_string())

print(f'\n===== 光伏模型日MAE相关性 =====')
pv_day_mae = np.array([[mean_absolute_error(pt[k], ppm[k,j,:]) for j in range(len(PV_MODELS))] for k in range(len(pt))])
print(pd.DataFrame(np.corrcoef(pv_day_mae.T), index=PV_MODELS, columns=PV_MODELS).round(3).to_string())
