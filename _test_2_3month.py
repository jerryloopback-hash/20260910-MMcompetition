# -*- coding: utf-8 -*-
"""测试脚本：只跑 2-3 月滚动预测，验证流程与精度"""
import sys, time
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
src = open(r'D:\CUMCM2026Problems\C题\预测_小区负载与光伏.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(src, globals())

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dates, load_mat, pv_mat = load_data()
t0 = time.time()
lp, pp, pd_, wrec = rolling_predict(dates, load_mat, pv_mat, '2025-02-01', '2025-03-31', 7)
print('2-3月耗时 %.1f s' % (time.time()-t0))

d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
recs = []
for k, day in enumerate(pd_):
    i = d2i[pd.Timestamp(day)]
    recs.append((day, load_mat[i], lp[k], pv_mat[i], pp[k]))
rec = pd.DataFrame(recs, columns=['date', 'lt', 'lp', 'pt', 'pp'])
lt = np.vstack(rec['lt'].values); lpp = np.vstack(rec['lp'].values)
pt = np.vstack(rec['pt'].values); pp_ = np.vstack(rec['pp'].values)
print('2-3月 负载MAE=%.1f RMSE=%.1f MAPE=%.2f%%' % (
    mean_absolute_error(lt, lpp), np.sqrt(mean_squared_error(lt, lpp)),
    100*np.mean(np.abs(lt-lpp)/(np.abs(lt)+1e-6))))
print('2-3月 光伏MAE=%.1f RMSE=%.1f' % (mean_absolute_error(pt, pp_), np.sqrt(mean_squared_error(pt, pp_))))
# 权重轨迹
wr = pd.DataFrame(wrec, columns=['date', 'w_ml_load', 'w_ml_pv', 'n_train'])
print('权重演化(负载ML):')
print(wr['w_ml_load'].describe().round(3).to_string())
