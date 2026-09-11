# -*- coding: utf-8 -*-
"""过拟合诊断 v2：用带早停的模型 + 前向80/20切分"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
src = open(r'D:\CUMCM2026Problems\C题\预测_小区负载与光伏.py', encoding='utf-8').read()
src = src.split('if __name__')[0]
exec(src, globals())

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dates, load_mat, pv_mat = load_data()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
n_days = len(dates)

check_points = ['2025-04-30', '2025-07-31', '2025-10-31']

def tscv_eval(cut_date, target='load'):
    cut = date_to_idx[pd.Timestamp(cut_date)]
    mat = load_mat if target == 'load' else pv_mat
    val_start = int(cut - (cut - 28) * 0.2)
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
    Xv, yv = _build(val_start, cut)
    Xt, yt = _build(28, val_start - 1)
    if Xt is None or len(Xt) < 100:
        return None
    # 早停：在训练尾部切最后7天作早停验证
    n_tr = len(Xt)
    split = int(n_tr * 0.95)
    mdl = train_lgb_early(Xt[:split], yt[:split], Xt[split:], yt[split:])
    tr_mae = mean_absolute_error(yt, mdl.predict(Xt))
    va_mae = mean_absolute_error(yv, mdl.predict(Xv))
    tr_rmse = np.sqrt(mean_squared_error(yt, mdl.predict(Xt)))
    va_rmse = np.sqrt(mean_squared_error(yv, mdl.predict(Xv)))
    return dict(截止=cut_date, 目标=target, 训练样本=len(Xt), 验证样本=len(Xv),
                训练MAE=round(tr_mae,1), 验证MAE=round(va_mae,1),
                训练RMSE=round(tr_rmse,1), 验证RMSE=round(va_rmse,1),
                过拟合比=round(va_mae/tr_mae, 3))

print('===== 过拟合诊断 v2（早停模型, 前向80/20） =====')
rows = []
for cut in check_points:
    for tgt in ['load', 'pv']:
        r = tscv_eval(cut, tgt)
        if r: rows.append(r)
diag = pd.DataFrame(rows)
print(diag.to_string(index=False))
diag.to_excel(r'D:\CUMCM2026Problems\C题\预测结果\过拟合诊断_时序切分.xlsx', index=False)
