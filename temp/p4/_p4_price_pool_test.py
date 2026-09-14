# -*- coding: utf-8 -*-
"""
问题4电价预测快速实验：负载池5模型(SeasonalNaive/HistMean/STL/kNN/LGB)+EGD
直接灌附件4电价矩阵，对比结构性基准，确定问题4-2预测层方案。
复用 预测_v5_扩展模型池.py 的全部泛型函数（滚动origin口径完全一致）。
"""
import importlib.util
import sys, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

V5_PATH = r'D:\CUMCM2026Problems\C题\预测_v5_扩展模型池.py'
spec = importlib.util.spec_from_file_location('v5', V5_PATH)
v5 = importlib.util.module_from_spec(spec)
sys.modules['v5'] = v5
spec.loader.exec_module(v5)

OUT = r'D:\CUMCM2026Problems\C题\p4'
os.makedirs(OUT, exist_ok=True)
USE_LGB = '--nolgb' not in sys.argv

# ---------- 数据：附件4 电价矩阵 (365×144) + 附件1 形状 ----------
a4 = pd.read_excel(r'D:\CUMCM2026Problems\C题\附件\附件4.xlsx', index_col=0)
a4.columns = range(v5.N_SLOT)
price_mat = a4.values.astype(float)
dates = pd.to_datetime(a4.index).values.astype('datetime64[D]')
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

a1 = pd.read_excel(r'D:\CUMCM2026Problems\C题\附件\附件1.xlsx')
shape144 = a1['电价'].values.astype(float)          # 附件1日内形状（=365天平均形状）
shape_mean = shape144.mean()

n_pred_days = pd.date_range('2025-02-01', '2025-12-31', freq='D')
n_pred = len(n_pred_days)

# ---------- 日水平(level)可预测性诊断 ----------
lev = price_mat.mean(axis=1)
lev_hat = {
    'lag1(昨日水平)': np.concatenate([[np.nan], lev[:-1]]),
    'lag7(上周水平)': np.concatenate([[np.nan]*7, lev[:-7]]),
    'ma7(近7日)': pd.Series(lev).rolling(7).mean().shift(1).values,
    'ma28(近28日)': pd.Series(lev).rolling(28).mean().shift(1).values,
}
lev_msk = ~pd.isna(list(lev_hat.values())[0])
print('===== 日水平可预测性（预测期334天, MAE×1e-3 元/kWh） =====')
for k, v in lev_hat.items():
    m = ~np.isnan(v[31:])
    print(f'  {k:14s} MAE={np.abs(v[31:][m]-lev[31:][m]).mean()*1e3:7.2f}')

# ---------- 模型池定义（负载池5 + ShapeLevel结构模型） ----------
POOL = ['SeasonalNaive', 'HistMean', 'STL', 'kNN', 'LGB', 'ShapeLevel']

def shape_level_predict(i_day, lev_win=7):
    """λ̂_t = 附件1形状_t × 日水平̂（近lev_win日水平均值）"""
    if i_day < lev_win:
        return shape144
    lv = lev[i_day-lev_win:i_day].mean()
    return shape144 * (lv / shape_mean)

def predict_pool(i_day, lgb_model):
    p = np.zeros((len(POOL), v5.N_SLOT))
    p[0] = v5.seasonal_naive(price_mat, i_day)
    p[1] = v5.hist_mean(price_mat, i_day)
    p[2] = v5.stl_predict(price_mat, i_day)
    p[3] = v5.knn_predict(price_mat, i_day, dates, by_dow=True)
    p[4] = v5.lgb_predict(lgb_model, v5.build_pred_matrix(dates[i_day], date_to_idx, price_mat)) \
        if lgb_model is not None else p[1]
    p[5] = shape_level_predict(i_day)
    return np.clip(p, 0.01, None)   # 电价非负（保底0.01防0除）

# ---------- 滚动预测 ----------
w = np.array([0.22, 0.18, 0.18, 0.17, 0.15, 0.10])   # 冷启动：简单+结构占大头
losses = []
pred_pool = np.full((n_pred, len(POOL), v5.N_SLOT), np.nan)
pred_ens = np.full((n_pred, v5.N_SLOT), np.nan)
wrec = []
lgb_model = None
t0 = time.time()

for k, day in enumerate(n_pred_days):
    i_day = date_to_idx[pd.Timestamp(day)]
    train_end_idx = i_day - 1
    n_train = train_end_idx + 1

    if USE_LGB and ((k % v5.RETRAIN_EVERY == 0) or lgb_model is None) and train_end_idx >= v5.COLD_START_DAYS:
        X_tr, y_tr, X_va, y_va = v5.build_train_matrix(dates, price_mat, train_end_idx, date_to_idx)
        if X_tr is not None and len(X_tr) > 100:
            lgb_model = v5.train_lgb(X_tr, y_tr, X_va, y_va)
        w = v5.k_step_reinit(w, n_train, lambda n: np.array([0.22, 0.18, 0.18, 0.17, 0.15, 0.10]))

    allp = predict_pool(i_day, lgb_model)
    pred_pool[k] = allp
    comb = np.dot(w, allp)
    pred_ens[k] = comb
    wrec.append(w.copy())

    true = price_mat[i_day]
    losses.append(np.array([mean_absolute_error(true, allp[j]) for j in range(len(POOL))]))
    w = v5.egd_update(w, losses)

    if (k+1) % 50 == 0:
        print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} {k+1}/{n_pred} '
              f'MAE={mean_absolute_error(true, comb)*1e3:.2f}e-3 | w={np.round(w,2)}')

print(f'滚动完成 {time.time()-t0:.1f}s')

# ---------- 基准 ----------
true_all = price_mat[31:]
bench = {
    '附件1形状(Climatology)': np.tile(shape144, (n_pred, 1)),
    '昨日曲线(Persistence)': price_mat[30:-1],
    '近7日均值曲线': np.array([price_mat[i-7:i].mean(axis=0) for i in range(31, 365)]),
    '近28日均值曲线': np.array([price_mat[max(0,i-28):i].mean(axis=0) for i in range(31, 365)]),
    'ShapeLevel(lag1水平)': np.array([shape144*(lev[i-1]/shape_mean) for i in range(31, 365)]),
    'ShapeLevel(ma7水平)': np.array([shape_level_predict(i, 7) for i in range(31, 365)]),
}
for j, name in enumerate(POOL):
    bench[f'池成员-{name}'] = pred_pool[:, j, :]
bench['EGD集成'] = pred_ens

# ---------- 评价 ----------
peak_msk = np.zeros(v5.N_SLOT, bool)
peak_msk[18*6:22*6] = True   # 18-22时晚高峰
rows = []
for name, p in bench.items():
    mae = np.abs(p - true_all).mean()
    mape = (np.abs(p - true_all) / true_all).mean() * 100
    rmse = np.sqrt(((p - true_all)**2).mean())
    mae_peak = np.abs(p[:, peak_msk] - true_all[:, peak_msk]).mean()
    rows.append((name, mae*1e3, mape, rmse*1e3, mae_peak*1e3))
res = pd.DataFrame(rows, columns=['方法', 'MAE(×1e-3元)', 'MAPE(%)', 'RMSE(×1e-3)', '晚高峰MAE(×1e-3)']) \
    .sort_values('MAE(×1e-3元)').reset_index(drop=True)
pd.set_option('display.width', 200)
print()
print('===== 电价预测334天全期对比（元/kWh） =====')
print(res.to_string(float_format=lambda x: f'{x:.2f}'))

# 月度MAE（集成 vs 最强基准）
ens_m = pd.Series(np.abs(pred_ens - true_all).mean(axis=1), index=n_pred_days)
clim_m = pd.Series(np.abs(bench['附件1形状(Climatology)'] - true_all).mean(axis=1), index=n_pred_days)
print()
print('===== 月度MAE(×1e-3) EGD集成 vs 附件1形状 =====')
mon = pd.DataFrame({'EGD集成': ens_m, '附件1形状': clim_m}).groupby(lambda d: d.month).mean()
mon['EGD增益%'] = (mon['EGD集成']/mon['附件1形状']-1)*100
print((mon*1e3).round(2).to_string())

np.save(os.path.join(OUT, 'p4_price_pred_ens.npy'), pred_ens)
np.save(os.path.join(OUT, 'p4_price_pred_pool.npy'), pred_pool)
res.to_excel(os.path.join(OUT, 'p4_电价预测对比.xlsx'), index=False)
pd.DataFrame(wrec, columns=POOL).to_excel(os.path.join(OUT, 'p4_电价EGD权重演化.xlsx'), index=False)
print()
print(f'已保存: {OUT}')
