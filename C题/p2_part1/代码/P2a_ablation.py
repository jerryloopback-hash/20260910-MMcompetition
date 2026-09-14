# -*- coding: utf-8 -*-
"""
v5消融实验：利用已保存的逐模型预测数组，去掉某类模型后重算EGD权重，看性能下降
不需要重跑模型预测，仅重算权重（秒级）
"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 复用v5的数据加载、模型名、EGD参数
src = open(r'D:\CUMCM2026Problems\C题\P2a_forecast.py', encoding='utf-8').read()
exec(src.split('# ---------------- 3.')[0])

OUT = r'D:\CUMCM2026Problems\C题\预测结果'
EGD_ETA = 0.005; EGD_WINDOW = 30; EGD_LAMBDA = 0.95
RETRAIN_EVERY = 7; COLD_START_DAYS = 35

def egd_update(w, losses_history):
    if len(losses_history) == 0: return w
    recent = np.array(losses_history[-EGD_WINDOW:])
    if len(recent) == 1: weighted_loss = recent[0]
    else:
        weights = np.array([EGD_LAMBDA**(len(recent)-1-i) for i in range(len(recent))])
        weights = weights / weights.sum()
        weighted_loss = np.average(recent, axis=0, weights=weights)
    log_w = np.log(w + 1e-12) - EGD_ETA * weighted_loss
    log_w -= log_w.max()
    w_new = np.exp(log_w)
    return w_new / w_new.sum()

def run_egd(pred_models, true_mat, model_names, init_weights_fn):
    """pred_models: (n_days, n_models, 144), true_mat: (n_days, 144)"""
    n_days, n_models, _ = pred_models.shape
    w = init_weights_fn(31)
    losses = []
    comb = np.zeros_like(true_mat)
    for k in range(n_days):
        comb[k] = np.dot(w, pred_models[k])
        day_loss = np.array([mean_absolute_error(true_mat[k], pred_models[k,j]) for j in range(n_models)])
        losses.append(day_loss)
        w = egd_update(w, losses)
        if (k+1) % RETRAIN_EVERY == 0:
            w_init = init_weights_fn(31 + k)
            alpha = max(0.05, 0.5 * np.exp(-(31+k - COLD_START_DAYS) / 60.0))
            w = alpha * w_init + (1-alpha) * w
    return comb

def pv_mape_daytime(t, p):
    hrs = ti_to_hour(np.arange(N_SLOT))
    day = (hrs > 6.0) & (hrs < 20.0)
    t = t[:, day]; p = p[:, day]
    m = t > 50.0
    if m.sum() == 0: return np.nan
    return 100*np.mean(np.abs(t[m]-p[m])/(np.abs(t[m])+1e-6))

# 加载数据
dates, load_mat, pv_mat = load_data()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
pred_days = pd.date_range('2025-02-01', '2025-12-31', freq='D')
idx = [date_to_idx[pd.Timestamp(d)] for d in pred_days]
lt = load_mat[idx]; pt = pv_mat[idx]

lpm = np.load(OUT + r'\load_pred_models_v5.npy')  # (334,5,144)
ppm = np.load(OUT + r'\pv_pred_models_v5.npy')    # (334,5,144)
print(f'负载预测数组: {lpm.shape}, 光伏: {ppm.shape}')

# 负载初始权重（5模型：SN, HM, STL, kNN, LGB）
def init_load_full(n):
    if n < COLD_START_DAYS: return np.array([0.25,0.20,0.20,0.20,0.15])
    return np.ones(5)/5
# 光伏初始权重（5模型）
def init_pv_full(n):
    if n < COLD_START_DAYS: return np.array([0.30,0.20,0.15,0.15,0.20])
    return np.ones(5)/5

def make_init(keep_idx, base_init):
    """去掉某些模型后，剩余模型初始权重重归一化"""
    def fn(n):
        w = base_init(n)[keep_idx]
        return w / w.sum()
    return fn

# 负载消融方案（5模型：0=SN, 1=HM, 2=STL, 3=kNN, 4=LGB）
load_ablations = {
    '完整5模型': list(range(5)),
    '去掉STL': [0,1,3,4],
    '去掉kNN': [0,1,2,4],
    '去掉LightGBM': [0,1,2,3],
    '去掉SeasonalNaive': [1,2,3,4],
    '去掉HistMean': [0,2,3,4],
    '仅统计模型(去LGB)': [0,1,2,3],
}
# 光伏消融方案
pv_ablations = {
    '完整5模型': list(range(5)),
    '去掉kNN': [0,2,3,4],
    '去掉STL': [0,1,3,4],
    '去掉FPCA': [0,1,2,4],
    '去掉LightGBM': [0,1,2,3],
    '去掉Persistence': [1,2,3,4],
}

print('\n===== 负载消融 =====')
load_rows = []
for name, keep in load_ablations.items():
    keep = np.array(keep)
    sub = lpm[:, keep, :]
    names_sub = [LOAD_MODELS[i] for i in keep]
    comb = run_egd(sub, lt, names_sub, make_init(keep, init_load_full))
    mae = mean_absolute_error(lt, comb)
    rmse = np.sqrt(mean_squared_error(lt, comb))
    mape = 100*np.mean(np.abs(lt-comb)/(np.abs(lt)+1e-6))
    load_rows.append({'消融方案': name, '模型数': len(keep), 'MAE': mae, 'RMSE': rmse, 'MAPE%': mape})
    print(f'  {name:28s} n={len(keep)} MAE={mae:7.2f} RMSE={rmse:7.2f} MAPE={mape:.2f}%')

print('\n===== 光伏消融 =====')
pv_rows = []
for name, keep in pv_ablations.items():
    keep = np.array(keep)
    sub = ppm[:, keep, :]
    names_sub = [PV_MODELS[i] for i in keep]
    comb = run_egd(sub, pt, names_sub, make_init(keep, init_pv_full))
    mae = mean_absolute_error(pt, comb)
    rmse = np.sqrt(mean_squared_error(pt, comb))
    mape = pv_mape_daytime(pt, comb)
    pv_rows.append({'消融方案': name, '模型数': len(keep), 'MAE': mae, 'RMSE': rmse, '白天MAPE%': mape})
    print(f'  {name:28s} n={len(keep)} MAE={mae:7.2f} RMSE={rmse:7.2f} 白天MAPE={mape:.2f}%')

# 保存
with pd.ExcelWriter(OUT + r'\消融实验_v5.xlsx') as writer:
    pd.DataFrame(load_rows).to_excel(writer, sheet_name='负载消融', index=False)
    pd.DataFrame(pv_rows).to_excel(writer, sheet_name='光伏消融', index=False)
print('\n消融结果已保存到 消融实验_v5.xlsx')
