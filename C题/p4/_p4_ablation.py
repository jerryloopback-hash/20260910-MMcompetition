# -*- coding: utf-8 -*-
"""
问题4电价池消融实验: 逐成员去除后重放EGD(η=15)全期集成, 看边际贡献。
与 _ablation_v5.py 同思路——预测不依赖权重, 可由 (334,6,144) 数组离线重算。
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

spec = importlib.util.spec_from_file_location('v6mod', r'D:\CUMCM2026Problems\C题\预测_v6_问题4电价.py')

# 不执行v6主流程, 手工复刻其EGD循环参数
POOL = ['SeasonalNaive', 'HistMean', 'STL', 'kNN', 'ShapeLevel', 'LightGBM']
ETA_, WIN_, LAM_ = 15.0, 30, 0.95
COLD = 35
INIT = np.array([0.20, 0.18, 0.18, 0.18, 0.14, 0.12])
OUT = r'D:\CUMCM2026Problems\C题\预测结果'

pool = np.load(rf'{OUT}\price_pred_pool_v6.npy')          # (334,6,144)
a4 = pd.read_excel(r'D:\CUMCM2026Problems\C题\附件\附件4.xlsx', index_col=0)
a4.columns = range(144)
pm = a4.values.astype(float)
true_all = pm[31:]
n_pred = pool.shape[0]


def egd_update(w, losses_history):
    recent = np.array(losses_history[-WIN_:])
    if len(recent) == 1:
        wl = recent[0]
    else:
        dcy = np.array([LAM_ ** (len(recent)-1-i) for i in range(len(recent))])
        wl = np.average(recent, axis=0, weights=dcy/dcy.sum())
    log_w = np.log(w + 1e-12) - ETA_*wl
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def replay(members):
    """members: 保留的池下标列表 → 重放EGD得集成MAE"""
    keep = np.array(members)
    init = INIT[keep] / INIT[keep].sum()
    w = init.copy()
    losses = []
    ens = np.zeros((n_pred, 144))
    for k in range(n_pred):
        n_train = 31 + k                       # train_end_idx+1, 与v6一致
        if k % 7 == 0 and k > 0:
            w_init = (INIT[keep] / INIT[keep].sum()) if n_train < COLD else np.ones(len(keep))/len(keep)
            alpha = max(0.05, 0.5*np.exp(-(n_train - COLD)/60.0))
            w = alpha*w_init + (1-alpha)*w
        p = pool[k, keep, :]
        ens[k] = np.dot(w, p)
        true = true_all[k]
        losses.append(np.array([mean_absolute_error(true, p[j]) for j in range(len(keep))]))
        w = egd_update(w, losses)
    return np.abs(ens - true_all).mean()*1e3, ens


rows = []
full_mae, full_ens = replay(list(range(6)))
rows.append(('完整池(6成员)', full_mae, 0.0))
for j, name in enumerate(POOL):
    mae, _ = replay([x for x in range(6) if x != j])
    rows.append((f'去{name}', mae, mae - full_mae))

res = pd.DataFrame(rows, columns=['方案', 'MAE(×1e-3元)', 'ΔMAE(去除后-完整)'])
pd.set_option('display.width', 200)
print(res.to_string(float_format=lambda x: f'{x:.2f}'))
res.to_excel(rf'{OUT}\电价消融实验_v6.xlsx', index=False)
np.save(rf'{OUT}\price_pred_ens_replay_v6.npy', full_ens)
print('\n已保存 电价消融实验_v6.xlsx')
