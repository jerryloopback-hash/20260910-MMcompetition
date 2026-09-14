# -*- coding: utf-8 -*-
"""本地代跑队友 LSTM baseline(仅移植路径, 算法零改动) -> lstm_load_pred.npy / lstm_pv_pred.npy
用法: python P2b_lstm.py load|pv|all
"""
import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # C题/
P1   = ROOT/"p2_part1"
PRED = P1/"预测结果"

target = sys.argv[1] if len(sys.argv) > 1 else "all"
assert target in ("load", "pv", "all")

# 读队友脚本源码, 内存内替换 D 盘绝对路径为本仓库路径(算法零改动)
# 注意: _lstm_baseline 内部会自行 open 磁盘上的 v5 脚本并 exec, 故 v5 的 DATA_PATH 已直接在磁盘文件中移植为本机路径
lstm_src = (P1/"_lstm_baseline.py").read_text(encoding="utf-8").replace(r"D:\CUMCM2026Problems\C题", str(P1))

g = {"__name__": "lstm_local", "__file__": str(P1/"_lstm_baseline.py")}
exec(compile(lstm_src, "_lstm_baseline", "exec"), g)                              # 函数定义(__main__块跳过)

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dates, load_mat, pv_mat = g["load_data"]()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
pd = g["pd"]

def run_one(mat, is_pv):
    lp, pdays = g["rolling_lstm"](mat, dates, "2025-02-01", "2025-12-31")
    if is_pv:                                          # 夜间置0(与队友主流程一致)
        night = (g["ti_to_hour"](np.arange(g["N_SLOT"])) <= 6.0) | (g["ti_to_hour"](np.arange(g["N_SLOT"])) >= 20.0)
        lp[:, night] = 0.0
    idxs = [date_to_idx[pd.Timestamp(d)] for d in pdays]
    act = np.vstack([mat[i] for i in idxs])
    print(f"[done] MAE={mean_absolute_error(act, lp):.1f}  RMSE={np.sqrt(mean_squared_error(act, lp)):.1f}")
    return lp

if target in ("load", "all"):
    lp = run_one(load_mat, is_pv=False)
    np.save(PRED/"lstm_load_pred.npy", lp)
    print("saved ->", PRED/"lstm_load_pred.npy")
if target in ("pv", "all"):
    pp = run_one(pv_mat, is_pv=True)
    np.save(PRED/"lstm_pv_pred.npy", pp)
    print("saved ->", PRED/"lstm_pv_pred.npy")
print("LSTM本地代跑完成")
