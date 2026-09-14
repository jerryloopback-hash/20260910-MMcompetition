# -*- coding: utf-8 -*-
"""
P1 敏感性分析：η（充/放各 η，题面给 90%）与功率上限 PMAX 的完整 LP 重解扫描。
题面 90% 是硬数据，本扫描只为展示模型对参数的响应结构（阈值 1/η² 随之移动），
论文中仅作简要讨论。结果同时作为 η 扫描图表数据。
"""
from pathlib import Path
import numpy as np
import openpyxl
from scipy.optimize import linprog

OUTDIR = Path(__file__).resolve().parent
BASE = OUTDIR.parent

wb = openpyxl.load_workbook(BASE / "附件" / "附件1.xlsx", data_only=True)
rows = list(wb["Sheet1"].iter_rows(min_row=2, values_only=True))
raw_price = np.array([r[1] for r in rows], float)
raw_load = np.array([r[2] for r in rows], float)
raw_pv = np.array([r[3] for r in rows], float)
price = np.concatenate([raw_price[143:], raw_price[:143]])
load = np.concatenate([raw_load[143:], raw_load[:143]])
pv = np.concatenate([raw_pv[143:], raw_pv[:143]])

DT, T = 1 / 6.0, 144
EMIN, EMAX, E0 = 1200.0, 10800.0, 6000.0
NV = T * 6
IDX = {k: np.arange(j, NV, 6) for j, k in enumerate("abgcdq")}


def solve(eta=0.9, pmax=5000.0):
    c_obj = np.zeros(NV)
    c_obj[IDX["a"]] = price * DT
    c_obj[IDX["b"]] = price * DT
    A_eq, b_eq = [], []
    for t in range(T):
        rL = np.zeros(NV); rL[IDX["a"][t]] = 1; rL[IDX["g"][t]] = 1; rL[IDX["d"][t]] = 1
        rP = np.zeros(NV); rP[IDX["g"][t]] = 1; rP[IDX["c"][t]] = 1; rP[IDX["q"][t]] = 1
        A_eq += [rL, rP]; b_eq += [load[t], pv[t]]
    rE = np.zeros(NV)
    for t in range(T):
        rE[IDX["b"][t]] += eta; rE[IDX["c"][t]] += eta; rE[IDX["d"][t]] += -1 / eta
    A_eq.append(rE); b_eq.append(0.0)
    A_ub, b_ub = [], []
    for t in range(T):
        rU = np.zeros(NV); rLw = np.zeros(NV)
        for k in range(t + 1):
            rU[IDX["b"][k]] += eta * DT; rU[IDX["c"][k]] += eta * DT; rU[IDX["d"][k]] += -DT / eta
            rLw[IDX["b"][k]] += -eta * DT; rLw[IDX["c"][k]] += -eta * DT; rLw[IDX["d"][k]] += DT / eta
        A_ub += [rU, rLw]; b_ub += [EMAX - E0, E0 - EMIN]
    for t in range(T):
        rC = np.zeros(NV); rC[IDX["b"][t]] = 1; rC[IDX["c"][t]] = 1
        rD = np.zeros(NV); rD[IDX["d"][t]] = 1
        A_ub += [rC, rD]; b_ub += [pmax, pmax]
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * NV, method="highs")
    assert res.success, res.message
    x = res.x
    pur = (x[IDX["a"]] + x[IDX["b"]]) * DT
    return float(res.fun), pur.sum(), price.min() * 0 + 0  # cost, purchase kWh


print("=== η 扫描（充、放各 η；题面 0.90）===")
print(f"{'η':>6} {'阈值1/η²':>9} {'购电费(元)':>12} {'全天购电量(kWh)':>16} {'较η=0.90增幅':>12}")
base = None
sens_eta = []
for eta in (0.85, 0.88, 0.90, 0.92, 0.95):
    cost, pur, _ = solve(eta=eta)
    if eta == 0.90:
        base = cost
    sens_eta.append((eta, cost))
    print(f"{eta:>6.2f} {1/eta**2:>9.4f} {cost:>12,.2f} {pur:>16,.1f} "
          f"{(cost/base-1)*100 if base else 0:>11.2f}%")

print("\n=== 功率上限 PMAX 扫描（η=0.90）===")
print(f"{'PMAX(kW)':>9} {'购电费(元)':>12} {'较5000增幅':>11}")
sens_pmax = []
for pmax in (4000.0, 5000.0, 6000.0):
    cost, pur, _ = solve(pmax=pmax)
    sens_pmax.append((pmax, cost))
    print(f"{pmax:>9.0f} {cost:>12,.2f} {(cost/ (sens_eta[2][1]) - 1)*100:>10.2f}%")

np.savez(OUTDIR / "_sensitivity.npz",
         eta=np.array([r[0] for r in sens_eta]),
         cost_eta=np.array([r[1] for r in sens_eta]),
         pmax=np.array([r[0] for r in sens_pmax]),
         cost_pmax=np.array([r[1] for r in sens_pmax]))
print(f"\n已存档 -> {OUTDIR/'_sensitivity.npz'}")
