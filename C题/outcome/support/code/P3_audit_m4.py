# -*- coding: utf-8 -*-
"""
P3-M4 复核脚本(独立于 P3_tree.py, 只读不改)
=================================================
审计1  执行重解的条件集规模: 逐日 n_cond(阶段 6/12/18), 含 n_cond==1 的占比。
审计2  条件 vs 无条件 的残差谱宽对比(执行重解实际用的 ε_s 路径), 量化"不问前缀"的代价。
审计3  6:00 节点 SOC 的散布来源: 0:00-6:00 光伏量级 —— 夜间 PV≈0 则 SOC(6:00) 逐场景近似确定。
审计4  1 月供体(w≡1) 构造可行性与尺度比。
用法: python P3_audit_m4.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
P3 = Path(__file__).resolve().parent
M1, M3 = P3 / "m1_out", P3 / "m3_out"
sys.path.insert(0, str(P3))
import P3_tree as M4                                    # noqa: E402

NS, NL, H0 = 4, 24, np.array([0, 6, 12, 18])
I0 = 31
C_adj = M4.C_adj

# ---------- 审计1: 条件集规模 ----------
print("[审计1] 执行重解条件集规模 (逐日路由已实现前缀; 334 日)")
rows = []
for e in range(1, 334):
    if M4.pool_of(e)[1] == 0:
        continue
    tree = M4.build_tree(e, M4.build_scenarios(e))
    nc = [M4.build_exec_ensemble(e, s, tree=tree)["n_cond"] for s in (1, 2, 3)]
    rows.append({"e": e, "池宽": M4.pool_of(e)[1], "6:00": nc[0], "12:00": nc[1], "18:00": nc[2]})
d = pd.DataFrame(rows)
for c in ("6:00", "12:00", "18:00"):
    print(f"  {c:>6}: 中位 {d[c].median():.0f}  均值 {d[c].mean():.1f}  "
          f"=1 的占比 {(d[c] == 1).mean() * 100:4.1f}%  最小 {d[c].min()}")

# ---------- 审计2: 条件/无条件 残差谱宽 ----------
print("\n[审计2] 执行重解残差谱宽: 条件集 vs 全池 (逐维 std 均值再对日平均; 仅 n_cond>=3 的日)")
res = {1: [], 2: [], 3: []}
for e in range(1, 334):
    if M4.pool_of(e)[1] == 0:
        continue
    tree = M4.build_tree(e, M4.build_scenarios(e))
    for s in (1, 2, 3):
        a = M4.build_exec_ensemble(e, s, tree=tree, cond=True)
        b = M4.build_exec_ensemble(e, s, tree=tree, cond=False)
        if a["n_cond"] < 3:
            continue
        ra = (a["paths_hour"] - a["root_hour"][None, :]).std(axis=0).mean()
        rb = (b["paths_hour"] - b["root_hour"][None, :]).std(axis=0).mean()
        res[s].append(ra / rb)
for s in (1, 2, 3):
    v = np.array(res[s])
    print(f"  会话{H0[s]:>2}:00 (n={len(v):3d} 日): 条件/全池 谱宽比 均值 {v.mean():.3f} "
          f"中位 {np.median(v):.3f}  (1.0 = 条件无信息, 越小 = 前缀越能定未来)")

# ---------- 审计3: 6:00 节点 SOC 的散布来源 ----------
print("\n[审计3] 0:00-6:00 光伏量级 (决定 SOC(6:00) 是否逐场景确定)")
P = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:145].astype(float).values
night = P[:, 0:36]
egdp = np.load(BASE / "p2_part1" / "预测结果" / "pv_pred_ens_v5.npy")
print(f"  实际光伏 slots 0..35 (0:00-6:00): 均值 {night.mean():6.2f} kW, 最大 {night.max():7.1f} kW")
print(f"  EGD  光伏 slots 0..35          : 均值 {egdp[:, 0:36].mean():6.2f} kW, 最大 {egdp[:, 0:36].max():7.1f} kW")
print(f"  夜间逐日累计电量 均值 {night.sum(axis=1).mean()/6:6.1f} kWh / 最大 {night.sum(axis=1).max()/6:6.1f} kWh "
      f"(SOC 区间宽 9600 kWh)")

# ---------- 审计4: 1 月供体(w≡1) ----------
print("\n[审计4] 1 月供体(w≡1) 构造与尺度")
F_corr = np.load(M1 / "F_corr.npy")
A_nat = np.load(M1 / "A_nat.npy")
day = np.arange(5, 20)
jan = (A_nat[:I0] - F_corr[:I0, 0, :])[:, day]
feb = (A_nat[I0:] - C_adj[:, 0, :])[:, day]
print(f"  1月 ε0(w≡1):   std={jan.std():6.1f} MAE={np.abs(jan).mean():6.1f} n={len(jan)}")
print(f"  2月起 ε0(混合): std={feb.std():6.1f} MAE={np.abs(feb).mean():6.1f} n={len(feb)}")
print(f"  尺度比 std {jan.std()/feb.std():.2f} / MAE {np.abs(jan).mean()/np.abs(feb).mean():.2f}")
