# -*- coding: utf-8 -*-
"""
P3-M6 内生分位诊断(PLAN §5.1 的数值核对, 抽样日)
================================================================================
理论(§5.1):偏差罚两侧对称 0.5λ ⇒ 最优计划 p_t 落在"真值分布"的**中位数**;
  调整步边际高于计划 1.5λ / 低于计划 0.5λ, 缺口 5λ ⇒ 调整落在 **0.7 / 0.9 分位**。
数值核对(用模型自身的情景集合作为"真值分布"的代理):
  ① 计划的中位性:对每个 t≥36, 取该时段各情景的成交 q_ω,t(节点常量), 求其(加权)中位数,
     与 p_t 比较; 若 §5.1 成立, 二者应接近(计划 = 成交分布的中位数)。
  ② 调整的分位位置:把 q_ω,t 放到该时段情景净需求 NL_ω,t=(L−PV)Δt 的经验分布里求分位,
     报全窗中位分位; 若 §5.1 成立应显著高于 0.5(落在 0.7–0.9 区间)。
只读诊断, 不改动任何主口径。用法: python P3_audit_m6.py [stride]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import P3_tree as M4                                            # noqa: E402
import P3_solve as M5                                           # noqa: E402

P3 = Path(__file__).resolve().parent
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
m5df = pd.read_excel(P3 / "m5_out" / "m5_结果.xlsx")
soc_map = dict(zip(m5df["e"].values, m5df["SOC初"].values))
days = list(range(28, 334, STRIDE))

pos_plan, pos_adj, med_gap, pos_plan0 = [], [], [], []
for e in days:
    soc0 = float(soc_map[e])
    scen = M4.build_scenarios(e); tree = M4.build_tree(e, scen)
    lp = M5.build_lp(scen, tree, soc0, M5.LAM)                  # 默认主模型
    r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs")
    assert r.success, f"日{e} 失败"
    x = r.x; idx = lp["idx"]; K = scen["K"]
    P = np.array([x[idx[("P", t)]] for t in range(144)])
    NL = (scen["L_slot"] - scen["PV_slot"]) * M5.DT             # (K,144) kWh 净需求
    for t in range(36, 144):
        s = 2 if t < 72 else (3 if t < 108 else 4)
        q = np.array([x[idx[("Q", s, lp["node_of"][s, w], t)]] for w in range(K)])
        wt = np.array([lp["sizes"][s][lp["node_of"][s, w]] for w in range(K)], float)
        # 加权中位数
        o = np.argsort(q); qs = q[o]; ws = wt[o]; c = np.cumsum(ws) / ws.sum()
        med = qs[np.searchsorted(c, 0.5)]
        med_gap.append(abs(med - P[t]))
        pos_plan.append(float((q <= P[t] + 1e-9).dot(wt) / wt.sum()))
        pos_adj.append(float((NL[:, t] <= np.median(q)).mean()))
    # 计划块内(t<36): 计划 vs 情景净需求分位
    for t in range(36):
        pos_plan0.append(float((NL[:, t] <= P[t]).mean()))

med_gap = np.array(med_gap)
print(f"===== M6 内生分位诊断(抽样 {len(days)} 日, stride={STRIDE}) =====")
print(f"  ① 计划中位性: |median_ω(q_t) − p_t| 中位 {np.median(med_gap):8.1f} kWh, "
      f"均值 {med_gap.mean():8.1f} kWh  (=0 表示计划恰为成交分布中位数)")
print(f"     p_t 落在成交分布中的分位(中位) {np.median(pos_plan):.3f}")
print(f"  ② 计划 vs 情景净需求 NL 的分位(0:00 块 t<36, 中位) {np.median(pos_plan0):.3f}")
print(f"  ③ 调整成交 vs NL 的分位(中位) {np.median(pos_adj):.3f}  (理论 0.7–0.9)")
print(f"     (注: 分布代理=当日情景集合; 净需求 NL 未含储能, 仅作分位位置的量级核对)")
