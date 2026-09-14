# -*- coding: utf-8 -*-
"""
P3-M5 四阶段随机规划求解:计划(阶段1) + 各会话节点调整(阶段2-4)联合优化, 逐日滚动
================================================================================
依据 p3_plan.md §2.2/§2.4/§2.6/§2.7 / D1-D2 / D9 / 里程碑 M5。

模型(扩展型多阶段 LP, 每个评估日一次):
  决策共享(非预期性):
    P[t]              t=0..143  计划购电量(阶段1, 全场景共享)         —— 偏差罚的参照
    D1[t]             t=0..35   计划执行的储能放电(§2.7 其五: 计划储能只在 0:00-6:00 执行)
    Q[s][n][t]       阶段 s(2..4) 节点 n 在其块内的购电量(块间共享, 块内跨场景共享)
    D[s][n][t]       同上, 储能放电(承诺量: §2.7 其三"块内单一决策, 缺口是唯一兜底")
  情景派生(逐场景实时响应, 支撑 §2.5"SOC 情景相关"):
    A,G,C,GAP,E      A=购电直供负载, G=光伏直供负载, C=光伏充电, GAP=紧急购电缺口, E=储电量
    (购电用于充电 B = Q - A; 弃光由 G+C <= 光伏 隐含)

  物理(逐场景逐段):
    负载平衡  A + G + D + GAP = L·Δt
    光伏平衡  G + C <= PV·Δt          (差量即弃光)
    储能动态  E_t = E_{t-1} + 0.9(B + C) - D/0.9
    界        1200 <= E <= 10800;  B + C <= 5000·Δt;  D <= min(5000·Δt, L·Δt)
    A <= Q(即 B >= 0)

  目标(§2.2/§2.7 其一, 期望口径):
    min  E_ω[ Σ_t λ_t Q_t^ω + 0.5 Σ_t λ_t |Q_t^ω − P_t| + 5 Σ_t λ_t GAP_t^ω ]
         − 0.9 λ_143 E[E_143^ω]
  注意: 0:00-6:00 段 Q_t = P_t 故偏差项自然为 0; 计划购电只在该段按 Q 计费,
       不再另立 Σλ_t P_t(§2.7 其一明确禁止, 否则 37-144 段计划被重复计费)。

结算/落盘(§九 逐日滚动): 每日解出策略后, 按**实测路径**执行——计划取 P, 各块调整取
  实测前缀所落节点的 (Q,D), 由此模拟实测 SOC 并逐日滚动; 上日实测期末 SOC 为次日初值。

用法:
  python P3_solve.py [days]      # days = 评估天数(默认 334 全量); 产物 p3/m5_out/
  python P3_solve.py 5           # 冒烟: 只跑 2.1-2.5
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack, csr_matrix
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
import P3_tree as M4                                        # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
P3 = Path(__file__).resolve().parent
M1, M3 = P3 / "m1_out", P3 / "m3_out"
OUT = P3 / "m5_out"
PRED = BASE / "p2_part1" / "预测结果"

# ---------- 物理常数 ----------
DT = 1.0 / 6.0                                              # 单段小时数
ETA = 0.9
EMIN, EMAX, E0 = 1200.0, 10800.0, 6000.0
PMAX = 5000.0 * DT                                          # 单段最大充/放电量 kWh
GAP_RATE = 5.0                                              # 紧急购电倍率
DEV_RATE = 0.5                                              # 偏差罚倍率
CYCLE_EPS = 1e-4                                            # 吞吐罚(元/kWh), 破"同段充放"退化(§2.6)
NSTAGE = 4
BLK = [(36 * (s - 1), 36 * s) for s in range(1, NSTAGE + 1)]  # 各阶段块 slot 范围

# ---------- 输入 ----------
A_nat = np.load(M1 / "A_nat.npy")                           # (365,24) 整点实际(终点读法)
C_adj = np.load(M3 / "C_adj.npy").astype(np.float64)        # (334,4,24)
C_slot_adj = np.load(M3 / "C_slot_adj.npy").astype(np.float64)  # (4,334,144)
L_adj = np.load(M3 / "L_adj.npy").astype(np.float64)        # (334,144)
P_act = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:145].astype(float).values
L_act = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="小区负载").iloc[:, 1:145].astype(float).values
LAM = pd.read_excel(BASE / "附件" / "附件1.xlsx").iloc[:144, 1].astype(float).values  # (144,) 元/kWh
assert LAM.shape == (144,) and np.isfinite(LAM).all()
I0 = 31                                                     # 2025-02-01 在附件2 的行号


# ============================================================
#  LP 组装(可退化: 会话数 n_sessions / 完美预见 perfect / 冻结 fix)
# ============================================================
def _blocks(n_sessions):
    """各阶段 slot 块。阶段1 = 计划块; 阶段2..n+1 = 调整块。
    n=0 → 计划覆盖全天(无调整); n>=1 → 首块 (0,36), 之后每块 36 段, 末块延到 144。
    例: n=1→[(0,36),(36,144)]; n=2→[…,(36,72),(72,144)]; n=3→[…,(36,72),(72,108),(108,144)]"""
    if n_sessions == 0:
        return [(0, 144)]
    edges = [0, 36] + [36 + 36 * s for s in range(1, n_sessions)] + [144]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def build_lp(scen, tree, soc0, lam, n_sessions=NSTAGE - 1, perfect=(), fix=None):
    """扩展型多阶段 LP。默认 (n_sessions=3, perfect=(), fix=None) 即 M5 主模型(M6 依赖此不变性)。
    n_sessions : 启用的调整会话数 0..3 (0 = 只 0:00 计划, 无调整)                  —— M8 会话族
    perfect    : 完美预见的阶段号集合(2..n_sessions+1)——该阶段按逐场景决策        —— WS 型臂
    fix        : {'P':(144,), 'D1':(blk0,)} 冻结首阶段为该计划(标准 EEV)"""
    K = scen["K"]
    pv = scen["PV_slot"] * DT                               # (K,144) kWh
    ld = scen["L_slot"] * DT
    blk = _blocks(n_sessions)
    nstage = n_sessions + 1
    nodes = tree["nodes"]                                   # nodes[lev], lev=0→阶段2
    perfect = set(int(x) for x in perfect)

    # 各阶段节点集合与 (阶段, 场景)→节点 映射
    stage_nodes = {1: [("root", 0)]}
    node_of = np.zeros((nstage + 1, K), dtype=int)           # node_of[s][ω]
    sizes = {1: [K]}
    for s in range(2, nstage + 1):
        if s in perfect:                                    # 逐场景决策: 每场景一个节点
            stage_nodes[s] = list(range(K))
            node_of[s] = np.arange(K)
            sizes[s] = [1] * K
        else:
            stage_nodes[s] = list(range(len(nodes[s - 2])))
            for n, nd in enumerate(nodes[s - 2]):
                node_of[s, nd["members"]] = n
            sizes[s] = [nd["n"] for nd in nodes[s - 2]]
    stg = np.zeros(144, dtype=int)                          # slot → 阶段
    for s in range(1, nstage + 1):
        stg[blk[s - 1][0]:blk[s - 1][1]] = s

    # ---- 变量表 ----
    idx = {}
    nv = 0
    for t in range(144):
        idx[("P", t)] = nv; nv += 1
    for t in range(*blk[0]):
        idx[("D1", t)] = nv; nv += 1
    for s in range(2, nstage + 1):
        for n in stage_nodes[s]:
            for t in range(*blk[s - 1]):
                idx[("Q", s, n, t)] = nv; nv += 1
                idx[("D", s, n, t)] = nv; nv += 1
                idx[("DP", s, n, t)] = nv; nv += 1           # DEV+ : Q - P
                idx[("DN", s, n, t)] = nv; nv += 1           # DEV- : P - Q
    for w in range(K):
        for t in range(144):
            for k in ("A", "G", "C", "GAP", "E"):
                idx[(k, w, t)] = nv; nv += 1
    NV = nv

    # ---- 目标 ----
    c = np.zeros(NV)
    for t in range(*blk[0]):                                # 计划购电只在其块内计费(§2.7 其一)
        c[idx[("P", t)]] += lam[t]                          # n_sessions>0: 块=(0,36); =0: 全天 144 段
        c[idx[("D1", t)]] += CYCLE_EPS                      # 吞吐罚
    for s in range(2, nstage + 1):
        for n in stage_nodes[s]:
            pn = sizes[s][n] / K
            for t in range(*blk[s - 1]):
                c[idx[("Q", s, n, t)]] += pn * lam[t] + pn * CYCLE_EPS
                c[idx[("D", s, n, t)]] += pn * CYCLE_EPS     # 充放吞吐罚(§2.6 CYCLE_EPS)
                c[idx[("DP", s, n, t)]] += pn * DEV_RATE * lam[t]
                c[idx[("DN", s, n, t)]] += pn * DEV_RATE * lam[t]
    for w in range(K):
        for t in range(144):
            c[idx[("GAP", w, t)]] += (1.0 / K) * GAP_RATE * lam[t]
            c[idx[("A", w, t)]] += -(1.0 / K) * CYCLE_EPS    # 充电 = Q−A+C, 故 −A 计负吞吐
            c[idx[("C", w, t)]] += (1.0 / K) * CYCLE_EPS
        c[idx[("E", w, 143)]] += -(1.0 / K) * ETA * lam[143]

    rows, cols, vals = [], [], []
    r = 0
    b_lo, b_hi = [], []

    def emit(coefs, lo, hi):
        nonlocal r
        for cc, vv in coefs:
            rows.append(r); cols.append(cc); vals.append(vv)
        b_lo.append(lo); b_hi.append(hi); r += 1

    # ---- 逐情景逐段物理 ----
    for w in range(K):
        for t in range(144):
            s = int(stg[t])
            n = node_of[s, w] if s >= 2 else None
            Qc = idx[("P", t)] if s == 1 else idx[("Q", s, n, t)]
            Dc = idx[("D1", t)] if s == 1 else idx[("D", s, n, t)]
            Aw, Gw, Cw = idx[("A", w, t)], idx[("G", w, t)], idx[("C", w, t)]
            Pw = idx[("GAP", w, t)]; Ew = idx[("E", w, t)]
            Ec = idx[("E", w, t - 1)] if t > 0 else None
            # (1) 负载平衡
            emit([(Aw, 1.), (Gw, 1.), (Dc, 1.), (Pw, 1.)], ld[w, t], ld[w, t])
            # (2) 光伏平衡 G + C <= PV
            emit([(Gw, 1.), (Cw, 1.)], -np.inf, pv[w, t])
            # (3) 储能动态 E_t - E_{t-1} - 0.9(Q - A + C) + D/0.9 = soc0·[t==0]
            soc = [(Ew, 1.), (Qc, -ETA), (Aw, ETA), (Cw, -ETA), (Dc, 1. / ETA)]
            if Ec is not None:
                soc.append((Ec, -1.))
            emit(soc, soc0 if t == 0 else 0.0, soc0 if t == 0 else 0.0)
            # (4) 充电功率 B + C <= PMAX
            emit([(Qc, 1.), (Aw, -1.), (Cw, 1.)], -np.inf, PMAX)
            # (5) 放电不超过负载(不反送)
            emit([(Dc, 1.)], -np.inf, ld[w, t])
            # (6) A <= Q
            emit([(Aw, 1.), (Qc, -1.)], -np.inf, 0.0)

    # ---- 偏差罚 ----
    for s in range(2, nstage + 1):
        for n in stage_nodes[s]:
            for t in range(*blk[s - 1]):
                Qc = idx[("Q", s, n, t)]; Pc = idx[("P", t)]
                emit([(idx[("DP", s, n, t)], 1.), (Qc, -1.), (Pc, 1.)], 0.0, np.inf)
                emit([(idx[("DN", s, n, t)], 1.), (Qc, 1.), (Pc, -1.)], 0.0, np.inf)

    # ---- 冻结首阶段(标准 EEV): P 全天, D1 取计划块 ----
    if fix:
        if "P" in fix:
            for t in range(144):
                emit([(idx[("P", t)], 1.)], float(fix["P"][t]), float(fix["P"][t]))
        if "D1" in fix:
            for j, t in enumerate(range(*blk[0])):
                emit([(idx[("D1", t)], 1.)], float(fix["D1"][j]), float(fix["D1"][j]))

    # ---- 界 ----
    bounds = [(0.0, None)] * NV
    for w in range(K):
        for t in range(144):
            bounds[idx[("E", w, t)]] = (EMIN, EMAX)
    for s in range(2, nstage + 1):
        for n in stage_nodes[s]:
            for t in range(*blk[s - 1]):
                bounds[idx[("D", s, n, t)]] = (0.0, PMAX)
    for t in range(*blk[0]):
        bounds[idx[("D1", t)]] = (0.0, PMAX)

    A_all = coo_matrix((vals, (rows, cols)), shape=(r, NV)).tocsr()
    b_lo = np.array(b_lo); b_hi = np.array(b_hi)
    eq = np.isclose(b_lo, b_hi)
    # 等式行
    A_eq, b_eq = A_all[eq], b_lo[eq]
    # 不等式行: 统一成 a·x <= b
    Aub_rows, Aub_b = [], []
    for i in np.where(~eq)[0]:
        row = A_all[i]
        if not np.isfinite(b_hi[i]):                        # 只有下界
            Aub_rows.append(-row); Aub_b.append(-b_lo[i])
        elif not np.isfinite(b_lo[i]):                      # 只有上界
            Aub_rows.append(row); Aub_b.append(b_hi[i])
        else:                                               # 双侧
            Aub_rows.append(row); Aub_b.append(b_hi[i])
            Aub_rows.append(-row); Aub_b.append(-b_lo[i])
    A_ub = vstack(Aub_rows).tocsr(); b_ub = np.array(Aub_b, dtype=float)
    return dict(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                idx=idx, NV=NV, node_of=node_of, sizes=sizes, K=K,
                nstage=nstage, blk=blk, stg=stg)


def solve_day(e, soc0, lam, verbose=False):
    """解第 e 个评估日的四阶段随机规划。e=0 为冷启动(确定性中心单场景)。"""
    if e == 0:
        # 中心十分钟光伏须截零: C_slot_adj 经终偏校正后个别段可为微负, 而 PV 是 G+C≤PV 的上界,
        # 负上界会使 LP 不可行(与 build_scenarios 的截零口径一致)。
        scen = dict(K=1, PV_slot=np.maximum(C_slot_adj[0, e], 0.0)[None, :].copy(),
                    L_slot=L_adj[e][None, :].copy(),
                    idx=np.array([0]), A_hour=C_adj[e, 0][None, :])
        tree = dict(nodes=[[dict(members=np.array([0]), n=1, parent=0)],
                           [dict(members=np.array([0]), n=1, parent=0)],
                           [dict(members=np.array([0]), n=1, parent=0)]],
                    leaf=np.array([0]), root=None)
    else:
        scen = M4.build_scenarios(e)
        tree = M4.build_tree(e, scen)
    lp = build_lp(scen, tree, soc0, lam)
    res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"], b_eq=lp["b_eq"],
                  bounds=lp["bounds"], method="highs")
    if not res.success:
        raise RuntimeError(f"日 {e} LP 失败: {res.message}")
    return scen, tree, lp, res.x


# ============================================================
#  实测路径执行
# ============================================================
def execute_realized(e, x, lp, tree, soc0, qty_pv, qty_ld):
    """按实测路径结算: 计划取 P, 各块调整取实测前缀所落节点的 (Q,D); 模拟实测 SOC。
    返回本日结算量(购电/偏差/紧急)、期末 SOC、以及逐段明细。"""
    idx = lp["idx"]
    P = np.array([x[idx[("P", t)]] for t in range(144)])
    Q = np.empty(144); D = np.empty(144)
    Q[:36] = P[:36]
    D[:36] = [x[idx[("D1", t)]] for t in range(*BLK[0])]
    rout = []
    for s in (2, 3, 4):
        if e == 0:
            n = 0
        else:
            n = M4.route_node(tree, e, s - 1)                # 阶段 s ↔ 会话 s-1 的已实现前缀
        rout.append(n)
        for t in range(*BLK[s - 1]):
            Q[t] = x[idx[("Q", s, n, t)]]
            D[t] = x[idx[("D", s, n, t)]] if x is not None else 0.0
    # --- 逐段实测调度 ---
    E = soc0
    gap = np.zeros(144); charge = np.zeros(144); curt = np.zeros(144)
    A_v = np.zeros(144); B_v = np.zeros(144); C_v = np.zeros(144); G_v = np.zeros(144)
    d_v = np.zeros(144)
    for t in range(144):
        pvt, ldt = qty_pv[t], qty_ld[t]
        # 调度次序: 已承诺购电 Q 已付费, 优先用于负载; 光伏次之(未用的可充电); 再放电; 缺口兜底
        A = min(Q[t], ldt)
        G = min(pvt, ldt - A)
        d_use = min(D[t], max(0.0, ldt - A - G), max(0.0, E - EMIN) * ETA)
        gap[t] = max(0.0, ldt - A - G - d_use)
        B = Q[t] - A                                        # 购电充电(未用于负载的部分)
        C = pvt - G                                         # 光伏盈余充电
        cap = max(0.0, EMAX - E) / ETA
        ch = min(B + C, PMAX, cap)
        curt[t] = max(0.0, (B + C) - ch)
        E = E + ETA * ch - d_use / ETA
        charge[t] = ch
        A_v[t], B_v[t], C_v[t], G_v[t] = A, B, C, G
        d_v[t] = d_use
    buy = float((LAM * Q).sum())
    dev = float((DEV_RATE * LAM * np.abs(Q - P)).sum())
    emg = float((GAP_RATE * LAM * gap).sum())
    return dict(P=P, Q=Q, D=D, gap=gap, charge=charge, curt=curt, E_end=E,
                buy=buy, dev=dev, emg=emg, cost=buy + dev + emg, routes=rout,
                sQ=Q.sum(), sA=A_v.sum(), sB=B_v.sum(), sC=C_v.sum(), sG=G_v.sum(),
                sD=d_v.sum(), sGAP=gap.sum(), sCUR=curt.sum(), sLD=qty_ld.sum(), sPV=qty_pv.sum(),
                duse=d_v, n_simul=int(((d_v > 1e-6) & (charge > 1e-6)).sum()))


# ============================================================
#  result3.xlsx 落盘(题面 §result3 四表; 口径见 m5_model.md §9)
# ============================================================
def _hhmm_slot(t):
    """slot t(0-based, 区间 [10t,10t+10) min) 的起点时刻字符串。"""
    m = 10 * t
    return f"{m // 60}:{m % 60:02d}"


def write_result3(rec, out_path):
    """rec: [{date, p, q, ch, dis, gap, soc0, soc1}, ...] —— 逐日实测路径结算明细。
    模板口径(与问题一 result1 一致): 144 列 = 区间起点 0:10..24:00, 即当日 slot 1..143 + slot 0。"""
    tpl = pd.read_excel(BASE / "附件" / "附件5" / "result3.xlsx", sheet_name="计划购电量", header=None)
    labels = [str(x) for x in tpl.iloc[0, 1:145]]           # 144 个时间段标签
    order = list(range(1, 144)) + [0]                       # 落盘列序: slot1..slot143, slot0
    rows_p, rows_q, rows_cd, rows_em = [], [], [], []
    for r in rec:
        def pack(v):                                        # 144 向量 → 模板列序
            return {labels[j]: float(v[order[j]]) for j in range(144)}
        sp, sq = float(r["p"].sum()), float(r["q"].sum())
        rows_p.append({"日期": r["date"], **pack(r["p"]),
                       "全天购电量": round(sp, 2), "全天购电费": round(float((LAM * r["p"]).sum()), 2)})
        rows_q.append({"日期": r["date"], **pack(r["q"]),
                       "全天购电量": round(sq, 2), "全天购电费": round(float((LAM * r["q"]).sum()), 2)})
        # 充放电量: 6 个四小时块
        blocks = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
        for b, blk in enumerate(blocks):
            sl = slice(24 * b, 24 * (b + 1))
            rows_cd.append({"日期": r["date"] if b == 0 else None, "时间段": blk,
                            "充电量": round(float(r["ch"][sl].sum()), 2),
                            "放电量": round(float(r["dis"][sl].sum()), 2),
                            "时刻": "0:00" if b == 0 else ("24:00" if b == 1 else None),
                            "储电量": None if b > 1 else round(float(r["soc0"] if b == 0 else r["soc1"]), 2)})
        # 紧急购电量: 连续缺口段合并
        g = r["gap"] > 1e-6
        t = 0
        first = True
        while t < 144:
            if g[t]:
                u = t
                while u + 1 < 144 and g[u + 1]:
                    u += 1
                rows_em.append({"日期": r["date"] if first else None,   # 表4 口径: 每天只写一次日期
                                "购电时间段": f"{_hhmm_slot(t)}-{_hhmm_slot(u + 1)}",
                                "购电量": round(float(r["gap"][t:u + 1].sum()), 2)})
                first = False
                t = u + 1
            else:
                t += 1
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        pd.DataFrame(rows_p).to_excel(w, sheet_name="计划购电量", index=False)
        pd.DataFrame(rows_q).to_excel(w, sheet_name="调整购电量", index=False)
        pd.DataFrame(rows_cd).to_excel(w, sheet_name="充放电量", index=False)
        if rows_em:
            pd.DataFrame(rows_em).to_excel(w, sheet_name="紧急购电量", index=False)
        else:
            pd.DataFrame(columns=["日期", "购电时间段", "购电量"]).to_excel(w, sheet_name="紧急购电量", index=False)


# ============================================================
#  主循环(逐日滚动)
# ============================================================
def main():
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 334
    ndays = min(ndays, 334)
    dates = pd.to_datetime(pd.read_excel(BASE / "附件" / "附件2.xlsx",
                                         sheet_name="小区负载").iloc[:, 0]).dt.strftime("%Y-%m-%d").values
    t0 = time.time()
    soc = E0
    rows, det = [], []
    for e in range(ndays):
        scen, tree, lp, x = solve_day(e, soc, LAM)
        rp = P_act[I0 + e] * DT
        rl = L_act[I0 + e] * DT
        ex = execute_realized(e, x, lp, tree, soc, rp, rl)
        det.append(dict(date=dates[I0 + e], p=ex["P"], q=ex["Q"], ch=ex["charge"],
                        dis=ex["duse"], gap=ex["gap"], soc0=soc, soc1=ex["E_end"]))
        rows.append({"e": e, "日期行": I0 + e, "池宽": scen["K"], "SOC初": round(soc, 1),
                     "计划购电费": round(ex["buy"], 2), "偏差费": round(ex["dev"], 2),
                     "紧急费": round(ex["emg"], 2), "日费用": round(ex["cost"], 2),
                     "期末SOC": round(ex["E_end"], 1), "缺口kWh": round(ex["gap"].sum(), 1),
                     "ΣQ": round(ex["sQ"], 0), "ΣA": round(ex["sA"], 0), "ΣB": round(ex["sB"], 0),
                     "ΣG": round(ex["sG"], 0), "ΣC": round(ex["sC"], 0), "ΣD": round(ex["sD"], 0),
                     "Σ弃光": round(ex["sCUR"], 0), "Σ负载": round(ex["sLD"], 0), "Σ光伏": round(ex["sPV"], 0),
                     "落节点": "/".join(map(str, ex["routes"]))})
        soc = ex["E_end"]
        if ndays <= 10 or (e + 1) % 50 == 0:
            print(f"  e={e:3d} 池{scen['K']:2d} SOC {rows[-1]['SOC初']:7.1f}→{rows[-1]['期末SOC']:7.1f} "
                  f"费用 {rows[-1]['日费用']:9.2f} (购{rows[-1]['计划购电费']:8.1f} "
                  f"偏{rows[-1]['偏差费']:6.1f} 急{rows[-1]['紧急费']:8.1f}) "
                  f"缺口 {rows[-1]['缺口kWh']:7.1f}")
    df = pd.DataFrame(rows)
    print(f"\n===== M5 汇总({ndays} 天, {time.time()-t0:.0f}s) =====")
    print(f"  购电费 {df['计划购电费'].sum()/1e4:8.2f} 万元 | 偏差费 {df['偏差费'].sum()/1e4:7.2f} 万元 "
          f"| 紧急费 {df['紧急费'].sum()/1e4:8.2f} 万元")
    print(f"  合计净成本 {df['日费用'].sum()/1e4:8.2f} 万元 | 全年缺口 {df['缺口kWh'].sum():,.0f} kWh "
          f"| 缺口段数 {(df['缺口kWh']>0).sum()}")
    print(f"  SOC 范围 [{df['期末SOC'].min():.0f}, {df['期末SOC'].max():.0f}] kWh")
    OUT.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT / "m5_结果.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="逐日结算", index=False)
    write_result3(det, OUT / "result3.xlsx")
    print(f"  已保存 -> {OUT} (m5_结果.xlsx + result3.xlsx)")


if __name__ == "__main__":
    main()
