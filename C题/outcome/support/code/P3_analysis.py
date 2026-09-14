# -*- coding: utf-8 -*-
"""
P3-M6 分析层:信息价值分解(EVPI/VSS)、会话信息价值、会话启用族、风险画像
================================================================================
依据 p3_plan.md §5.1–5.5 / D7 / 里程碑 M6。

口径(三条硬约束, 勿违背):
 A. 分析窗口 = 3.1–12.31(e=28..333, 306 天, 供体池满 28)。池宽 ≤1 的日子 VSS 恒 0,
    混进分母会系统性压低 VSS;result3 交付口径仍为 2.1–12.31(见 m5_model.md §9)。
 B. 所有臂共用同一条实测 SOC 初值链(soc0 取 M5 主模型逐日实测期末, m5_结果.xlsx
    '期末SOC'),保证各臂在"同一起点、同一结算"下可比。
 C. 各臂的费用一律取 LP 目标的期望值(与 RP 同口径, 含期末储能价值 -0.9λ143·E143),
    故 EVPI/VSS 均为"净成本"意义下的差。

定义:
 RP    = 四阶段主模型最优期望费用(306 天均值)。                 [n_sessions=3]
 WS    = 逐情景完美预见的下界(每情景单独确定性最优的均值)。       [n_sessions=0, 逐情景]
 EEV   = 先按 0:00 中心拍确定性计划(P 全天 + D1 计划块), 冻结后仅优化追索(阶段 2–4)。
 EVPI  = RP − WS。
 VSS   = EEV − RP。
 会话信息价值(完美预见后缀族, W_k):W_1=仅 18:00 见证未来; W_2=12:00+18:00;
   W_3=6:00+12:00+18:00; W_4=WS(连 0:00 计划也完美)。则
     18:00 上限 = W_0−W_1, 12:00 = W_1−W_2, 6:00 = W_2−W_3, 0:00计划 = W_3−W_4,
     且 Σ = W_0−W_4 = EVPI(严格分割)。
 会话启用族(§5.5/M8, 实际预报):k=0..3 次调整, RP_k = 启用 k 次调整的随机规划最优。
 风险画像(§5.4):在 RP 最优解上取逐情景净费用分布 → P50/P90/P95/CVaR95。

用法: python P3_analysis.py [days]      # days=分析天数(默认 306 全窗); 产物 p3/m6_out/
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
P3 = Path(__file__).resolve().parent
OUT = P3 / "m6_out"

I0 = 31                                  # 2025-02-01 在附件2 的行号(e=0 ↔ 行 31)
E_FULL = 334                             # 2.1–12.31 的天数
E_ANA0 = 28                              # 3.1 (e=28) —— 分析窗起点(池宽满)
E_ANA1 = E_FULL - 1                      # 12.31 (e=333)
DEV_RATE = 0.5
GAP_RATE = 5.0
ETA = 0.9

# M5 报告值(万元), 用于瀑布基线的一致性核对
Q2_TOTAL = 1638.75                       # 问题二日前计划(购 1225.41 + 急 413.39)
Q2_BUY, Q2_EMG = 1225.41, 413.39
NOSTOR_TOTAL = 1640.73                   # 无储能完美净额基线


# ============================================================
#  单日各臂(worker)
# ============================================================
def _solve_lp(lp):
    res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"], b_eq=lp["b_eq"],
                  bounds=lp["bounds"], method="highs")
    if not res.success:
        raise RuntimeError(f"LP 失败: {res.message}")
    return res


def _obj(e, soc0, scen, tree, n_sessions, perfect=(), fix=None):
    """解一条臂, 返回期望目标值与解向量。"""
    import P3_solve as M5
    lp = M5.build_lp(scen, tree, soc0, M5.LAM, n_sessions=n_sessions, perfect=perfect, fix=fix)
    res = _solve_lp(lp)
    return float(res.fun), res.x, lp


def _scen_costs(x, lp, K, lam):
    """在给定解上取逐情景净费用(与目标同口径: 含期末储能价值)。
    用途: 风险画像。恒等式: mean_w cost_w ≈ 目标值(仅差 CYCLE_EPS 量级)。"""
    idx = lp["idx"]; stg = lp["stg"]; node_of = lp["node_of"]
    P = np.array([x[idx[("P", t)]] for t in range(144)])
    q = np.zeros((K, 144)); gap = np.zeros((K, 144)); E144 = np.zeros(K)
    for w in range(K):
        for t in range(144):
            s = int(stg[t])
            q[w, t] = P[t] if s == 1 else x[idx[("Q", s, node_of[s, w], t)]]
            gap[w, t] = x[idx[("GAP", w, t)]]
        E144[w] = x[idx[("E", w, 143)]]
    dev = DEV_RATE * lam[None, :] * np.abs(q - P[None, :])         # 0.5λ|q−p|
    return (lam[None, :] * q).sum(axis=1) + dev.sum(axis=1) + \
           (GAP_RATE * lam[None, :] * gap).sum(axis=1) - ETA * lam[143] * E144


def execute_plan(P, D, pv, ld, soc0, lam):
    """按 M5 的执行/结算规则, 在一个情景上结算**冻结计划**(P 购电, D 储能)。
    调度次序与 P3_solve.execute_realized 一致(购电优先供载→光伏→放电→缺口兜底),
    缺口兜底保证永不不可行。返回该情景净费用(含期末储能价值)与期末 SOC。"""
    import P3_solve as M5
    E = soc0; gap = np.zeros(144)
    for t in range(144):
        A = min(P[t], ld[t])
        G = min(pv[t], max(0.0, ld[t] - A))
        d_use = min(D[t], max(0.0, ld[t] - A - G), max(0.0, E - M5.EMIN) * M5.ETA)
        gap[t] = max(0.0, ld[t] - A - G - d_use)
        cap = max(0.0, M5.EMAX - E) / M5.ETA
        E = E + M5.ETA * min((P[t] - A) + (pv[t] - G), M5.PMAX, cap) - d_use / M5.ETA
    return (lam * P).sum() + GAP_RATE * (lam * gap).sum() - M5.ETA * lam[143] * E, E


def run_day(e, soc0):
    """解一个评估日的全部 M6 臂, 返回 dict。"""
    import P3_tree as M4
    import P3_solve as M5
    lam = M5.LAM
    scen = M4.build_scenarios(e)
    tree = M4.build_tree(e, scen)
    K = scen["K"]
    out = {"e": e, "K": K, "soc0": soc0}

    # ---- RP(主模型) ----
    f_rp, x_rp, lp_rp = _obj(e, soc0, scen, tree, 3)
    out["RP"] = f_rp
    out["RP_scen"] = _scen_costs(x_rp, lp_rp, K, lam)
    out["RP_check"] = float(np.mean(out["RP_scen"]))                # 恒等式自检

    # ---- WS: 逐情景完美预见 ----
    ws = np.empty(K)
    for w in range(K):
        s1 = dict(K=1, PV_slot=scen["PV_slot"][w][None, :], L_slot=scen["L_slot"][w][None, :])
        ws[w], _, _ = _obj(e, soc0, s1, dict(nodes=[]), 0)
    out["WS"] = float(ws.mean())
    out["WS_scen"] = ws

    # ---- 完美预见后缀族 W_1..W_3(W_0=RP, W_4=WS) ----
    for k, pf in ((1, (4,)), (2, (3, 4)), (3, (2, 3, 4))):
        out[f"W{k}"], _, _ = _obj(e, soc0, scen, tree, 3, perfect=pf)
    out["W0"] = f_rp; out["W4"] = out["WS"]

    # ---- 会话启用族 RP_0..RP_2(RP_3=RP) ----
    for k in (0, 1, 2):
        out[f"RP{k}"], _, _ = _obj(e, soc0, scen, tree, k)
    out["RP3"] = f_rp

    # ---- EEV: 0:00 中心确定性计划, 冻结后按实测执行规则逐情景结算 ----
    # 中心光伏须截零(与 build_scenarios 同口径): C_slot_adj 经终偏校正后个别段微负,
    # 而 PV 是 G+C≤PV 的上界, 负上界使 LP 不可行。
    # 注: 直接冻结 P 的追索型 LP 会不可行(低负载情景无法在 5000kW 并网功率内吸收承诺购电),
    #     故 EEV 取"计划冻结 + 缺口兜底"的评估口径(与 D1 的结算精神一致, 且恒可行)。
    s1 = dict(K=1, PV_slot=np.maximum(M4.C_slot_adj[0, e], 0.0)[None, :], L_slot=M4.L_adj[e][None, :])
    _, xc, lpc = _obj(e, soc0, s1, dict(nodes=[]), 0)
    Pc = np.array([xc[lpc["idx"][("P", t)]] for t in range(144)])
    Dc = np.array([xc[lpc["idx"][("D1", t)]] for t in range(144)])
    eev = np.empty(K)
    for w in range(K):
        eev[w], _ = execute_plan(Pc, Dc, scen["PV_slot"][w] * M5.DT,
                                 scen["L_slot"][w] * M5.DT, soc0, lam)
    out["EEV"] = float(eev.mean())
    out["EEV_scen"] = eev
    # 同一冻结计划在**实测路径**上的结算(与 M5 头号数字同口径, 供样本外对照)
    out["EEV_real"], _ = execute_plan(Pc, Dc, M5.P_act[I0 + e] * M5.DT,
                                      M5.L_act[I0 + e] * M5.DT, soc0, lam)
    return out


# ============================================================
#  主流程
# ============================================================
def main():
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else (E_ANA1 - E_ANA0 + 1)
    ndays = min(ndays, E_ANA1 - E_ANA0 + 1)
    days = list(range(E_ANA0, E_ANA0 + ndays))

    # SOC 初值链: 取 M5 主模型逐日实测期末
    m5df = pd.read_excel(OUT.parent / "m5_out" / "m5_结果.xlsx")
    soc_map = dict(zip(m5df["e"].values, m5df["SOC初"].values))
    soc0 = np.array([soc_map[e] for e in days])
    assert all(np.isfinite(soc0))
    print(f"===== M6 分析窗 {days[0]}..{days[-1]} ({ndays} 天) =====")
    print(f"  SOC 初值范围 [{soc0.min():.0f}, {soc0.max():.0f}] kWh (取自 M5 实测链)")

    t0 = time.time()
    res = []
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6     # 每个 worker 需导入 scipy(约 0.5GB), 需限并发
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, r in enumerate(ex.map(run_day, days, soc0, chunksize=2)):
            res.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(days):
                print(f"  {i+1}/{len(days)}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_scen")} for r in res])

    # 恒等式自检: 逐情景净费用的均值 ≡ 目标值(差为 CYCLE_EPS 量级)
    gap_ident = float(np.max(np.abs(df["RP"] - df["RP_check"])))
    print(f"\n[自检] 逐情景净费用均值 vs LP 目标 最大偏差 {gap_ident:.4f} 元 (应 ~0)")
    print(f"[自检] W 族单调 (W0≥W1≥W2≥W3≥W4): "
          f"{'OK' if (df.W0.ge(df.W1-1e-6)&df.W1.ge(df.W2-1e-6)&df.W2.ge(df.W3-1e-6)&df.W3.ge(df.W4-1e-6)).all() else '**违反**'}")
    print(f"[自检] 会话族单调 (RP0≥RP1≥RP2≥RP3): "
          f"{'OK' if (df.RP0.ge(df.RP1-1e-6)&df.RP1.ge(df.RP2-1e-6)&df.RP2.ge(df.RP3-1e-6)).all() else '**违反**'}")
    print(f"[自检] EEV≥RP: {'OK' if (df.EEV >= df.RP-1e-6).all() else '**违反**'} | "
          f"WS≤RP: {'OK' if (df.WS <= df.RP+1e-6).all() else '**违反**'}")

    OUT.mkdir(exist_ok=True)
    df.to_excel(OUT / "m6_逐日.xlsx", index=False)
    np.save(OUT / "RP_scen.npy", np.array([r["RP_scen"] for r in res]))
    np.save(OUT / "WS_scen.npy", np.array([r["WS_scen"] for r in res]))
    np.save(OUT / "days.npy", np.array(days))
    print(f"\n逐日结果已保存 -> {OUT / 'm6_逐日.xlsx'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
