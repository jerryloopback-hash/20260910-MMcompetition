# -*- coding: utf-8 -*-
"""
P4C 问题4-3求解: 波动电价下重算问题3 (四阶段随机规划 + 实测路径结算)
================================================================================
架构: import 队友 m5_solve(build_lp/solve_day 原封不动) + m4_tree(情景树, 与电价无关);
结算与落盘 vendor 后参数化电价。价格注入两点:
  计划/调整 LP 目标 λ̂_e  —— 电价预测(v6 EGD / 附件1形状 / 完美预见, 按组切换)
  实测路径结算 λ_e       —— 附件4实际价(交叉验证模式用附件1固定价)
理论依据: D1 目标对 λ 线性, 风险中性+价格与负载/光伏近似独立 ⇒ E_λ[λ]作系数即精确;
slot 空间 = 附件4列空间(M3 相位定案: 标签=时段终点), v6 预测数组直接对位, 无需旋转。

用法:
  python p4c_solve.py            # 交叉验证(复现m5) + 三组对照 + PF下界 + result4-3.xlsx
  python p4c_solve.py 5          # 冒烟5天
  python p4c_solve.py cv         # 仅交叉验证
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                    # C题/
P3 = Path(r"D:\GitRepos\20260910-MMcompetition\C题\p3")   # 队友p3管线(GitRepos树)
sys.path.insert(0, str(P3))
import m5_solve as M5                                 # noqa: E402  (m4_tree 随之可用)

OUT = HERE
SMOKE = len(sys.argv) > 1 and sys.argv[1].isdigit()
CV_ONLY = len(sys.argv) > 1 and sys.argv[1] == "cv"
NDAYS = min(int(sys.argv[1]), 334) if SMOKE else 334

# ---------- 电价输入 ----------
LAM_A1 = M5.LAM                                       # (144,) 附件1固定曲线(slot=终点标签)
a4 = pd.read_excel(BASE / "附件" / "附件4.xlsx", index_col=0)
a4.columns = range(144)
LAM_SET4 = a4.values.astype(float)[31:31 + 334]       # (334,144) 附件4实际价, 行e↔评估日e
PRED_PRICE = np.load(BASE / "预测结果" / "price_pred_ens_v6.npy")  # (334,144) v6 EGD电价预测
assert PRED_PRICE.shape == (334, 144) and np.isfinite(PRED_PRICE).all()
assert LAM_SET4.shape == (334, 144) and np.isfinite(LAM_SET4).all()

PLAN_SRC = {"egd": PRED_PRICE, "fixed": None, "perfect": LAM_SET4}   # fixed→LAM_A1


def plan_lam(src, e):
    return LAM_A1 if src == "fixed" else PLAN_SRC[src][e]


# ---------- 情景树缓存(跨组复用; m4/m3 确定性, 固定种子) ----------
_SCEN_CACHE = {}


def get_scen_tree(e):
    if e not in _SCEN_CACHE:
        if e == 0:
            scen = dict(K=1, PV_slot=np.maximum(M5.C_slot_adj[0, 0], 0.0)[None, :].copy(),
                        L_slot=M5.L_adj[0][None, :].copy(),
                        idx=np.array([0]), A_hour=M5.C_adj[0, 0][None, :])
            tree = dict(nodes=[[dict(members=np.array([0]), n=1, parent=0)],
                               [dict(members=np.array([0]), n=1, parent=0)],
                               [dict(members=np.array([0]), n=1, parent=0)]],
                        leaf=np.array([0]), root=None)
        else:
            scen = M5.M4.build_scenarios(e)
            tree = M5.M4.build_tree(e, scen)
        _SCEN_CACHE[e] = (scen, tree)
    return _SCEN_CACHE[e]


def solve_day_v4(e, soc0, lam):
    scen, tree = get_scen_tree(e)
    lp = M5.build_lp(scen, tree, soc0, lam)
    from scipy.optimize import linprog
    res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                  b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs")
    if not res.success:
        raise RuntimeError(f"日 {e} LP 失败: {res.message}")
    return scen, tree, lp, res.x


# ---------- 实测路径结算(vendor m5.execute_realized, 电价参数化) ----------
def execute_realized_v4(e, x, lp, tree, soc0, qty_pv, qty_ld, lam_set):
    idx = lp["idx"]
    P = np.array([x[idx[("P", t)]] for t in range(144)])
    Q = np.empty(144); D = np.empty(144)
    Q[:36] = P[:36]
    D[:36] = [x[idx[("D1", t)]] for t in range(*M5.BLK[0])]
    rout = []
    for s in (2, 3, 4):
        n = 0 if e == 0 else M5.M4.route_node(tree, e, s - 1)
        rout.append(n)
        for t in range(*M5.BLK[s - 1]):
            Q[t] = x[idx[("Q", s, n, t)]]
            D[t] = x[idx[("D", s, n, t)]]
    E = soc0
    gap = np.zeros(144); charge = np.zeros(144)
    A_v = np.zeros(144); B_v = np.zeros(144); C_v = np.zeros(144); G_v = np.zeros(144)
    d_v = np.zeros(144)
    for t in range(144):
        pvt, ldt = qty_pv[t], qty_ld[t]
        A = min(Q[t], ldt)
        G = min(pvt, ldt - A)
        d_use = min(D[t], max(0.0, ldt - A - G), max(0.0, E - M5.EMIN) * M5.ETA)
        gap[t] = max(0.0, ldt - A - G - d_use)
        B = Q[t] - A
        C = pvt - G
        cap = max(0.0, M5.EMAX - E) / M5.ETA
        ch = min(B + C, M5.PMAX, cap)
        E = E + M5.ETA * ch - d_use / M5.ETA
        charge[t] = ch
        A_v[t], B_v[t], C_v[t], G_v[t] = A, B, C, G
        d_v[t] = d_use
    buy = float((lam_set * Q).sum())
    dev = float((M5.DEV_RATE * lam_set * np.abs(Q - P)).sum())
    emg = float((M5.GAP_RATE * lam_set * gap).sum())
    return dict(P=P, Q=Q, D=D, gap=gap, charge=charge, E_end=E,
                buy=buy, dev=dev, emg=emg, cost=buy + dev + emg, routes=rout,
                sQ=Q.sum(), sGAP=gap.sum(), sLD=qty_ld.sum(), sPV=qty_pv.sum(),
                duse=d_v)


# ---------- result4-3 落盘(vendor m5.write_result3, 电价参数化+模板换result4-3) ----------
def _hhmm_slot(t):
    m = 10 * t
    return f"{m // 60}:{m % 60:02d}"


def write_result43(rec, lam_set_all, out_path):
    tpl = pd.read_excel(BASE / "附件" / "附件5" / "result4-3.xlsx",
                        sheet_name="计划购电量", header=None)
    labels = [str(x) for x in tpl.iloc[0, 1:145]]
    order = list(range(1, 144)) + [0]                 # 落盘列序: slot1..143, slot0
    rows_p, rows_q, rows_cd, rows_em = [], [], [], []
    for k, r in enumerate(rec):
        lam_e = lam_set_all[k]

        def pack(v):
            return {labels[j]: float(v[order[j]]) for j in range(144)}
        sp, sq = float(r["p"].sum()), float(r["q"].sum())
        rows_p.append({"日期": r["date"], **pack(r["p"]),
                       "全天购电量": round(sp, 2),
                       "全天购电费": round(float((lam_e * r["p"]).sum()), 2)})
        rows_q.append({"日期": r["date"], **pack(r["q"]),
                       "全天购电量": round(sq, 2),
                       "全天购电费": round(float((lam_e * r["q"]).sum()), 2)})
        blocks = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
        for b in range(6):
            sl = slice(24 * b, 24 * (b + 1))
            rows_cd.append({"日期": r["date"] if b == 0 else None, "时间段": blocks[b],
                            "充电量": round(float(r["ch"][sl].sum()), 2),
                            "放电量": round(float(r["dis"][sl].sum()), 2),
                            "时刻": "0:00" if b == 0 else ("24:00" if b == 1 else None),
                            "储电量": None if b > 1 else round(float(r["soc0"] if b == 0 else r["soc1"]), 2)})
        g = r["gap"] > 1e-6
        t = 0
        first = True
        while t < 144:
            if g[t]:
                u = t
                while u + 1 < 144 and g[u + 1]:
                    u += 1
                rows_em.append({"日期": r["date"] if first else None,
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


# ---------- 单组全年 ----------
def run_group(src, settle="actual4", tag="", collect=False, verbose=True):
    """src: 计划层电价来源 egd|fixed|perfect; settle: actual4|fixed(交叉验证)."""
    lam_set_all = LAM_SET4 if settle == "actual4" else np.tile(LAM_A1, (334, 1))
    dates = M5_MAIN_DATES
    soc = M5.E0
    rows, det = [], []
    t0 = time.time()
    for e in range(NDAYS):
        lam_p = plan_lam(src, e)
        scen, tree, lp, x = solve_day_v4(e, soc, lam_p)
        rp = M5.P_act[M5.I0 + e] * M5.DT
        rl = M5.L_act[M5.I0 + e] * M5.DT
        ex = execute_realized_v4(e, x, lp, tree, soc, rp, rl, lam_set_all[e])
        if collect:
            det.append(dict(date=dates[M5.I0 + e], p=ex["P"], q=ex["Q"], ch=ex["charge"],
                            dis=ex["duse"], gap=ex["gap"], soc0=soc, soc1=ex["E_end"]))
        rows.append({"e": e, "日期": dates[M5.I0 + e], "SOC初": round(soc, 1),
                     "计划购电费": round(ex["buy"], 2), "偏差费": round(ex["dev"], 2),
                     "紧急费": round(ex["emg"], 2), "日费用": round(ex["cost"], 2),
                     "期末SOC": round(ex["E_end"], 1), "缺口kWh": round(ex["gap"].sum(), 1),
                     "ΣQ": round(ex["sQ"], 0), "Σ负载": round(ex["sLD"], 0), "Σ光伏": round(ex["sPV"], 0)})
        soc = ex["E_end"]
        if verbose and ((e + 1) % 100 == 0 or e == NDAYS - 1):
            print(f"   [{tag}] {e+1}/{NDAYS} 累计购{sum(r['计划购电费'] for r in rows):,.0f} "
                  f"偏{sum(r['偏差费'] for r in rows):,.0f} 急{sum(r['紧急费'] for r in rows):,.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return dict(rows=pd.DataFrame(rows), det=det,
                buy=sum(r["计划购电费"] for r in rows), dev=sum(r["偏差费"] for r in rows),
                emg=sum(r["紧急费"] for r in rows), total=sum(r["日费用"] for r in rows),
                gapE=sum(r["缺口kWh"] for r in rows))


# ---------- PF 完美预见下界(实际负载/光伏, 实际电价, 单确定性LP/日) ----------
def run_pf():
    from scipy.optimize import linprog
    dates = M5_MAIN_DATES
    soc = M5.E0
    rows = []
    t0 = time.time()
    for e in range(NDAYS):
        scen = dict(K=1, PV_slot=M5.P_act[M5.I0 + e][None, :].copy(),
                    L_slot=M5.L_act[M5.I0 + e][None, :].copy(),
                    idx=np.array([0]), A_hour=M5.P_act[M5.I0 + e][None, :])
        tree = dict(nodes=[[dict(members=np.array([0]), n=1, parent=0)]], leaf=np.array([0]), root=None)
        lp = M5.build_lp(scen, tree, soc, LAM_SET4[e], n_sessions=0)
        res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                      b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs")
        assert res.success, res.message
        x = res.x
        buy = float(LAM_SET4[e] @ np.array([x[lp["idx"][("P", t)]] for t in range(144)]))
        e_end = float(x[lp["idx"][("E", 0, 143)]])
        rows.append({"e": e, "日期": dates[M5.I0 + e], "计划购电费": round(buy, 2),
                     "偏差费": 0.0, "紧急费": 0.0, "日费用": round(buy, 2),
                     "期末SOC": round(e_end, 1), "缺口kWh": 0.0})
        soc = e_end
        if (e + 1) % 100 == 0:
            print(f"   [PF] {e+1}/{NDAYS} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    return dict(rows=df, buy=df["计划购电费"].sum(), dev=0.0, emg=0.0,
                total=df["日费用"].sum(), gapE=0.0)


M5_MAIN_DATES = pd.to_datetime(pd.read_excel(M5.BASE / "附件" / "附件2.xlsx",
                                             sheet_name="小区负载").iloc[:, 0]).dt.strftime("%Y-%m-%d").values

# ---------- 主流程 ----------
if __name__ == "__main__":
    print(f"P4C 问题4-3: 计划层电价=v6 EGD预测(主配置) | 结算=附件4实际价 | 树/中心=m3/m4原样")

    # 0) 交叉验证: fixed/fixed 应逐日复现 m5 (RP=1468.63万)
    print("\n=== 交叉验证: 固定电价复现问题3 m5 ===")
    cv = run_group("fixed", settle="fixed", tag="复现P3", verbose=False)
    print(f"   本管线: 购 {cv['buy']:,.2f} 偏 {cv['dev']:,.2f} 急 {cv['emg']:,.2f} "
          f"合计 {cv['total']:,.2f} 元 | 缺口 {cv['gapE']:,.1f} kWh")
    ref = pd.read_excel(P3 / "m5_out" / "m5_结果.xlsx", sheet_name="逐日结算")
    print(f"   m5存档: 购 {ref['计划购电费'].sum():,.2f} 偏 {ref['偏差费'].sum():,.2f} "
          f"急 {ref['紧急费'].sum():,.2f} 合计 {ref['日费用'].sum():,.2f} 元 "
          f"| 缺口 {ref['缺口kWh'].sum():,.1f} kWh")
    dmax = max(abs(cv["buy"] - ref["计划购电费"].sum()), abs(cv["dev"] - ref["偏差费"].sum()),
               abs(cv["emg"] - ref["紧急费"].sum()), abs(cv["total"] - ref["日费用"].sum()))
    print(f"   差异: {dmax:.4f} 元")
    if not SMOKE:
        assert dmax < 1.0, "交叉验证失败: 与 m5 存档不一致"
        print("   交叉验证通过 ✓ — p4c 与 m5 完全同构")
    if CV_ONLY:
        sys.exit(0)

    # 1) 三组电价来源对照 (结算一律附件4实际价)
    results = {}
    for src, gname in [("egd", "本文EGD电价预测"), ("fixed", "附件1形状(忽视波动)"),
                       ("perfect", "完美预见电价")]:
        print(f"\n=== 组: {gname} ===")
        results[gname] = run_group(src, settle="actual4", tag=gname, collect=(src == "egd"))
        r = results[gname]
        print(f"   净(费合计) {r['total']:>12,.0f} 元 = 购 {r['buy']:,.0f} + 偏 {r['dev']:,.0f} + 急 {r['emg']:,.0f}"
              f" | 缺口 {r['gapE']:,.0f} kWh")

    print("\n=== PF 完美预见下界(实际负载/光伏+实际电价) ===")
    pf = run_pf()
    print(f"   下界 {pf['total']:>12,.0f} 元")

    df_all = pd.DataFrame([
        {"组": g, **{k: results[g][k] for k in ("buy", "dev", "emg", "total", "gapE")}}
        for g in results] + [{"组": "完美预见下界", **{k: pf[k] for k in ("buy", "dev", "emg", "total", "gapE")}}])
    print("\n===== 三组电价来源对比(净成本=购+偏+急) =====")
    print(df_all.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    if not SMOKE:
        with pd.ExcelWriter(OUT / "p4c_三组电价对比.xlsx", engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="组间对比", index=False)
            for g in results:
                results[g]["rows"].to_excel(w, sheet_name=f"逐日_{g}"[:31], index=False)
        print(f"\n三组对比已保存 -> {OUT / 'p4c_三组电价对比.xlsx'}")
        with pd.ExcelWriter(OUT / "p4c_结果.xlsx", engine="openpyxl") as w:
            results["本文EGD电价预测"]["rows"].to_excel(w, sheet_name="逐日结算_G1", index=False)
            pf["rows"].to_excel(w, sheet_name="逐日结算_PF", index=False)
        write_result43(results["本文EGD电价预测"]["det"], LAM_SET4, OUT / "result4-3.xlsx")
        print(f"result4-3.xlsx 已保存 -> {OUT / 'result4-3.xlsx'}")
