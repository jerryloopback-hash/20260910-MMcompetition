# -*- coding: utf-8 -*-
"""
P4B 求解器: 波动电价下重算问题2 (问题4-2)  v1
================================================
口径继承 p2b_solve.py v3 全部规则(计划LP同构/严格执行结算/缺口5λ硬约束/E0滚动),
唯一变化: 电价由附件1固定曲线 → 附件4逐日波动曲线, 且计划层电价与结算层电价分离:
  计划层 λ̂_d: 电价预测(EGD模型池 v6, I_k映射) —— 0:00 决策可用信息
  结算层 λ_d:  附件4 实际电价(I_k映射) —— 计划费、5倍紧急费、期末储能价值均按实际价落账
三组电价来源对照(负载/光伏预测固定为问题2模型池组v5 EGD集成, 同管线 isolating 电价预测价值):
  G1 本文EGD电价预测 | G2 附件1形状(忽视波动) | G3 完美预见电价 | PF 完美预见下界(全信息)
决策分位(残差经验分位, 负载P_τ/光伏P_{1-τ})保留 --tau 遍历开关, 波动电价下重选 τ*。
用法:
  python p4b_solve.py                # 三组对照 + τ遍历 + result4-2.xlsx
  python p4b_solve.py smoke          # 5天冒烟
  python p4b_solve.py tau 0.79 0.86  # 仅τ细分遍历
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parent.parent           # C题/
ATT2 = BASE / "附件" / "附件2.xlsx"
ATT4 = BASE / "附件" / "附件4.xlsx"
PRED = BASE / "预测结果"
OUT = Path(__file__).resolve().parent

DT = 1 / 6.0; ETA = 0.9; T = 144; EMIN, EMAX = 1200.0, 10800.0; PMAX = 5000.0
CYCLE_EPS = 1e-4
WIN = 56; K0 = 7.0                                      # 残差池窗/池深收缩(同p2b)
ROLLING_WINDOW = 30                                     # 残差分位窗(同_quantile_v5)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
TAU_ONLY = len(sys.argv) > 3 and sys.argv[1] == "tau"
NDAYS = 5 if SMOKE else 334
N = NDAYS

# ---------- 1. 数据 ----------
d1 = pd.read_excel(BASE / "附件" / "附件1.xlsx", sheet_name=0)
pr = pd.to_numeric(d1.iloc[:, 1], errors="coerce").values.astype(float)
assert len(pr) == T and not np.isnan(pr).any()
lam_fixed = np.concatenate([pr[143:], pr[:143]])        # 附件1 I_k映射(恒定, G2用)

df_l = pd.read_excel(ATT2, sheet_name="小区负载")
df_p = pd.read_excel(ATT2, sheet_name="光伏发电实际功率")
dates = pd.to_datetime(df_l.iloc[:, 0])
Lmat = df_l.iloc[:, 1:1+T].astype(float).values
Pmat = df_p.iloc[:, 1:1+T].astype(float).values
assert Lmat.shape == (365, T)
i_feb1 = int(np.where(dates.values == np.datetime64("2025-02-01"))[0][0])
assert i_feb1 == 31

a4 = pd.read_excel(ATT4, index_col=0)
a4.columns = range(T)
p4mat = a4.values.astype(float)                          # (365,144) 附件4实际电价
assert p4mat.shape == (365, T) and np.isfinite(p4mat).all()

ActL = Lmat[i_feb1:i_feb1+334]
ActP = Pmat[i_feb1:i_feb1+334]

# 结算层实际电价(I_k映射)与计划层电价预测(I_k映射)
settle_lam = np.zeros((334, T))
for d in range(334):
    r = i_feb1 + d
    settle_lam[d] = np.concatenate([p4mat[r-1, 143:144], p4mat[r, 0:143]])
plan_lam = np.load(PRED / "price_pred_plan_lam_v6.npy")
assert plan_lam.shape == (334, T) and np.isfinite(plan_lam).all()

load_point = np.load(PRED / "load_pred_ens_v5.npy")
pv_point = np.load(PRED / "pv_pred_ens_v5.npy")
assert load_point.shape == (334, T) and pv_point.shape == (334, T)

# ---------- 2. LP 结构(A_eq/A_ub 只建一次; 目标系数逐日随 λ̂ 重建) ----------
NV = T * 6
idx = {k: np.arange(j, NV, 6) for j, k in enumerate("abgcdq")}


def soc_row(u):
    r = np.zeros(NV)
    r[idx["b"][:u+1]] += ETA*DT; r[idx["c"][:u+1]] += ETA*DT; r[idx["d"][:u+1]] -= DT/ETA
    return r


A_eq = np.zeros((2*T, NV))
for t in range(T):
    A_eq[2*t,   idx["a"][t]] = 1; A_eq[2*t,   idx["g"][t]] = 1; A_eq[2*t, idx["d"][t]] = 1
    A_eq[2*t+1, idx["g"][t]] = 1; A_eq[2*t+1, idx["c"][t]] = 1; A_eq[2*t+1, idx["q"][t]] = 1
A_ub = np.zeros((4*T, NV))
for t in range(T):
    A_ub[2*t] = soc_row(t); A_ub[2*t+1] = -soc_row(t)
for t in range(T):
    A_ub[2*T+2*t,   idx["b"][t]] = 1; A_ub[2*T+2*t,   idx["c"][t]] = 1
    A_ub[2*T+2*t+1, idx["d"][t]] = 1
_SOC_LAST = soc_row(T-1)
_b_const = np.concatenate([np.zeros(2*T), np.full(2*T, PMAX)])   # b_ub中与决策无关的功率上限部分


def solve_day(Lf, Pf, E0, lam_plan):
    c_obj = np.zeros(NV)
    c_obj[idx["a"]] = lam_plan * DT
    c_obj[idx["b"]] = lam_plan * DT + CYCLE_EPS * DT
    c_obj[idx["c"]] = CYCLE_EPS * DT
    c_obj[idx["d"]] = CYCLE_EPS * DT
    c_obj -= ETA * lam_plan[-1] * _SOC_LAST              # 日末储能价值按计划口径估值
    b_eq = np.empty(2*T)
    b_eq[0:2*T:2] = Lf; b_eq[1:2*T:2] = Pf
    b_ub = np.empty(4*T)
    b_ub[0:2*T:2] = EMAX - E0; b_ub[1:2*T:2] = E0 - EMIN; b_ub[2*T:] = PMAX
    r = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)]*NV, method="highs")
    assert r.success, r.message
    x = r.x
    return x[idx["a"]], x[idx["b"]], x[idx["g"]], x[idx["c"]], x[idx["d"]]


# ---------- 3. 严格执行结算(缺口5λ按实际价; 回传逐时段数组供result表) ----------
def settle(a, b, gp, dp, Lact, Pact, E0, lam_act):
    E = E0; pcost = 0.0; ecost = 0.0; gapE = 0.0; ngap = 0
    smin = E; smax = E
    ch_block = np.zeros(6); dis_block = np.zeros(6)
    chin_e = np.empty(T); dis_e = np.empty(T); gap_e = np.empty(T)
    for t in range(T):
        gr = gp[t] if gp[t] < Pact[t] else Pact[t]
        da = dp[t]
        cap = ETA*(E - EMIN)/DT
        if da > cap: da = cap
        S = a[t] + gr + da
        gap = Lact[t] - S
        gap_kw = gap if gap > 0 else 0.0
        if gap_kw > 0:
            ecost += 5*lam_act[t]*gap_kw*DT; gapE += gap_kw*DT; ngap += 1
        cr = Pact[t] - gp[t]
        if cr < 0: cr = 0.0
        chin = (b[t] + cr)*DT
        if chin > PMAX*DT: chin = PMAX*DT
        cap2 = (EMAX - E)/ETA
        if chin > cap2: chin = cap2
        chin_e[t] = chin; dis_e[t] = da*DT; gap_e[t] = gap_kw*DT
        E += ETA*chin - da*DT/ETA
        blk = min(t // 24, 5)
        ch_block[blk] += chin; dis_block[blk] += da*DT
        if E < smin: smin = E
        if E > smax: smax = E
        pcost += lam_act[t]*(a[t] + b[t])*DT
    return dict(pcost=pcost, ecost=ecost, gapE=gapE, ngap=ngap, smin=smin, smax=smax,
                Eend=E, E0=E0, ch_block=ch_block, dis_block=dis_block,
                chin_e=chin_e, dis_e=dis_e, gap_e=gap_e)


# ---------- 4. 残差分位曲线(与p2b_quantile_sweep.rolling_quantile_at逐字同口径) ----------
ROLLING_WINDOW = 30
NIGHT = (((np.arange(T) + 1) * 10.0 / 60.0) <= 6.0) | (((np.arange(T) + 1) * 10.0 / 60.0) >= 20.0)
_QCACHE = {}


def rolling_quantile_curve(point_pred, true_mat, q, is_pv):
    """调整曲线在ti空间构造(30天滚动残差经验分位, 冷启动persistence), PV夜间置零, clip≥0。"""
    pred_idx = np.arange(i_feb1, i_feb1 + 334)
    pred_set = set(pred_idx.tolist())
    out = np.zeros((334, T))
    for k, i_day in enumerate(pred_idx):
        lo = max(0, i_day - ROLLING_WINDOW)
        res = []
        for j in range(lo, i_day):
            if j in pred_set:
                pj = point_pred[j - i_feb1]
            else:                                     # 冷启动 persistence
                pj = true_mat[j - 1] if j > 0 else true_mat[j]
            res.append(true_mat[j] - pj)
        out[k] = point_pred[k] + np.quantile(np.asarray(res), q, axis=0)
        if is_pv:
            out[k, NIGHT] = 0.0
        out[k] = np.clip(out[k], 0, None)
    return out


def qcurves(tau):
    if tau not in _QCACHE:
        _QCACHE[tau] = (rolling_quantile_curve(load_point, Lmat, tau, False),
                        rolling_quantile_curve(pv_point, Pmat, 1.0 - tau, True))
    return _QCACHE[tau]


def build_residuals():
    """负载/光伏残差池(334天前序), 冷启动日前用persistence, 同p2b_quantile_sweep."""
    pred_idx = np.arange(i_feb1, i_feb1 + 334)
    pred_set = set(pred_idx.tolist())
    def _res(point, true_mat):
        res = np.zeros((334, T))
        for k, i_day in enumerate(pred_idx):
            lo = max(0, i_day - ROLLING_WINDOW)
            rows = []
            for j in range(lo, i_day):
                pj = point[k - (i_day - j)] if j in pred_set else (true_mat[j-1] if j > 0 else true_mat[j])
                rows.append(true_mat[j] - pj)
            res[k] = np.asarray(rows).mean(axis=0) * 0  # 占位, 下面逐日quantile在run内算
        return None
    # 直接仿 p2b: 残差矩阵按 (天,ti) 存, run_year 内逐 k 取窗
    resL = np.zeros((334, T)); resP = np.zeros((334, T))
    for k, i_day in enumerate(pred_idx):
        lo = max(0, i_day - ROLLING_WINDOW)
        rl, rp = [], []
        for j in range(lo, i_day):
            if j in pred_set:
                pl = load_point[k - (i_day - j)]; pp = pv_point[k - (i_day - j)]
            else:
                pl = Lmat[j-1] if j > 0 else Lmat[j]
                pp = Pmat[j-1] if j > 0 else Pmat[j]
            rl.append(Lmat[j] - pl); rp.append(Pmat[j] - pp)
        resL[k] = np.quantile(np.asarray(rl), 0.5, axis=0) if False else np.asarray(rl).mean(axis=0)*0
        resL[k] = 0; resP[k] = 0
    return None  # 残差池改为运行内逐日构建(见 run_year), 此函数仅保留接口说明


# ---------- 5. 单组全年 ----------
def run_year(price_kind, tag="", kind="none", tau=None, collect=False, settle_kind="actual4"):
    """price_kind: 'egd'|'fixed'|'perfect'  计划层电价来源.
       kind: 'none'|'pf'(负载光伏也完美)|'quantile'(残差分位曲线τ, 与p2b_quantile_sweep同口径)
       settle_kind: 'actual4'(附件4实际价结算)|'fixed'(附件1固定价结算, 用于复现问题2交叉验证)"""
    E0 = 6000.0
    initial_value = ETA * (settle_lam[0][0] if settle_kind == "actual4" else lam_fixed[0]) * E0
    pc = ec = ge = 0.0; pe = 0.0; ngap = 0
    smin = 1e18; smax = -1e18
    monthly = defaultdict(lambda: [0.0, 0.0, 0.0])
    e0_traj = np.empty(N)
    traj = [] if collect else None
    qL = qP = None
    if kind == "quantile":
        qL, qP = qcurves(tau)

    t0 = time.time()
    for k in range(N):
        drow = i_feb1 + k
        Lact = np.concatenate([Lmat[drow-1, 143:144], Lmat[drow, 0:143]])
        Pact = np.concatenate([Pmat[drow-1, 143:144], Pmat[drow, 0:143]])
        lam_s = settle_lam[k] if settle_kind == "actual4" else lam_fixed
        lam_plan = {"egd": plan_lam[k], "fixed": lam_fixed, "perfect": settle_lam[k]}[price_kind]

        if kind == "pf":
            Lf, Pf = Lact.copy(), Pact.copy()
        else:
            srcL, srcP = (qL, qP) if kind == "quantile" else (load_point, pv_point)
            if k == 0:
                fcl_prev = Lmat[drow-1, 142:143]; fcp_prev = Pmat[drow-1, 142:143]
            else:
                fcl_prev = srcL[k-1, 143:144]; fcp_prev = srcP[k-1, 143:144]
            Lf = np.maximum(np.concatenate([fcl_prev, srcL[k, 0:143]]), 0.0)
            Pf = np.maximum(np.concatenate([fcp_prev, srcP[k, 0:143]]), 0.0)

        a, b, gp, cp, dp = solve_day(Lf, Pf, E0, lam_plan)
        st = settle(a, b, gp, dp, Lact, Pact, E0, lam_s)
        m = dates[drow].month
        monthly[m][0] += st["pcost"]; monthly[m][1] += st["ecost"]; monthly[m][2] += st["gapE"]
        pc += st["pcost"]; ec += st["ecost"]; ge += st["gapE"]; ngap += st["ngap"]
        pe += (a.sum() + b.sum())*DT
        smin = min(smin, st["smin"]); smax = max(smax, st["smax"])
        E0 = st["Eend"]; e0_traj[k] = E0
        if collect:
            traj.append(dict(day=drow, a=a, b=b, g=gp, c=cp, d=dp, st=st))
        if (k+1) % 120 == 0:
            print(f"   [{tag}] {k+1}/{N} 天 累计计划{pc:,.0f} 紧急{ec:,.0f} 元 ({time.time()-t0:.0f}s)", flush=True)

    terminal_value = ETA * (settle_lam[N-1][-1] if settle_kind == "actual4" else lam_fixed[-1]) * E0
    gross = pc + ec
    net = gross - terminal_value
    net_change = gross - (terminal_value - initial_value)
    return dict(group=tag, plan=pc, emerg=ec, gross=gross, terminal_value=terminal_value,
                total=net, net_change=net_change, initial_value=initial_value, end_soc=E0,
                gapE=ge, ngap=ngap, purchE=pe, smin=smin, smax=smax, monthly=monthly,
                e0=e0_traj, traj=traj)


# ---------- 6. result4-2.xlsx (填表约定与 p2b_make_result2.py 完全一致) ----------
def fmt_clock(m):
    return "24:00" if m >= 1440 else f"{m//60}:{m%60:02d}"


BLOCKS = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]


def write_result42(res, path, settle_kind="actual4"):
    """模板填充; 列序 [I_2..I_144 | I_1](与p2b_make_result2同); 全天购电费按表内取整值计."""
    import openpyxl
    from datetime import time as dtime
    wb = openpyxl.load_workbook(BASE / "附件" / "附件5" / "result4-2.xlsx")

    ws = wb["计划购电量"]
    for k, rec in enumerate(res["traj"]):
        r = k + 2
        plan_e = (rec["a"] + rec["b"]) * DT                       # kWh, I_1..I_144
        vals = np.concatenate([plan_e[1:], plan_e[:1]])           # 模板列序 [I_2..I_144 | I_1]
        vals_r = np.round(vals, 2) + 0.0
        lam_s = settle_lam[rec["day"] - i_feb1] if settle_kind == "actual4" else lam_fixed
        lam_sheet = np.concatenate([lam_s[1:], lam_s[:1]])
        for j, v in enumerate(vals_r):
            ws.cell(row=r, column=2 + j, value=float(v))
        ws.cell(row=r, column=146, value=round(float(vals_r.sum()), 2))
        ws.cell(row=r, column=147, value=round(float(np.sum(lam_sheet * vals_r)), 2))

    ws = wb["充放电量"]
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for rec in res["traj"]:
        st = rec["st"]
        base = ws.max_row + 1
        for s in range(6):
            r = base + s
            if s == 0:
                ws.cell(row=r, column=1, value=pd.Timestamp(dates[rec["day"]]).to_pydatetime())
            ws.cell(row=r, column=2, value=BLOCKS[s])
            ws.cell(row=r, column=3, value=round(float(st["chin_e"][s*24:(s+1)*24].sum()), 2) + 0.0)
            ws.cell(row=r, column=4, value=round(float(st["dis_e"][s*24:(s+1)*24].sum()), 2) + 0.0)
            if s == 0:
                ws.cell(row=r, column=5, value=dtime(0, 0))
                ws.cell(row=r, column=6, value=round(float(st["E0"]), 2))
            elif s == 1:
                ws.cell(row=r, column=5, value="24:00")
                ws.cell(row=r, column=6, value=round(float(st["Eend"]), 2))

    ws = wb["紧急购电量"]
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for rec in res["traj"]:
        gap_e = rec["st"]["gap_e"]
        idx = np.where(gap_e > 1e-9)[0]
        if idx.size == 0:
            continue
        brk = np.where(np.diff(idx) > 1)[0]
        first = True
        for s0, se in zip(np.concatenate([[0], brk + 1]), np.concatenate([brk, [idx.size - 1]])):
            s, e = idx[s0], idx[se]
            r = ws.max_row + 1
            if first:
                ws.cell(row=r, column=1, value=pd.Timestamp(dates[rec["day"]]).to_pydatetime())
                first = False
            ws.cell(row=r, column=2, value=f"{fmt_clock(s*10)}-{fmt_clock((e+1)*10)}")
            ws.cell(row=r, column=3, value=round(float(gap_e[s:e+1].sum()), 2))
    wb.save(path)


# ---------- 7. 主流程 ----------
P2B_DIR = Path(r"D:\GitRepos\20260910-MMcompetition\C题\p2_part2")

if __name__ == "__main__":
    print(f"负载/光伏预测: 模型池组v5 EGD集成(固定); 计划电价: v6 EGD预测; 结算电价: 附件4实际")
    rows = []; monthly_store = {}

    def report(r):
        print(f"   净 {r['total']:>10,.0f} 元 = 支出 {r['gross']:>10,.0f} - 储能价值 {r['terminal_value']:>7,.0f}"
              f" | 计划 {r['plan']:>9,.0f} + 紧急 {r['emerg']:>9,.0f}"
              f" | 缺口 {r['gapE']:>8,.0f} kWh ({r['ngap']}时段) | SOC[{r['smin']:.0f},{r['smax']:.0f}]"
              f" | 期末SOC {r['end_soc']:.0f}")

    # 0) 交叉验证: 固定电价+τ=0.81 应精确复现问题2队友存档 (p2b_分位遍历_0.78_0.86.xlsx)
    print("\n=== 交叉验证: 固定电价 τ=0.81 复现问题2 ===")
    rv = run_year("fixed", tag="复现P2", kind="quantile", tau=0.81, settle_kind="fixed")
    print(f"   本管线: 计划 {rv['plan']:,.2f} 紧急 {rv['emerg']:,.2f} 净 {rv['total']:,.2f} 元"
          f" | 缺口 {rv['gapE']:,.1f} kWh ({rv['ngap']}段)")
    if not SMOKE:
        ref = pd.read_excel(P2B_DIR / "p2b_分位遍历_0.78_0.86.xlsx")
        rr = ref[np.isclose(ref["tau"].astype(float), 0.81)].iloc[0]
        d_plan = abs(rv["plan"] - rr["plan"]); d_em = abs(rv["emerg"] - rr["emerg"])
        d_tot = abs(rv["total"] - rr["total"])
        print(f"   队友存档: 计划 {rr['plan']:,.2f} 紧急 {rr['emerg']:,.2f} 净 {rr['total']:,.2f} 元")
        print(f"   差异: 计划{d_plan:.4f} 紧急{d_em:.4f} 净{d_tot:.4f} 元")
        assert d_tot < 1.0 and d_plan < 1.0 and d_em < 1.0, "交叉验证失败: 与队友τ=0.81存档不一致"
        print("   交叉验证通过 ✓ — 管线与p2b完全同构")

    print("\n=== 完美预见下界(电价+负载+光伏全完美) ===")
    r = run_year("perfect", tag="PF", kind="pf")
    rows.append({"组": "完美预见下界", **{k: r[k] for k in
                 ("plan", "emerg", "gross", "terminal_value", "total", "net_change",
                  "end_soc", "gapE", "ngap", "purchE", "smin", "smax")}})
    report(r); pf_total = r["total"]
    assert r["gapE"] < 1.0, "完美预见下缺口应≈0"

    for pk, gname in [("egd", "本文EGD电价预测"), ("fixed", "附件1形状(忽视波动)"), ("perfect", "完美预见电价")]:
        print(f"\n=== 组: {gname} (负载光伏=模型池组, 结算=附件4实际价) ===")
        r = run_year(pk, tag=gname, kind="none")
        rows.append({"组": gname, **{k: r[k] for k in
                     ("plan", "emerg", "gross", "terminal_value", "total", "net_change",
                      "end_soc", "gapE", "ngap", "purchE", "smin", "smax")}})
        monthly_store[gname] = r["monthly"]
        report(r)
        print(f"   误差代价(净-下界) {r['total']-pf_total:,.0f} 元")

    if not (SMOKE or TAU_ONLY):
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(OUT / "p4b_三组电价对比.xlsx", engine="openpyxl") as w:
            df.to_excel(w, sheet_name="组间对比", index=False)
            for g, mm in monthly_store.items():
                pd.DataFrame([{"月份": f"{m}月", "计划费": v[0], "紧急费": v[1], "缺口kWh": v[2]}
                              for m, v in sorted(mm.items())]).to_excel(w, sheet_name=f"月度_{g}"[:31], index=False)
        print(f"\n三组对比已保存 -> {OUT / 'p4b_三组电价对比.xlsx'}")

    # ---------- τ遍历(波动电价下重选决策分位, 仅G1口径): 粗扫0.73~0.93步0.04 + 最优点±0.04细化0.01 ----------
    if not SMOKE:
        if TAU_ONLY:
            taus = [round(t, 4) for t in np.arange(float(sys.argv[2]), float(sys.argv[3]) + 0.005, 0.01)]
            sweep = []
            print(f"\n=== τ遍历(EGD电价预测 + 残差分位: 负载P_τ/光伏P_(1-τ)) ===")
            for tau in taus:
                r = run_year("egd", tag=f"τ={tau}", kind="quantile", tau=tau)
                sweep.append({"tau": tau, **{k: r[k] for k in ("plan", "emerg", "gross", "total", "gapE", "ngap")}})
                print(f"   τ={tau:.2f} 净 {r['total']:>10,.0f} 元", flush=True)
            sw = pd.DataFrame(sweep).sort_values("total")
            print(sw.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
            sys.exit(0)
        sweep = []
        print(f"\n=== τ粗扫(EGD电价预测 + 残差分位: 负载P_τ/光伏P_(1-τ)) ===")
        for tau in [round(t, 4) for t in np.arange(0.73, 0.9301, 0.04)]:
            r = run_year("egd", tag=f"τ={tau}", kind="quantile", tau=tau)
            sweep.append({"tau": tau, **{k: r[k] for k in ("plan", "emerg", "gross", "total", "gapE", "ngap")}})
            print(f"   τ={tau:.2f} 净 {r['total']:>10,.0f} 元 (计划 {r['plan']:,.0f} 紧急 {r['emerg']:,.0f})", flush=True)
        best_coarse = min(sweep, key=lambda x: x["total"])["tau"]
        lo_, hi_ = best_coarse - 0.04, best_coarse + 0.04
        print(f"\n=== τ细化 [{lo_:.2f},{hi_:.2f}] 步长0.01 ===")
        done = {s["tau"] for s in sweep}
        for tau in [round(t, 4) for t in np.arange(lo_, hi_ + 0.005, 0.01)]:
            if tau in done:
                continue
            r = run_year("egd", tag=f"τ={tau}", kind="quantile", tau=tau)
            sweep.append({"tau": tau, **{k: r[k] for k in ("plan", "emerg", "gross", "total", "gapE", "ngap")}})
            print(f"   τ={tau:.2f} 净 {r['total']:>10,.0f} 元", flush=True)
        sw = pd.DataFrame(sweep).sort_values("total")
        print("\nτ遍历结果(按净成本升序):")
        print(sw.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
        sw.to_excel(OUT / "p4b_分位遍历_波动电价.xlsx", index=False)
        tau_star = float(sw.iloc[0]["tau"])
        print(f"\n波动电价下最优 τ* = {tau_star}")
        print("=== 用 τ* 生成 result4-2.xlsx ===")
        r = run_year("egd", tag=f"τ={tau_star}最终", kind="quantile", tau=tau_star, collect=True)
        write_result42(r, OUT / "result4-2.xlsx")
        report(r)
        print(f"result4-2.xlsx 已保存 -> {OUT / 'result4-2.xlsx'}")
