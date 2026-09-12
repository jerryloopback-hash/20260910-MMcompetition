# -*- coding: utf-8 -*-
"""
P2B 求解器:三组预测来源对比  v3
================================================
组内决议(2026-09-11):预测层交付**三组**日前预测, p2b 逐组跑同一管线, 以全年总费用横向比较:
  模型池组  — EGD 六模型时变集成(已交付: load_pred_ens.npy / pv_pred_ens.npy)
  LSTM组    — 待交付(建议 load_pred_lstm.npy / pv_pred_lstm.npy)
  XGBoost组 — 待交付(建议 load_pred_xgb.npy  / pv_pred_xgb.npy)
每组运行: 0:00 计划层(输入=该组预测, E0=前日执行末SOC, 不钉扎 E(24)=E(0))
        -> 目标扣除日末储能可利用价值 0.9*λ_144*E(24)
        -> 日内严格执行(仅物理截断, [1200,10800] 恒成立), 缺口 max(0, L-[购电+实际光伏+放电])
        -> 紧急购电 5λ 强制补足(硬约束), 购电按计划量计费。
另跑: 完美预见下界(预测=实际, hindsight 基准)。
保守度修正分支(逐时段残差分位, 方案B桥接版)保留但默认不启用——待组内确定在哪个组上叠加时再开。
时段映射(p2b_model.md §3.2): I_1 <- 前一日 ti=143; I_k(k>=2) <- 当日 ti=k-2; 2.1 的 I_1 持久性补丁。
用法: python p2b_solve.py [smoke]   输出: p2b_三组预测对比.xlsx
"""
import sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parent.parent          # C题/
ATT1 = BASE/"附件"/"附件1.xlsx"
ATT2 = BASE/"附件"/"附件2.xlsx"
PRED = BASE/"p2_part1"/"预测结果"
OUT  = Path(__file__).resolve().parent/"p2b_三组预测对比.xlsx"

DT = 1/6.0; ETA = 0.9; T = 144; EMIN, EMAX = 1200.0, 10800.0; PMAX = 5000.0
CYCLE_EPS = 1e-4                                       # 元/kWh, 仅用于打破同成本储能内循环退化解
WIN = 56; K0 = 7.0                                     # 残差窗/池深收缩(修正分支备用)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
NDAYS = 5 if SMOKE else 334

GROUPS = {                                             # 组名 -> (负载预测npy, 光伏预测npy)
    "模型池组":    ("load_pred_ens_v5.npy", "pv_pred_ens_v5.npy"),  # v5.1 EGD集成(5+5异构池)
    "XGBoost对照组": ("load_pred_xgb.npy",   "pv_pred_xgb.npy"),    # 队友 2026-09-11 交付
    "LSTM组":     ("lstm_load_pred.npy",   "lstm_pv_pred.npy"),     # 队友 _lstm_baseline.py 产物(由 _run_lstm_local.py 本地生成)
}

# ---------- 1. 数据 ----------
d1 = pd.read_excel(ATT1, sheet_name=0)
pr = pd.to_numeric(d1.iloc[:, 1], errors="coerce").values.astype(float)
assert len(pr) == T and not np.isnan(pr).any()
lam = np.concatenate([pr[143:], pr[:143]])             # I_1<-第144行, I_k<-第k-1行 (同P1)

df_l = pd.read_excel(ATT2, sheet_name="小区负载")
df_p = pd.read_excel(ATT2, sheet_name="光伏发电实际功率")
dates = pd.to_datetime(df_l.iloc[:, 0])
Lmat = df_l.iloc[:, 1:1+T].astype(float).values          # 365 x 144
Pmat = df_p.iloc[:, 1:1+T].astype(float).values
assert Lmat.shape == (365, T)
i_feb1 = int(np.where(dates.values == np.datetime64("2025-02-01"))[0][0])
assert i_feb1 == 31, f"附件2 起点异常: 2025-02-01 在第 {i_feb1} 行"
ActL = Lmat[i_feb1:i_feb1+334]                         # 334 x 144 实际值(结算用)
ActP = Pmat[i_feb1:i_feb1+334]

# 对齐自检: 预测文件列名 == 附件2 列名(语义: 附件2为 time 对象, 预测为 'HH:MM:SS' 字符串)
def _norm_col(c):
    return c if isinstance(c, str) else f"{c.hour:02d}:{c.minute:02d}:00"
pxl = pd.read_excel(PRED/"小区负载预测_2025_02-12_带日期.xlsx", nrows=1)
assert [_norm_col(c) for c in df_l.columns[1:]] == [str(c) for c in pxl.columns[1:]], "预测列名与附件2不一致"

N = NDAYS

# 组注册与自动发现: 文件落盘即纳入对比
avail, missing = {}, []
for g, (fl, fp) in GROUPS.items():
    pl, pp = PRED/fl, PRED/fp
    if not (pl.exists() and pp.exists()):          # 兼容队友直接放在 p2_part1 根目录的交付
        pl, pp = BASE/"p2_part1"/fl, BASE/"p2_part1"/fp
    if pl.exists() and pp.exists():
        aL = np.load(pl); aP = np.load(pp)
        assert aL.shape == (334, T) and np.isfinite(aL).all(), f"{g} 负载预测形状/数值异常"
        assert aP.shape == (334, T) and np.isfinite(aP).all(), f"{g} 光伏预测形状/数值异常"
        avail[g] = (aL, aP)
    else:
        missing.append(g)

# LightGBM单模型组(代"XGBoost组": v5.1 池已剔除 XGBoost, 见队友文档 §6.2.1; LightGBM 为池内唯一 boosting 单模型)
if (PRED/"load_pred_models_v5.npy").exists() and (PRED/"pv_pred_models_v5.npy").exists():
    mL = np.load(PRED/"load_pred_models_v5.npy"); mP = np.load(PRED/"pv_pred_models_v5.npy")
    assert mL.shape == (334, 5, T) and mP.shape == (334, 5, T), "逐模型预测形状异常(应为 334×5×144, 末位=LightGBM)"
    assert np.isfinite(mL).all() and np.isfinite(mP).all(), "逐模型预测含非有限值"
    avail["LightGBM单模型组"] = (np.ascontiguousarray(mL[:, 4, :]), np.ascontiguousarray(mP[:, 4, :]))

# ---------- 2. LP 结构(只建一次; 逐日仅换 b 向量) ----------
NV = T*6
idx = {k: np.arange(j, NV, 6) for j, k in enumerate("abgcdq")}


def soc_row(u):
    r = np.zeros(NV)
    r[idx["b"][:u+1]] += ETA*DT; r[idx["c"][:u+1]] += ETA*DT; r[idx["d"][:u+1]] -= DT/ETA
    return r


A_eq = np.zeros((2*T, NV))
for t in range(T):
    A_eq[2*t,   idx["a"][t]] = 1; A_eq[2*t,   idx["g"][t]] = 1; A_eq[2*t, idx["d"][t]] = 1   # 负载平衡
    A_eq[2*t+1, idx["g"][t]] = 1; A_eq[2*t+1, idx["c"][t]] = 1; A_eq[2*t+1, idx["q"][t]] = 1 # 光伏平衡
A_ub = np.zeros((4*T, NV))
for t in range(T):                                                                             # SOC 带(所有节点)
    A_ub[2*t] = soc_row(t); A_ub[2*t+1] = -soc_row(t)
for t in range(T):                                                                             # 功率上限(并网侧)
    A_ub[2*T+2*t,   idx["b"][t]] = 1; A_ub[2*T+2*t,   idx["c"][t]] = 1
    A_ub[2*T+2*t+1, idx["d"][t]] = 1
c_obj = np.zeros(NV); c_obj[idx["a"]] = lam*DT; c_obj[idx["b"]] = lam*DT
# 极小吞吐惩罚只用于排除同段充放的退化解，不计入报告成本。
c_obj[idx["b"]] += CYCLE_EPS*DT
c_obj[idx["c"]] += CYCLE_EPS*DT
c_obj[idx["d"]] += CYCLE_EPS*DT
# 日末储能价值: -η*λ_144*E_144。常数项 -η*λ_144*E0 不影响最优解，
# 对 E_144-E0=soc_row(T-1)@x 的线性系数直接并入目标。
c_obj -= ETA*lam[-1]*soc_row(T-1)


def solve_day(Lf, Pf, E0):
    """给定当日144时段预测与初始SOC, 解含日末储能价值的计划购电 LP。"""
    b_eq = np.empty(2*T)                      # 行序与 A_eq 一致: 偶=负载平衡, 奇=光伏平衡
    b_eq[0:2*T:2] = Lf; b_eq[1:2*T:2] = Pf
    b_ub = np.empty(4*T)
    b_ub[0:2*T:2] = EMAX - E0; b_ub[1:2*T:2] = E0 - EMIN; b_ub[2*T:] = PMAX
    r = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)]*NV, method="highs")
    assert r.success, r.message
    x = r.x
    return x[idx["a"]], x[idx["b"]], x[idx["g"]], x[idx["c"]], x[idx["d"]]

# ---------- 3. 严格执行结算(p2b_model.md §5.2) ----------
def settle(a, b, gp, dp, Lact, Pact, E0):
    E = E0; pcost = 0.0; ecost = 0.0; gapE = 0.0; ngap = 0
    smin = E; smax = E
    for t in range(T):
        gr  = gp[t] if gp[t] < Pact[t] else Pact[t]          # 实际光伏先满足计划直供
        da  = dp[t]
        cap = ETA*(E - EMIN)/DT                              # 1200 底线截断(恒不穿透)
        if da > cap: da = cap
        S = a[t] + gr + da
        gap = Lact[t] - S
        if gap > 0:                                          # 硬约束: 缺口 5λ 强制补足
            ecost += 5*lam[t]*gap*DT; gapE += gap*DT; ngap += 1
        cr = Pact[t] - gp[t]                                 # 实际光伏超出计划直供部分
        if cr < 0: cr = 0.0
        chin = (b[t] + cr)*DT                                # 充电=计划购电充电+实现的光伏盈余
        if chin > PMAX*DT: chin = PMAX*DT
        cap2 = (EMAX - E)/ETA
        if chin > cap2: chin = cap2
        E += ETA*chin - da*DT/ETA
        if E < smin: smin = E
        if E > smax: smax = E
        pcost += lam[t]*(a[t] + b[t])*DT                     # 购电按计划量计费
    return pcost, ecost, gapE, ngap, smin, smax, E

# ---------- 4. 保守度修正分支(方案B桥接版, 默认不启用; 组内确定叠加对象后开) ----------
def margins_quantile(tau, k, resL, resP):
    """第k天逐时段残差分位边际: 池=滚动WIN天(不足则扩张), 池深收缩 w=m/(m+7)。
       负载取 Q_τ(上调), 光伏取 Q_{1-τ}(下调)。k=0 无池 -> 零边际。"""
    j0 = max(0, k - WIN)
    m = k - j0
    if m == 0:
        return np.zeros(T), np.zeros(T)
    w = m / (m + K0)
    return w*np.quantile(resL[j0:k], tau, axis=0), w*np.quantile(resP[j0:k], 1.0-tau, axis=0)

# ---------- 5. 单组全年 ----------
def run_year(fL_g, fP_g, tag="", kind="none", params=None):
    E0 = 6000.0
    initial_value = ETA*lam[0]*E0
    pc = ec = ge = 0.0; pe = 0.0; ngap = 0
    smin = 1e18; smax = -1e18
    monthly = defaultdict(lambda: [0.0, 0.0, 0.0])           # 月 -> [计划费, 紧急费, 缺口kWh]
    e0_traj = np.empty(N)
    if kind == "quantile":                                   # 残差池按本组原始预测口径
        resL_g = ActL - fL_g; resP_g = ActP - fP_g
    t0 = time.time()
    for k in range(N):
        drow = i_feb1 + k
        Lact = np.concatenate([Lmat[drow-1, 143:144], Lmat[drow, 0:143]])
        Pact = np.concatenate([Pmat[drow-1, 143:144], Pmat[drow, 0:143]])
        if kind == "pf":                                     # 完美预见: 预测=实际
            Lf, Pf = Lact.copy(), Pact.copy()
        else:
            if k == 0:   # 2.1 的 I_1 无前日预测 -> 持久性补丁(1.31 23:50 实际值)
                fcl_prev = Lmat[drow-1, 142:143]; fcp_prev = Pmat[drow-1, 142:143]
            else:
                fcl_prev = fL_g[k-1, 143:144];    fcp_prev = fP_g[k-1, 143:144]
            Lf = np.concatenate([fcl_prev, fL_g[k, 0:143]])
            Pf = np.concatenate([fcp_prev, fP_g[k, 0:143]])
            Lf = np.maximum(Lf, 0.0); Pf = np.maximum(Pf, 0.0)   # 负功率无物理意义(逐模型PV可出现小幅负值)
            if kind == "quantile":
                mL, mP = margins_quantile(params, k, resL_g, resP_g)
                Lf = np.maximum(Lf + mL, 0.0); Pf = np.maximum(Pf + mP, 0.0)
        a, b, gp, cp, dp = solve_day(Lf, Pf, E0)
        p1, e1, g1, n1, sn, sx, Eend = settle(a, b, gp, dp, Lact, Pact, E0)
        m = dates[drow].month
        monthly[m][0] += p1; monthly[m][1] += e1; monthly[m][2] += g1
        pc += p1; ec += e1; ge += g1; ngap += n1
        pe += (a.sum() + b.sum())*DT
        smin = min(smin, sn); smax = max(smax, sx)
        E0 = Eend; e0_traj[k] = E0
        if (k+1) % 120 == 0:
            print(f"   [{tag}] {k+1}/{N} 天  累计 计划{pc:,.0f} 紧急{ec:,.0f} 元  ({time.time()-t0:.0f}s)")
    terminal_value = ETA*lam[-1]*E0
    gross = pc + ec
    net = gross - terminal_value
    net_change = gross - (terminal_value - initial_value)
    return dict(group=tag, plan=pc, emerg=ec, gross=gross,
                terminal_value=terminal_value, total=net, net_change=net_change,
                initial_value=initial_value, end_soc=E0, gapE=ge, ngap=ngap,
                purchE=pe, smin=smin, smax=smax, monthly=monthly, e0=e0_traj)

# ---------- 6. 主流程 ----------
if __name__ == "__main__":
    print(f"组注册: {list(GROUPS)}")
    if missing:
        print(f"等待交付: {missing} (文件落盘后自动纳入对比)")
    rows = []; monthly_store = {}

    print("\n=== 完美预见下界(预测=实际) ===")
    r = run_year(None, None, tag="PF", kind="pf")
    rows.append({"组": "完美预见下界", **{k: r[k] for k in
                 ("plan", "emerg", "gross", "terminal_value", "total", "net_change",
                  "end_soc", "gapE", "ngap", "purchE", "smin", "smax")}})
    print(f"   净成本 {r['total']:,.0f} 元 = 能源支出 {r['gross']:,.0f} - "
          f"期末储能价值 {r['terminal_value']:,.0f} | 期末SOC {r['end_soc']:.0f} kWh"
          f" | 缺口 {r['gapE']:,.3f} kWh")
    pf_total = r["total"]
    assert r["gapE"] < 1.0, "完美预见下缺口应≈0"

    for g, (aL, aP) in avail.items():
        print(f"\n=== 组: {g} (原始预测直喂) ===")
        r = run_year(aL, aP, tag=g, kind="none")
        rows.append({"组": g, **{k: r[k] for k in
                     ("plan", "emerg", "gross", "terminal_value", "total", "net_change",
                      "end_soc", "gapE", "ngap", "purchE", "smin", "smax")}})
        monthly_store[g] = r["monthly"]
        print(f"   净 {r['total']:>10,.0f} 元 = 支出 {r['gross']:>10,.0f} - 储能价值 {r['terminal_value']:>7,.0f}"
              f" | 期末SOC {r['end_soc']:.0f} kWh | 计划 {r['plan']:>9,.0f} + 紧急 {r['emerg']:>9,.0f}"
              f" | 缺口 {r['gapE']:>8,.0f} kWh ({r['ngap']}时段) | SOC[{r['smin']:.0f},{r['smax']:.0f}]"
              f" | 误差代价 {r['total']-pf_total:,.0f} 元")

    # ---------- 决策分位组合(队友交付分位数 v5 直读: 负载取 P_τ, 光伏取 P_{1-τ}) ----------
    qL = np.load(PRED/"load_quantiles_v5.npy"); qP = np.load(PRED/"pv_quantiles_v5.npy")
    assert qL.shape == (334, T, 4) and qP.shape == (334, T, 4), "分位数交付形状异常"
    assert np.isfinite(qL).all() and np.isfinite(qP).all()
    QCOMBOS = [("分位负载P0.50/光伏P0.50(离散对照)", 1, 1),
                ("分位负载P0.833/光伏P0.10(离散对照)", 2, 0),
                ("分位负载P0.90/光伏P0.10(离散对照)", 3, 0),
                ("分位负载P0.10/光伏P0.90(离散对照)", 0, 3)]
    print("\n=== 决策分位组合(队友分位数直读: 负载P_τ / 光伏P_{1-τ}) ===")
    for lab, il, ip in QCOMBOS:
        r = run_year(np.ascontiguousarray(qL[:, :, il]), np.ascontiguousarray(qP[:, :, ip]), tag=lab, kind="none")
        rows.append({"组": lab, **{k: r[k] for k in
                     ("plan", "emerg", "gross", "terminal_value", "total", "net_change",
                      "end_soc", "gapE", "ngap", "purchE", "smin", "smax")}})
        monthly_store[lab] = r["monthly"]
        print(f"   {lab} | 净 {r['total']:>10,.0f} = 支出 {r['gross']:>10,.0f} - 储能价值 {r['terminal_value']:>7,.0f}"
              f" | 期末SOC {r['end_soc']:.0f} | 计划 {r['plan']:>9,.0f} + 紧急 {r['emerg']:>9,.0f}"
              f" | 缺口 {r['gapE']:>8,.0f} kWh ({r['ngap']}时段)")

    df = pd.DataFrame(rows)
    groups_df = df[df["组"] != "完美预见下界"].sort_values("total")
    print("\n===== 全方案横向对比(按净成本升序) =====")
    print(groups_df[["组", "gross", "terminal_value", "total", "end_soc", "gapE", "ngap"]].to_string(index=False))
    if len(groups_df) >= 2:
        best, worst = groups_df.iloc[0], groups_df.iloc[-1]
        print(f"\n最优组: {best['组']} ({best['total']:,.0f} 元);"
              f"最差组: {worst['组']} ({worst['total']:,.0f} 元);组间差 {worst['total']-best['total']:,.0f} 元")
    print(f"完美预见下界净成本 {pf_total:,.0f} 元;各组误差代价 = 净成本 - 下界")

    if not SMOKE:
        with pd.ExcelWriter(OUT, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="组间对比", index=False)
            for g, mm in monthly_store.items():
                dmm = pd.DataFrame([{"月份": f"{m}月", "计划费": v[0], "紧急费": v[1], "缺口kWh": v[2]}
                                    for m, v in sorted(mm.items())])
                dmm.to_excel(w, sheet_name=f"月度_{g}"[:31], index=False)
        print(f"\n结果已保存 -> {OUT}")
    else:
        print("\n[smoke] 通过" + (",未写文件" if SMOKE else ""))
