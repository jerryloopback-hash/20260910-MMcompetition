# -*- coding: utf-8 -*-
"""
决策分位 tau 的全年费用遍历(2026-09-12)
================================================
动机: 论文 6.2.2 的 0.8333 来自报童临界分位 Cu/(Cu+Co)=5/6, 是公式结论而非遍历结论。
      本节以 0.833 为中心、2% 为梯度遍历决策分位 tau, 用真实结算费用判定最优点。
口径(与 _quantile_v5.py + p2b_solve.py 完全一致):
  修正形状 = 点预测(EGD集成v5) + 过去30天同时点滚动残差的经验分位(按 ti 分别算)
  决策组合 = 负载取 P_tau, 光伏取 P_{1-tau}
  执行结算 = p2b_solve 的 solve_day(LP) + settle(严格执行, 缺口5lambda)
只跑模型池组(论文 6.2 分位数讨论所用组)。另含一条自检: 重建的 tau=0.833 负载曲线
与队友交付的 load_quantiles_v5[:,:,2] 应逐点一致。
用法: python p2b_quantile_sweep.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2b_solve import (solve_day, settle, Lmat, Pmat, dates, i_feb1, T, DT,
                       PRED, lam as S_lam)  # noqa: E402

# ---- 遍历点: 0.833 为中心, 2% 梯度, 覆盖 0.73~0.93 ----
# 命令行可覆盖: python p2b_quantile_sweep.py 0.79 0.01 0.85   (起点 步长 终点)
if len(sys.argv) == 4:
    _a, _s, _b = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    TAUS = [round(t, 4) for t in np.arange(_a, _b + _s / 2, _s)]
else:
    TAUS = [round(0.833 + d, 4) for d in np.arange(-0.10, 0.101, 0.02)]
ROLLING_WINDOW = 30
N = 334

LOADE = PRED / "load_pred_ens_v5.npy"
PVE = PRED / "pv_pred_ens_v5.npy"
load_point = np.load(LOADE)                       # (334,144)
pv_point = np.load(PVE)
assert load_point.shape == (334, T) and pv_point.shape == (334, T)

# 预测日索引: 2025-02-01..12-31 -> 附件2 中的行号(断言与 p2b_solve 一致)
pred_idx = np.arange(i_feb1, i_feb1 + 334)
pred_set = set(pred_idx.tolist())
hrs = (np.arange(T) + 1) * 10.0 / 60.0
night = (hrs <= 6.0) | (hrs >= 20.0)


def rolling_quantile_at(point_pred, true_mat, q, is_pv):
    """点预测 + 过去30天同时点残差的经验 q 分位; 与 _quantile_v5.py 同法。"""
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
            out[k, night] = 0.0
        out[k] = np.clip(out[k], 0, None)
    return out


def run_tau(qL, qP, tag):
    """与 p2b_solve.run_year(kind='none') 同构的单组全年结算。"""
    E0 = 6000.0
    initial_value = 0.9 * S_lam[0] * E0
    pc = ec = ge = 0.0
    ngap = 0
    smin, smax = 1e18, -1e18
    for k in range(N):
        drow = i_feb1 + k
        Lact = np.concatenate([Lmat[drow - 1, 143:144], Lmat[drow, 0:143]])
        Pact = np.concatenate([Pmat[drow - 1, 143:144], Pmat[drow, 0:143]])
        if k == 0:
            fcl_prev = Lmat[drow - 1, 142:143]
            fcp_prev = Pmat[drow - 1, 142:143]
        else:
            fcl_prev = qL[k - 1, 143:144]
            fcp_prev = qP[k - 1, 143:144]
        Lf = np.maximum(np.concatenate([fcl_prev, qL[k, 0:143]]), 0.0)
        Pf = np.maximum(np.concatenate([fcp_prev, qP[k, 0:143]]), 0.0)
        a, b, gp, cp, dp = solve_day(Lf, Pf, E0)
        p1, e1, g1, n1, sn, sx, Eend = settle(a, b, gp, dp, Lact, Pact, E0)
        pc += p1; ec += e1; ge += g1; ngap += n1
        smin = min(smin, sn); smax = max(smax, sx); E0 = Eend
    terminal_value = 0.9 * S_lam[-1] * E0
    gross = pc + ec
    return dict(tau=tag, plan=pc, emerg=ec, gross=gross,
                terminal_value=terminal_value, total=gross - terminal_value,
                net_change=gross - terminal_value + initial_value,
                end_soc=E0, gapE=ge, ngap=ngap, smin=smin, smax=smax)


if __name__ == "__main__":
    t0 = time.time()
    # ---- 自检: 重建 vs 队友交付 ----
    qL833 = rolling_quantile_at(load_point, Lmat, 0.833, False)
    refL = np.load(PRED / "load_quantiles_v5.npy")[:, :, 2]
    d = np.abs(qL833 - refL).max()
    print(f"[自检] 重建 tau=0.833 负载分位 vs 交付 load_quantiles_v5[:,:,2]: 逐点最大差 {d:.3e}")

    rows = []
    for tau in TAUS:
        tt = time.time()
        qL = rolling_quantile_at(load_point, Lmat, tau, False)
        qP = rolling_quantile_at(pv_point, Pmat, 1.0 - tau, True)
        r = run_tau(qL, qP, tau)
        rows.append(r)
        print(f"  tau={tau:.4f} | 计划 {r['plan']:>13,.0f}  紧急 {r['emerg']:>12,.0f}  "
              f"支出 {r['gross']:>13,.0f}  储能价值 {r['terminal_value']:>7,.0f}  "
              f"净成本 {r['total']:>13,.0f} 元 | 期末SOC {r['end_soc']:.0f} "
              f"| 缺口 {r['gapE']:>8,.0f} kWh ({r['ngap']}段) | {time.time()-tt:.0f}s")

    df = pd.DataFrame(rows)
    df["total_wan"] = df["total"] / 1e4
    df = df.sort_values("total").reset_index(drop=True)
    print("\n===== 分位遍历(按净成本升序) =====")
    print(df[["tau", "plan", "emerg", "gross", "terminal_value", "total", "end_soc", "gapE", "ngap"]].to_string(index=False))
    best = df.iloc[0]
    print(f"\n费用最优 tau* = {best['tau']:.4f} (净成本 {best['total']:,.0f} 元);"
          f" 参考分位 0.833 附近净成本 "
          f"{df.loc[df['tau'].sub(0.833).abs().idxmin(),'total']:,.0f} 元")

    out = HERE / f"p2b_分位遍历_{TAUS[0]}_{TAUS[-1]}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="tau遍历", index=False)
    print(f"结果已保存 -> {out}  ({time.time()-t0:.0f}s)")
