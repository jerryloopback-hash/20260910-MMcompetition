# -*- coding: utf-8 -*-
"""
P3-M6 价值瀑布(PLAN §5.3):问题二基线 → 日内调整 → 矫正＋融合 → 完美信息下界
================================================================================
全部在同一口径(实测路径结算, 万元, 334 天 2.1–12.31)上:
  级别0 问题二日前计划(母题已报)              1638.75
  级别1 本节模型＋原始附件3 预报(无矫正/融合)  —— 原始中心 + 原始残差路径库
  级别2 本节四阶段随机规划(矫正＋融合)         RP 实测 = 1468.63(M5 头号数字)
  级别3 完美信息下界                            WS_real:逐日按实测光伏/负载确定性最优

原始臂构造(与 m3 同构, 只把"校正后中心"换成"原始预报"):
  中心  C_raw[e,s,h] = F_raw[I0+e, s, h−H0[s]]         (原始附件3, 未去偏未融合)
  供体  ε0=A−C_raw[:,0], G_k=C_raw[:,k]−C_raw[:,k−1]  (原始坐标的混合残差路径库)
  十分钟下沉同 M3(EGD 形状, 守小时均值)。
用法: python P3_waterfall.py [days]   → m6_out/m6_瀑布.xlsx
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
P3 = Path(__file__).resolve().parent
OUT = P3 / "m6_out"
PRED = BASE / "p2_part1" / "预测结果"
H0 = np.array([0, 6, 12, 18]); NS, NL = 4, 24; I0 = 31


def build_raw_inputs():
    """原始附件3 坐标的中心/供体库/十分钟中心(与 m3 同构, 但不做 M1/M2/M3 任何校正)。"""
    F = np.load(P3 / "m1_out" / "F_raw.npy")            # (365,4,24) 原始预报(发布日×会话×步长)
    A = np.load(P3 / "m1_out" / "A_nat.npy")            # (365,24) 整点实际
    egdp = np.load(PRED / "pv_pred_ens_v5.npy")         # (334,144) EGD 光伏(供下沉形状)
    Cr = np.full((334, NS, NL), np.nan)
    for s in range(NS):
        for h in range(H0[s], NL):
            Cr[:, s, h] = F[I0:I0 + 334, s, h - H0[s]]
    eps0 = A[I0:] - Cr[:, 0, :]
    G = [Cr[:, s, H0[s]:] - Cr[:, s - 1, H0[s]:] for s in range(1, NS)]
    donor = np.concatenate([eps0] + G, axis=1).astype(np.float32)     # (334,60)
    Cs = np.full((NS, 334, 144), np.nan)
    for s in range(NS):
        for h in range(H0[s], NL):
            sl = np.arange(6 * h, 6 * h + 6)
            egsl = egdp[:334, sl]; egs = egsl.sum(axis=1); ok = egs > 1e-9
            shp = np.where(ok[:, None], egsl * 6.0 / np.where(ok, egs, 1.0)[:, None], 1.0)
            Cs[s][:, sl] = Cr[:, s, h][:, None] * shp
            assert np.allclose(Cs[s][:, sl].mean(axis=1)[ok], Cr[ok, s, h], rtol=1e-6), "原始下沉未守均值"
    return Cr.astype(np.float32), Cs.astype(np.float32), donor


def run_raw_arm():
    """把 M4/M5 的预报输入换成原始附件3, 逐日滚动解 RP, 按实测路径结算。"""
    import P3_tree as M4
    import P3_solve as M5
    Cr, Cs, donor = build_raw_inputs()
    M4.C_adj, M4.C_slot_adj, M4.donor_pv = Cr, Cs, donor
    M5.C_adj, M5.C_slot_adj = Cr, Cs
    rows = []
    soc = M5.E0
    for e in range(334):
        scen, tree, lp, x = M5.solve_day(e, soc, M5.LAM)
        rp = M5.P_act[I0 + e] * M5.DT; rl = M5.L_act[I0 + e] * M5.DT
        ex = M5.execute_realized(e, x, lp, tree, soc, rp, rl)
        rows.append({"e": e, "buy": ex["buy"], "emg": ex["emg"], "dev": ex["dev"],
                     "cost": ex["cost"], "gap": ex["gap"].sum(), "soc1": ex["E_end"]})
        soc = ex["E_end"]
        if (e + 1) % 60 == 0:
            print(f"  [原始臂] e={e+1}/334")
    return pd.DataFrame(rows)


def ws_real(soc_map):
    """完美信息下界(实测路径): 逐日按当日**实测**光伏/负载确定性最优。"""
    import P3_solve as M5
    Pv = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:145].astype(float).values
    Ld = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="小区负载").iloc[:, 1:145].astype(float).values
    out = np.zeros(334)
    for e in range(334):
        # PV_slot / L_slot 传**功率(kW)**, build_lp 内部再乘 Δt 得电量(与 build_scenarios 同口径)
        s1 = dict(K=1, PV_slot=Pv[I0 + e][None, :], L_slot=Ld[I0 + e][None, :])
        lp = M5.build_lp(s1, dict(nodes=[]), soc_map[e], M5.LAM, n_sessions=0)
        r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                    b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs")
        assert r.success, f"WS_real 日{e} 失败"
        out[e] = r.fun
    return out


def main():
    P3p = P3
    m5df = pd.read_excel(P3p / "m5_out" / "m5_结果.xlsx")
    soc_map = dict(zip(m5df["e"].values, m5df["SOC初"].values))
    print("===== M6 价值瀑布(实测路径结算, 万元) =====")
    print("计算完美信息下界 WS_real ...")
    ws = ws_real(soc_map)
    np.save(OUT / "WS_real.npy", ws)
    cache = OUT / "m6_原始臂逐日.xlsx"
    if cache.exists() and "--force" not in sys.argv:
        print("复用已缓存的原始臂结果(加 --force 重算)")
        raw = pd.read_excel(cache)
    else:
        print("计算原始预报臂 ...")
        raw = run_raw_arm()
    OUT.mkdir(exist_ok=True)
    raw.to_excel(cache, index=False)

    m5d = m5df.set_index("e")
    idx334 = np.arange(334); idx306 = np.arange(28, 334)
    rp_real = m5d["日费用"].reindex(range(334)).values          # RP 实测路径逐日费用
    rawv = raw["cost"].values
    def tot(v, ii): return float(np.asarray(v)[ii].sum() / 1e4)
    bars = {
        "问题二日前计划": dict(full=1638.75, win=None),
        "本节+原始预报(无矫正融合)": dict(full=tot(rawv, idx334), win=tot(rawv, idx306)),
        "本节四阶段随机规划(矫正＋融合)": dict(full=tot(rp_real, idx334), win=tot(rp_real, idx306)),
        "完美信息下界": dict(full=tot(ws, idx334), win=tot(ws, idx306)),
    }
    print("\n层级                                334天      306天(3.1-12.31)")
    rows = []
    for k, v in bars.items():
        w = f"{v['win']:9.2f}" if v["win"] is not None else "     —   "
        print(f"  {k:28s} {v['full']:9.2f}  {w}")
        rows.append({"层级": k, "334天": round(v["full"], 2),
                     "306天": None if v["win"] is None else round(v["win"], 2)})
    pd.DataFrame(rows).to_excel(OUT / "m6_瀑布.xlsx", index=False)
    print(f"\n已保存 -> {OUT/'m6_瀑布.xlsx'}")


if __name__ == "__main__":
    main()
