# -*- coding: utf-8 -*-
"""
P3-M6 汇总结论与图:EVPI 分解、会话信息价值、会话启用族、风险画像、内生分位诊断
用法: python m6_report.py   → m6_out/m6_结论.xlsx + fig1..fig4.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
P3 = Path(__file__).resolve().parent
BASE = P3.parent
OUT = P3 / "m6_out"
I0 = 31
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
W = 1e4                                                     # 元 → 万元


def load():
    df = pd.read_excel(OUT / "m6_逐日.xlsx")
    rp_scen = np.load(OUT / "RP_scen.npy")                  # (D,K)
    m5 = pd.read_excel(P3 / "m5_out" / "m5_结果.xlsx")
    return df, rp_scen, m5


def report(df, rp_scen, m5):
    n = len(df)
    M = {k: df[k].mean() / W for k in ["RP", "WS", "EEV", "W0", "W1", "W2", "W3", "W4",
                                       "RP0", "RP1", "RP2", "RP3"]}
    tot = {k: df[k].sum() / W for k in ["RP", "WS", "EEV"]}
    # --- 信息价值(306 天窗口, 期望口径) ---
    evpi = df.RP.sum() / W - df.WS.sum() / W
    vss = df.EEV.sum() / W - df.RP.sum() / W
    v18 = df.W0.sum() / W - df.W1.sum() / W
    v12 = df.W1.sum() / W - df.W2.sum() / W
    v6 = df.W2.sum() / W - df.W3.sum() / W
    v0 = df.W3.sum() / W - df.W4.sum() / W
    # --- 样本外对照(实测路径; 与 M5 头号数字同口径) ---
    idx306 = np.arange(28, 334)
    rp_real = m5.set_index("e")["日费用"].reindex(range(334)).values[idx306].sum() / W
    eev_real = 0.0
    # EEV_real 列按日给出; m6_逐日 的 e 顺序即 days
    eev_real = df.set_index("e")["EEV_real"].reindex(range(28, 334)).values.sum() / W
    print(f"\n===== M6 汇总结论(306 天窗 3.1–12.31, 万元) =====")
    print(f"  期望口径   RP {tot['RP']:9.2f} | WS {tot['WS']:9.2f} | EEV {tot['EEV']:9.2f}")
    print(f"  EVPI = RP−WS = {evpi:7.2f}    VSS = EEV−RP = {vss:7.2f}")
    print(f"  实测量口径 RP {rp_real:9.2f} | EEV(实测量) {eev_real:9.2f}  → VSS_out = {eev_real-rp_real:7.2f}")
    print(f"  会话信息价值(完美预见): 18:00 {v18:6.2f} | 12:00 {v12:6.2f} | 6:00 {v6:6.2f} | 0:00计划 {v0:6.2f}  Σ={v18+v12+v6+v0:.2f}")

    fam = [df[f"RP{k}"].sum() / W for k in range(4)]
    print(f"  会话启用族(次调整 0/1/2/3): " + " ".join(f"{x:8.2f}" for x in fam))
    print(f"  边际价值: " + " ".join(f"{fam[i]-fam[i+1]:6.2f}" for i in range(3)))

    # --- 风险画像:在 RP 最优解上的逐情景净费用分布 ---
    sc = rp_scen.ravel() / W
    qs = np.percentile(sc, [50, 90, 95])
    cvar95 = sc[sc >= np.percentile(sc, 95)].mean()
    real = m5.set_index("e")["日费用"].reindex(range(28, 334)).values / W
    print(f"  风险画像(逐情景净费用, {len(sc)} 样本, 万元): P50 {qs[0]:7.2f} | P90 {qs[1]:7.2f} "
          f"| P95 {qs[2]:7.2f} | CVaR95 {cvar95:7.2f}")
    print(f"  风险画像(逐日实测费用, {len(real)} 样本): P50 {np.percentile(real,50):7.2f} | "
          f"P90 {np.percentile(real,90):7.2f} | P95 {np.percentile(real,95):7.2f} | "
          f"CVaR95 {real[real>=np.percentile(real,95)].mean():7.2f}")

    # --- 稳定性: 前半(3–7月)/后半(8–12月)切分 ---
    half = len(df) // 2
    stab = []
    for tag, sub in (("前半 3–7月", df.iloc[:half]), ("后半 8–12月", df.iloc[half:])):
        e = (sub.RP.sum() - sub.WS.sum()) / W
        v = (sub.EEV.sum() - sub.RP.sum()) / W
        s = [(sub[f"W{i}"].sum() - sub[f"W{i+1}"].sum()) / W for i in range(4)]
        stab.append({"区间": tag, "天数": len(sub), "EVPI": round(e, 2), "VSS": round(v, 2),
                     "18:00": round(s[0], 2), "12:00": round(s[1], 2),
                     "6:00": round(s[2], 2), "0:00计划": round(s[3], 2)})
    print("  稳定性(半年切分):")
    for r in stab:
        print(f"    {r['区间']} n={r['天数']}: EVPI {r['EVPI']:6.2f} | VSS {r['VSS']:6.2f} | "
              f"会话(18/12/6/0:00) {r['18:00']:5.2f}/{r['12:00']:5.2f}/{r['6:00']:5.2f}/{r['0:00计划']:5.2f}")

    with pd.ExcelWriter(OUT / "m6_结论.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"指标": "RP 期望(万元)", "值": round(tot["RP"], 2)},
            {"指标": "WS 期望", "值": round(tot["WS"], 2)},
            {"指标": "EEV 期望", "值": round(tot["EEV"], 2)},
            {"指标": "EVPI=RP−WS", "值": round(evpi, 2)},
            {"指标": "VSS=EEV−RP", "值": round(vss, 2)},
            {"指标": "RP 实测", "值": round(rp_real, 2)},
            {"指标": "EEV 实测", "值": round(eev_real, 2)},
            {"指标": "VSS 实测", "值": round(eev_real - rp_real, 2)},
            {"指标": "会话信息价值 18:00", "值": round(v18, 2)},
            {"指标": "会话信息价值 12:00", "值": round(v12, 2)},
            {"指标": "会话信息价值 6:00", "值": round(v6, 2)},
            {"指标": "会话信息价值 0:00计划", "值": round(v0, 2)},
        ]).to_excel(w, sheet_name="信息价值", index=False)
        pd.DataFrame([{"次调整": k, "费用(万元)": round(fam[k], 2),
                       "边际价值": None if k == 0 else round(fam[k - 1] - fam[k], 2)} for k in range(4)]
                     ).to_excel(w, sheet_name="会话启用族", index=False)
        pd.DataFrame([
            {"口径": "逐情景净费用", "P50": qs[0], "P90": qs[1], "P95": qs[2], "CVaR95": cvar95},
        ]).round(2).to_excel(w, sheet_name="风险画像", index=False)
        pd.DataFrame(stab).to_excel(w, sheet_name="稳定性", index=False)
    return dict(evpi=evpi, vss=vss, v18=v18, v12=v12, v6=v6, v0=v0, fam=fam,
                sc=sc, real=real, qs=qs, cvar95=cvar95)


def figs(R):
    # 图1: 价值瀑布
    wf = OUT / "m6_瀑布.xlsx"
    if wf.exists():
        t = pd.read_excel(wf)
        t = t.dropna(subset=["306天"])
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        vals = t["306天"].values
        LBL = {"本节+原始预报(无矫正融合)": "四阶段模型＋原始预报",
               "本节四阶段随机规划(矫正＋融合)": "四阶段随机规划，矫正＋融合"}
        labs = [LBL.get(x, x) for x in t["层级"].values]
        ax.bar(range(len(vals)), vals, color=["#e08a5b", "#4b8fd0", "#5aa469"])
        for i, v in enumerate(vals):
            ax.text(i, v + 8, f"{v:.1f}", ha="center", fontsize=10)
        for i in range(len(vals) - 1):
            d = vals[i] - vals[i + 1]
            dd = round(d + 1e-9, 1)
            ax.annotate("", xy=(i + 1, vals[i + 1]), xytext=(i, vals[i]),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
            ax.text(i + 0.5, (vals[i] + vals[i + 1]) / 2 + 20, f"−{dd:.1f}", color="#c0392b",
                    ha="center", fontsize=10)
        ax.axhline(1638.75, color="#7b241c", ls="--", lw=1.3)
        ax.text(len(vals) - 1, 1638.75 + 12, "问题二日前计划 1638.75，334 天口径，仅作参照",
                color="#7b241c", ha="right", fontsize=8.5)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8.5)
        ax.set_ylim(0, 1750)
        ax.set_ylabel("费用/万元"); ax.set_title("价值瀑布，306 天实测路径结算")
        fig.tight_layout(); fig.savefig(OUT / "fig1_瀑布.png", dpi=160); plt.close(fig)

    # 图2: 会话信息价值(完美预见)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = ["0:00计划", "6:00", "12:00", "18:00"]
    vs = [R["v0"], R["v6"], R["v12"], R["v18"]]
    ax.bar(ks, vs, color="#4b8fd0")
    for i, v in enumerate(vs):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("信息价值上限/万元")
    ax.set_title("各会话完美预报的信息价值上限，306 天")
    fig.tight_layout(); fig.savefig(OUT / "fig2_会话信息价值.png", dpi=160); plt.close(fig)

    # 图3: 会话启用族
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fam = R["fam"]
    ax.bar([f"{k} 次" for k in range(4)], fam, color="#5aa469")
    for i, v in enumerate(fam):
        ax.text(i, v + 3, f"{v:.1f}", ha="center", fontsize=10)
    ax2 = ax.twinx()
    marg = [fam[i] - fam[i + 1] for i in range(3)]
    ax2.plot([0.5, 1.5, 2.5], marg, "o-", color="#c0392b")
    for i, m in enumerate(marg):
        ax2.text(i + 0.5, m + 1, f"{m:.1f}", color="#c0392b", ha="center", fontsize=9)
    ax2.set_ylabel("边际价值/万元", color="#c0392b")
    ax.set_ylabel("费用/万元"); ax.set_title("会话启用族与边际信息价值，306 天")
    fig.tight_layout(); fig.savefig(OUT / "fig3_会话启用族.png", dpi=160); plt.close(fig)

    # 图4: 风险画像
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(R["sc"], bins=60, color="#9ec7e8", edgecolor="white", alpha=0.85, label="逐情景净费用")
    for q, lab, c in zip(R["qs"], ["P50", "P90", "P95"], ["#2c7fb8", "#e08a5b", "#c0392b"]):
        ax.axvline(q, color=c, ls="--", lw=1.4, label=f"{lab}={q:.1f}")
    ax.axvline(R["cvar95"], color="#7b241c", ls=":", lw=2.0, label=f"CVaR95={R['cvar95']:.1f}")
    ax.set_xlabel("单日净费用/万元"); ax.set_ylabel("情景数")
    ax.set_title("主模型费用分布与风险画像，306 天共 28 情景")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_风险画像.png", dpi=160); plt.close(fig)
    print(f"  图已保存 -> {OUT}/fig1..4.png")


if __name__ == "__main__":
    df, rp_scen, m5 = load()
    R = report(df, rp_scen, m5)
    figs(R)
