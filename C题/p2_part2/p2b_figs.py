# -*- coding: utf-8 -*-
"""P2 论文图表: 组间对比 / 决策τ曲线 / 月度分解 -> outcome/figs/"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

XL = r"..\p2_part2\p2b_三组预测对比.xlsx"
FIGS = r"..\outcome\figs"
df = pd.read_excel(XL, sheet_name="组间对比")
pf = float(df[df["组"] == "完美预见下界"]["total"].iloc[0])
W = 1e4  # 万元

# ---- 图A: 三组模型来源对比(柱状) ----
groups = ["模型池组", "XGBoost对照组", "LightGBM单模型组", "LSTM组"]
vals = [float(df[df["组"] == g]["total"].iloc[0]) / W for g in groups]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(range(len(groups)), vals, width=0.55,
              color=["#2b6cb0", "#718096", "#a0aec0", "#e53e3e"])
for i, v in enumerate(vals):
    ax.text(i, v + 30, f"{v:,.0f}", ha="center", fontsize=10)
ax.axhline(pf / W, color="green", ls="--", lw=1.2)
ax.text(len(groups) - 0.45, pf / W + 30, f"完美预见下界 {pf/W:,.0f}", color="green", fontsize=9, ha="right")
ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
ax.set_ylabel("全年总费用 / 万元")
ax.set_title("三组预测来源的同管线全年总费用对比")
ax.set_ylim(0, max(vals) * 1.15)
fig.tight_layout(); fig.savefig(FIGS + r"\fig4_p2_groups.png", dpi=200); plt.close(fig)

# ---- 图B: 决策分位 τ 曲线 ----
qrows = df[df["组"].str.startswith("分位τ")].copy()
qrows["tau"] = qrows["组"].str.extract(r"τ=([0-9.]+)").astype(float)
qrows = qrows.sort_values("tau")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ens = float(df[df["组"] == "模型池组"]["total"].iloc[0]) / W
ax.plot(qrows["tau"], qrows["total"] / W, "o-", color="#2b6cb0", lw=1.8, label="分位数决策档(负载$P_\\tau$/光伏$P_{1-\\tau}$)")
for _, r0 in qrows.iterrows():
    ax.annotate(f"{r0['total']/W:,.0f}", (r0["tau"], r0["total"] / W), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
ax.axhline(ens / W, color="#dd6b20", ls="--", lw=1.2, label="模型池组点预测直喂(不修正)")
ax.axhline(pf / W, color="green", ls=":", lw=1.2, label="完美预见下界")
ax.set_xlabel("决策分位 τ(负载取 $P_\\tau$、光伏取 $P_{1-\\tau}$)")
ax.set_ylabel("全年总费用 / 万元")
ax.set_title("决策分位 τ 的全年总费用曲线(U 形)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIGS + r"\fig5_p2_tau.png", dpi=200); plt.close(fig)

# ---- 图C: 月度紧急费用(三组) ----
months = list(range(2, 13))
def monthly_emerg(sheet):
    m = pd.read_excel(XL, sheet_name=sheet)
    return np.array([m[m["月份"] == f"{i}月"]["紧急费"].sum() for i in months]) / W
e_ens = monthly_emerg("月度_模型池组")
e_xgb = monthly_emerg("月度_XGBoost对照组")
e_lstm = monthly_emerg("月度_LSTM组")
x = np.arange(len(months))
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar(x - 0.22, e_ens, width=0.22, label="模型池组", color="#2b6cb0")
ax.bar(x,        e_xgb, width=0.22, label="XGBoost对照组", color="#a0aec0")
ax.bar(x + 0.22, e_lstm, width=0.22, label="LSTM组", color="#e53e3e")
ax.set_xticks(x); ax.set_xticklabels([f"{m}月" for m in months])
ax.set_ylabel("月度紧急购电费 / 万元")
ax.set_title("三组预测来源的月度紧急购电费用对比")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIGS + r"\fig6_p2_monthly.png", dpi=200); plt.close(fig)

# ---- 论文用月度数字 ----
m833 = pd.read_excel(XL, sheet_name="月度_分位τ=0.833(报童)")
m833 = m833.set_index("月份")
mens = pd.read_excel(XL, sheet_name="月度_模型池组").set_index("月份")
print("月度总费用对比(万元): 模型池组 | 分位τ=0.833 | 节省")
for m in m833.index:
    t0 = mens.loc[m, "计划费"] + mens.loc[m, "紧急费"]
    t1 = m833.loc[m, "计划费"] + m833.loc[m, "紧急费"]
    print(f"  {m}: {t0/W:,.1f} | {t1/W:,.1f} | {(t0-t1)/W:,.1f}")
print("\n三图已生成 ->", FIGS)
