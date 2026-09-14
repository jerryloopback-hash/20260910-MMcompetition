# -*- coding: utf-8 -*-
"""
P3 论文插图:四阶段信息结构示意图 + 会话价值双栏图(合并旧 fig2/fig3)。
用法: python P3_figs.py  → m6_out/fig_p3_structure.png, m6_out/fig_p3_session.png
数值一律取自 m6_out/m6_结论.xlsx,不手抄。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.stdout.reconfigure(encoding="utf-8")
P3 = Path(__file__).resolve().parent
OUT = P3 / "m6_out"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 数据(取自 m6_结论.xlsx) ----------------
wb = pd.read_excel(OUT / "m6_结论.xlsx", sheet_name=None)
iv = dict(zip(wb["信息价值"]["指标"], wb["信息价值"]["值"]))
fam = wb["会话启用族"]

SESSION_VAL = [iv["会话信息价值 0:00计划"], iv["会话信息价值 6:00"],
               iv["会话信息价值 12:00"], iv["会话信息价值 18:00"]]
FAM_COST = fam["费用(万元)"].tolist()
FAM_MARG = fam["边际价值"].tolist()[1:]

C_TXT = "#1a1a1a"


# ---------------- 图 A:四阶段信息结构 ----------------
def fig_structure():
    fig = plt.figure(figsize=(11.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.5)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.6, 24.9)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    sess = [0, 6, 12, 18]
    blocks = [(0, 6, "第 1 块(0:00–6:00)", "#d6e4f0"),
              (6, 12, "第 2 块(6:00–12:00)", "#e8f0d8"),
              (12, 18, "第 3 块(12:00–18:00)", "#fde9d0"),
              (18, 24, "第 4 块(18:00–24:00)", "#ecdff0")]
    for x0, x1, lab, col in blocks:
        ax.add_patch(Rectangle((x0, 0.9), x1 - x0, 1.5, facecolor=col,
                               edgecolor="#888888", lw=0.8))
        ax.text((x0 + x1) / 2, 1.62, lab, ha="center", va="center", fontsize=10.5)

    # 会话箭头与标注
    flab = ["0:00 预报 $F_0$", "6:00 预报 $F_1$", "12:00 预报 $F_2$", "18:00 预报 $F_3$"]
    for i, x in enumerate(sess):
        ax.annotate("", xy=(x, 2.4), xytext=(x, 3.35),
                    arrowprops=dict(arrowstyle="-|>", color="#2c5f8a", lw=1.8))
        ax.text(x, 3.55, flab[i], ha="center", va="bottom", fontsize=10.5,
                color="#1f4e79", fontweight="bold")
        if i > 0:
            ax.text(x + 0.25, 2.85, "＋实测 SOC", ha="left", va="center",
                    fontsize=9, color="#8a5a2c")

    # 块内成交规则
    ax.text(3.0, 0.45, "成交＝计划 $p_t$", ha="center", fontsize=9.5, color="#333333")
    for i, x0 in enumerate([6, 12, 18]):
        ax.text(x0 + 3, 0.45, f"成交＝第 {i+1} 次调整", ha="center", fontsize=9.5,
                color="#333333")
    ax.text(12, 4.35, "每个会话更新全天预报，从实测 SOC 出发重优化剩余全天",
            ha="center", fontsize=9.5, color="#555555")

    # 时间轴
    ax.annotate("", xy=(24.4, 0.0), xytext=(-0.2, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2))
    for x, lab in [(0, "0:00"), (6, "6:00"), (12, "12:00"), (18, "18:00"), (24, "24:00")]:
        ax.plot([x], [0.0], marker="|", ms=8, color="#444444")
        ax.text(x, -0.42, lab, ha="center", va="top", fontsize=9.5, color="#444444")

    # 情景树 1→3→9→27
    axt = fig.add_subplot(gs[1])
    axt.set_xlim(-0.8, 24.9)
    levels = [0, 6, 12, 18]
    counts = [1, 3, 9, 27]
    YMAX, NB = 10.0, 27
    pos = []
    for lv, (x, c) in enumerate(zip(levels, counts)):
        span = 1.0 + 8.2 * (lv / 3) ** 0.9
        ys = np.linspace(-span / 2, span / 2, c) + YMAX / 2
        pos.append(ys)
        axt.scatter([x] * c, ys, s=26 - 12 * (lv == 3), color="#2c5f8a",
                    zorder=3, edgecolor="white", lw=0.5)
    for lv in range(3):
        for i, y0 in enumerate(pos[lv]):
            kids = pos[lv + 1][3 * i:3 * i + 3] if lv < 2 else np.linspace(
                pos[lv + 1][0], pos[lv + 1][-1], 27)[3 * i::3]
            for y1 in (kids if len(kids) else []):
                axt.plot([levels[lv], levels[lv + 1]], [y0, y1],
                         color="#9bb7cf", lw=0.5, zorder=1)
    for i, x in enumerate(levels):
        axt.text(x, YMAX + 0.9, f"{counts[i]} 支", ha="center", fontsize=9.5,
                 color="#1f4e79")
    axt.text(24.2, YMAX / 2, "情景\n每支含\n完整日\n路径", ha="left", va="center",
             fontsize=8.5, color="#555555")
    axt.axis("off")
    axt.text(12, -1.35, "每会话按修正量 $R_s$ 的 PCA 主子方向聚 3 支，逐层嵌套成 3×3×3 情景树，"
                        "叶内情景共享同一套决策",
             ha="center", fontsize=9.5, color="#555555")

    fig.tight_layout()
    fig.savefig(OUT / "fig_p3_structure.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------- 图 B:会话价值双栏 ----------------
def fig_session():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.0))

    labs = ["0:00 计划", "6:00", "12:00", "18:00"]
    cols = ["#9bb7cf", "#9bb7cf", "#c0392b", "#9bb7cf"]
    b = axL.bar(range(4), SESSION_VAL, color=cols, width=0.62,
                edgecolor="#666666", lw=0.6)
    for i, v in enumerate(SESSION_VAL):
        axL.text(i, v + 0.9, f"{v:.1f}", ha="center", fontsize=10)
    axL.set_xticks(range(4))
    axL.set_xticklabels(labs, fontsize=10)
    axL.set_ylabel("信息价值上限 / 万元", fontsize=10.5)
    axL.set_ylim(0, 47)
    axL.set_title("若预报完美时的各会话信息价值上限", fontsize=11)
    axL.grid(axis="y", ls=":", alpha=0.45)

    x = np.arange(4)
    bars = axR.bar(x, FAM_COST, color="#6aa84f", width=0.6,
                   edgecolor="#4a7038", lw=0.6)
    for i, v in enumerate(FAM_COST):
        axR.text(i, v + 3, f"{v:.1f}", ha="center", fontsize=9.5)
    axR.set_xticks(x)
    axR.set_xticklabels(["0 次", "1 次", "2 次", "3 次"], fontsize=10)
    axR.set_ylabel("费用 / 万元", fontsize=10.5)
    axR.set_ylim(1120, 1230)
    axR.set_title("启用 $k$ 次调整的费用与边际价值", fontsize=11)

    axR2 = axR.twinx()
    axR2.plot(np.arange(1, 4), FAM_MARG, "o-", color="#c0392b", lw=1.8, ms=6)
    for i, v in enumerate(FAM_MARG):
        axR2.text(i + 1, v + 0.35, f"{v:.1f}", color="#c0392b", ha="center", fontsize=9.5)
    axR2.set_ylabel("边际价值 / 万元", fontsize=10.5, color="#c0392b")
    axR2.set_ylim(0, 15.5)
    axR2.tick_params(axis="y", colors="#c0392b")
    axR.grid(axis="y", ls=":", alpha=0.35)

    fig.suptitle("各会话预报的价值，306 天窗口", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_p3_session.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_structure()
    fig_session()
    print("done ->", OUT / "fig_p3_structure.png", OUT / "fig_p3_session.png")
