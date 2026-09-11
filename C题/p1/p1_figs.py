# -*- coding: utf-8 -*-
"""
P1 论文图（精而不多，共 3 张）：
  fig1_data.png      数据总览：电价 / 负载 / 光伏
  fig2_dispatch.png  调度结果：SOC 轨迹（上）+ 购电/充电/放电功率（下）
  fig3_shadow.png    对偶分析：节点边际价值 γ 与电价 λ（充/放窗口着色）
读取 _traj.npy（p1_solve.py 产物，行序 a,b,g,c,d,q,a+b,SOC[1:],γ）与附件1。
"""
from pathlib import Path
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTDIR = Path(__file__).resolve().parent
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(exist_ok=True)

# 数据
wb = openpyxl.load_workbook(OUTDIR.parent / "附件" / "附件1.xlsx", data_only=True)
rows = list(wb["Sheet1"].iter_rows(min_row=2, values_only=True))
raw_price = np.array([r[1] for r in rows], float)
raw_load = np.array([r[2] for r in rows], float)
raw_pv = np.array([r[3] for r in rows], float)
price = np.concatenate([raw_price[143:], raw_price[:143]])
load = np.concatenate([raw_load[143:], raw_load[:143]])
pv = np.concatenate([raw_pv[143:], raw_pv[:143]])

traj = np.load(OUTDIR / "_traj.npy")
a, b, g, c, d, q, ab, SOC, psi = traj
SOC = np.concatenate([[6000.0], SOC])              # 补节点0（0:00 初值 6000）
T = 144
h = np.arange(T) * (10 / 60.0)                     # 每段起点（小时）
h24 = np.append(h, 24.0)

# ---- 图1 数据总览 ----
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.fill_between(h24, np.append(pv, pv[-1]), step="post", alpha=0.35, color="tab:green", label="光伏预测功率")
ax.plot(h24, np.append(load, load[-1]), drawstyle="steps-post", color="tab:red", lw=1.4, label="小区负载")
ax.set_ylabel("功率 (kW)")
ax.set_ylim(0, 9000)
ax2 = ax.twinx()
ax2.plot(h24, np.append(price, price[-1]), drawstyle="steps-post", color="tab:blue", lw=1.2, label="电价")
ax2.set_ylabel("电价 (元/kWh)")
ax2.set_ylim(0, 1.6)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.set_xlabel("时刻 (h)")
l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8, ncol=3)
ax.set_title("附件1 数据总览：电价、小区负载与光伏预测（双谷双峰价格曲线）", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR / "fig1_data.png", dpi=300); plt.close(fig)

# ---- 图2 调度结果 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.3]})
ax1.plot(h24, SOC, drawstyle="steps-post", color="tab:purple", lw=1.5)
ax1.axhline(1200, color="gray", ls="--", lw=0.8); ax1.axhline(10800, color="gray", ls="--", lw=0.8)
ax1.axhline(6000, color="gray", ls=":", lw=0.8)
ax1.annotate("10800 上限", (23.9, 10800), ha="right", va="bottom", fontsize=7.5, color="gray")
ax1.annotate("1200 下限", (23.9, 1200), ha="right", va="bottom", fontsize=7.5, color="gray")
ax1.annotate("6000 (0:00/24:00)", (23.9, 6000), ha="right", va="bottom", fontsize=7.5, color="gray")
ax1.set_ylabel("储电量 (kWh)")
ax1.set_title("最优计划：储能荷电状态与分时段购电/充放电功率", fontsize=10)
w = 10 / 1440.0 * 24  # 10min in h
ax2.bar(h, ab, width=w, align="edge", color="tab:blue", alpha=0.8, label="购电功率")
ax2.plot(h24, np.append(b + c, (b + c)[-1]), drawstyle="steps-post", color="tab:green", lw=1.3, label="充电功率(并网侧)")
ax2.plot(h24, np.append(d, d[-1]), drawstyle="steps-post", color="tab:orange", lw=1.3, label="放电功率(并网侧)")
ax2.set_ylabel("功率 (kW)"); ax2.set_xlabel("时刻 (h)")
ax2.set_xlim(0, 24); ax2.set_xticks(range(0, 25, 2))
ax2.legend(loc="upper left", fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(FIGDIR / "fig2_dispatch.png", dpi=300); plt.close(fig)

# ---- 图3 对偶分析 ----
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ch_on = (b + c) > 1e-6; di_on = d > 1e-6
ax.fill_between(h24, 0, 1.7, where=np.append(ch_on, ch_on[-1]), step="post",
                alpha=0.15, color="tab:green", label="充电窗口")
ax.fill_between(h24, 0, 1.7, where=np.append(di_on, di_on[-1]), step="post",
                alpha=0.15, color="tab:orange", label="放电窗口")
ax.step(h24, np.append(price, price[-1]), where="post", color="tab:blue", lw=1.3, label="电价 λ")
ax.step(h24, np.append(psi, psi[-1]), where="post", color="tab:purple", lw=1.5, label="节点边际价值 γ")
ax.text(0.3, 0.06, "阈值判据: 充电 λ≈0.9γ，放电 σ≈γ/0.9，套利阈值 1/0.81≈1.235", fontsize=7.5, color="dimgray")
ax.set_ylabel("元/kWh"); ax.set_xlabel("时刻 (h)")
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.set_ylim(0, 1.7)
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.set_title("对偶分析：储能在各时刻的影子价格 γ 与电价 λ", fontsize=10)
fig.tight_layout(); fig.savefig(FIGDIR / "fig3_shadow.png", dpi=300); plt.close(fig)

for f in sorted(FIGDIR.glob("*.png")):
    print(f, f.stat().st_size, "bytes")
