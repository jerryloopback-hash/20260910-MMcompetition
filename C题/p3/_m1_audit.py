# -*- coding: utf-8 -*-
"""
P3-M1 复核脚本(独立于 m1_correction.py, 只读不改)
=================================================
用途: 对 M1 交付物做三项交叉验证, 结论写入 m1_model.md §5/§7。
  审计1  会话顺序是否全年恒为 (0,6,12,18) —— 目标映射按位置取 H0[s], 顺序错位即全盘错位。
  审计2  窗口全扫 {7,14,21,28,42,56,90,expanding}, 复核"14–42 平台"与选窗规则。
  审计3  M2 前馈: 矫正后附件3 与 EGD 的误差相关与最优线性融合 MAE —— 校核 PLAN D8
         所称"误差相关 0.42"与"融合可压到 160 量级"。
  审计4  EGD 自身是否带系统偏差(v5 与旧版), 以及逐会话的互补性。
口径与 m1_correction.py 完全一致: 整点级 MAE, 白天 = 目标小时 5..19, 334 天, 终点读法。
用法: python _m1_audit.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
T, NS, NL = 144, 4, 24
H0 = np.array([0, 6, 12, 18])
I0, NE = 31, 334
I1 = I0 + NE - 1

# ---------- 数据(与 m1_correction.py 同源同读法) ----------
d2 = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率")
P = d2.iloc[:, 1:1 + T].astype(float).values
d3 = pd.read_excel(BASE / "附件" / "附件3.xlsx")
sess = d3.iloc[:, 1].astype(str).values
F = d3.iloc[:, 2:2 + NL].astype(float).values.reshape(365, NS, NL)

# ---------- 审计1: 会话顺序 ----------
_ss = np.asarray(sess, dtype=object).reshape(365, NS)
_expect = np.array([f"{h}:00" for h in H0])
print(f"[审计1] 全年每日会话顺序恒为 {list(_expect)}: {(_ss == _expect[None, :]).all()}")

# ---------- 目标映射与掩码 ----------
A = P.reshape(365, NL, 6).mean(axis=2)
di = np.arange(365)[:, None, None]
li = np.arange(NL)[None, None, :]
th = (H0[None, :, None] + li) % NL                          # (1,4,24)
td = di + (H0[None, :, None] + li >= NL).astype(int)        # (365,4,24)
valid = td <= 364
th0 = th[0]
target = np.where(valid, A[np.clip(td, 0, 364), th], np.nan)
Rraw = np.where(valid, F - target, np.nan)
sub = slice(I0, I1 + 1)
m_day0 = (th0 >= 5) & (th0 <= 19)
M = valid & (di >= I0) & (di <= I1) & m_day0
m3 = valid[sub] & m_day0[None]                              # 334 天白天, (334,4,24)


def trailing_mean(r, W):
    """r:(365,) 含 NaN → t[d]=mean(r[d-W:d]), 只用 d 之前。W<=0 视为 expanding。"""
    D = len(r)
    v = np.where(np.isfinite(r), r, 0.0)
    c = np.isfinite(r).astype(float)
    cv = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    lo = np.maximum(0, np.arange(D) - W if W > 0 else 0)
    hi = np.arange(D)
    return np.where((cc[hi] - cc[lo]) > 0, (cv[hi] - cv[lo]) / np.maximum(cc[hi] - cc[lo], 1.0), 0.0)


def bias_of(W):
    B = np.empty((365, NS, NL))
    for s in range(NS):
        for l in range(NL):
            B[:, s, l] = trailing_mean(Rraw[:, s, l], W)
    return B


# ---------- 审计2: 窗口全扫 ----------
print("\n[审计2] 窗口全扫(334 天白天 MAE, kW):")
for W in [7, 14, 21, 28, 42, 56, 90, 3650]:
    Fc = np.maximum(F - bias_of(W), 0.0)
    tag = "expanding" if W > 365 else str(W)
    print(f"   W={tag:>9}: {np.nanmean(np.abs(Fc - target)[M]):7.2f}")

# ---------- 审计3: M2 前馈(误差相关与最优线性融合) ----------
Fc28 = np.maximum(F - bias_of(28), 0.0)


def egd_hourly(fname):
    e = np.load(BASE / "p2_part1" / "预测结果" / fname).reshape(NE, NL, 6).mean(axis=2)
    Eg = np.full((365, NS, NL), np.nan)
    for d in range(I0, I1 + 1):
        Eg[d] = e[np.clip(td[d] - I0, 0, NE - 1), th0]        # 跨午夜步长对齐次日 EGD 预报
    return Eg


Eg = egd_hourly("pv_pred_ens_v5.npy")
a = (Fc28 - target)[sub][m3]                                 # 矫正附件3 误差
b = (Eg - target)[sub][m3]                                   # EGD 同点误差
print(f"\n[审计3] M2 前馈(整点白天): 矫正附件3 MAE={np.mean(np.abs(a)):.1f}, EGD MAE={np.mean(np.abs(b)):.1f}")
print(f"   误差相关 corr={np.corrcoef(a, b)[0, 1]:.3f}")
for tag, x in [("原始附件3", (F - target)[sub][m3]), ("矫正附件3", a)]:
    wg, vg = min(((np.mean(np.abs(t * x + (1 - t) * b)), t) for t in np.linspace(0, 1, 101)),
                 key=lambda z: z[0])
    print(f"   [{tag}×EGD] 最优融合 MAE={wg:.1f} (w_附件3={vg:.2f})")
print("\n[审计4] M2 前馈(逐会话): 矫正附件3 与 EGD 的误差相关")
for s in range(NS):
    ms = m3 & (np.arange(NS)[None, :, None] == s)
    x, y = (Fc28 - target)[sub][ms], (Eg - target)[sub][ms]
    wg, vg = min(((np.mean(np.abs(t * x + (1 - t) * y)), t) for t in np.linspace(0, 1, 101)),
                 key=lambda z: z[0])
    print(f"   会话{H0[s]:>2}:00 corr={np.corrcoef(x, y)[0, 1]:+.3f}  MAE 附件3={np.mean(np.abs(x)):6.1f} "
          f"EGD={np.mean(np.abs(y)):6.1f}  最优融合={wg:6.1f} (w3={vg:.2f})")
print("\n[审计5] EGD 自身系统偏差(整点白天):")
for nm in ["pv_pred_ens_v5.npy", "pv_pred_ens.npy"]:
    ee = (egd_hourly(nm) - target)[sub][m3]
    print(f"   [{nm}] MAE={np.nanmean(np.abs(ee)):.1f} 均值偏差={np.nanmean(ee):+.1f} "
          f"与矫正附件3 corr={np.corrcoef(a, ee)[0, 1]:.3f}")
