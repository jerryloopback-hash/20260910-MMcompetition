# -*- coding: utf-8 -*-
"""
P3-M3 复核脚本(独立于 P3_scenarios.py, 只读不改)
=================================================
审计1  终偏校正的"慢漂移"是否真实: δ=A−EGD 与混合残差 ε0 的逐月白天均值。
审计2  终偏校正增益是否只是选窗过拟合: 逐会话 4 个候选窗 vs 预注册固定 W=14;
       再用"前半年选窗/后半年评估"的切分复核。
审计3  相位定案复核(移窗 o 曲线)。
审计4  冷启动量化: 各日供体池宽; 1 月"附件3单源"残差与 2 月"混合"残差的尺度对比,
       —— 用于判断能否用 1 月补供体。
用法: python P3_audit_m3.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
M1 = Path(__file__).resolve().parent / "m1_out"
M2 = Path(__file__).resolve().parent / "m2_out"
M3 = Path(__file__).resolve().parent / "m3_out"
PRED = BASE / "p2_part1" / "预测结果"
NS, NL, H0 = 4, 24, np.array([0, 6, 12, 18])
I0, NE = 31, 334

A_nat = np.load(M1 / "A_nat.npy")                       # (365,24)
F_corr = np.load(M1 / "F_corr.npy")                     # (365,4,24)
egdp = np.load(PRED / "pv_pred_ens_v5.npy").reshape(NE, NL, 6).mean(axis=2)
P = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:145].astype(float).values
months = pd.to_datetime(pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 0]).dt.month.values

A_e = A_nat[I0:]                                        # (334,24)
mm = months[I0:]
day = np.arange(5, 20)

# ---------- 审计1: 漂移 ----------
d_egd = (A_e - egdp)[:, day]
eps0 = (A_e - F_corr[I0:, 0, :])[:, day]                # M1 去偏后附件3 的 0:00 残差 η0
print("[审计1] 白天(5..19)逐月均值 (kW)")
print(f"{'月':>4} {'δ=A−EGD':>10} {'η0=A−F0′':>12} {'混合残差ε0':>12}")
C_adj = np.load(M3 / "C_adj.npy"); C_ses = np.load(M2 / "C_ses.npy")
mix_raw = (A_e[:, None, :] - C_ses)[:, 0, day]
mix_adj = (A_e[:, None, :] - C_adj)[:, 0, day]
for m in range(2, 13):
    r = mm == m
    if r.sum() == 0:
        continue
    print(f"{m:>4} {d_egd[r].mean():>10.1f} {eps0[r].mean():>12.1f} "
          f"{mix_raw[r].mean():>12.1f}")
print(f"  全年 {d_egd.mean():>7.1f} {eps0.mean():>12.1f} {mix_raw.mean():>12.1f}"
      f"   (校正后混合 ε0 全年均值 {mix_adj.mean():+.1f})")
print(f"  月度极差: δ={d_egd[:, :].mean(0).ptp():.0f}?" if False else
      f"  月度极差: δ={np.ptp([d_egd[mm==m].mean() for m in range(2,13)]):.0f} kW, "
      f"η0={np.ptp([eps0[mm==m].mean() for m in range(2,13)]):.0f} kW")

# ---------- 审计2: 选窗是否过拟合 ----------
def tmean(X, W):
    E = X.shape[0]; out = np.zeros_like(X, dtype=float)
    for e in range(1, E):
        w = X[max(0, e - W):e]
        v = np.isfinite(w).any(axis=0)
        out[e, v] = np.nanmean(w[:, v], axis=0)
    return out


eps_raw = (A_e[:, None, :] - C_ses)
print("\n[审计2a] 各候选窗的会话 MAE(334天), 及预注册固定 W=14 的对照")
for s in range(NS):
    hb = slice(H0[s], NL)
    base = np.nanmean(np.abs(eps_raw[:, s, hb]))
    vals = {W: np.nanmean(np.abs((eps_raw[:, s, hb] - tmean(eps_raw[:, s, :], W)[:, hb])))
            for W in (7, 10, 14, 28)}
    b14 = vals[14]
    best = min(vals, key=vals.get)
    print(f"  会话{H0[s]:>2}:00 原始 {base:7.2f} | W=7 {vals[7]:7.2f} W=10 {vals[10]:7.2f} "
          f"W=14 {vals[14]:7.2f} W=28 {vals[28]:7.2f} | 择优 W={best}({vals[best]:.2f}) "
          f"vs 固定14({b14:.2f}) 差 {vals[best]-b14:+.2f}")

print("\n[审计2b] 切分复核: 前167天选窗 → 后167天评估(反之亦然)")
for tag, tr, te in [("前选后评", slice(0, 167), slice(167, NE)), ("后选前评", slice(167, NE), slice(0, 167))]:
    line = []
    for s in range(NS):
        hb = slice(H0[s], NL)
        B = tmean(eps_raw[:, s, :], 28)
        m_raw = np.nanmean(np.abs(eps_raw[te, s, hb]))
        sc = {W: np.nanmean(np.abs((eps_raw[tr, s, hb] - tmean(eps_raw[:, s, :], W)[tr, hb]))) for W in (7, 10, 14, 28)}
        Wb = min(sc, key=sc.get)
        m_sel = np.nanmean(np.abs((eps_raw[te, s, hb] - tmean(eps_raw[:, s, :], Wb)[te, hb])))
        m_14 = np.nanmean(np.abs((eps_raw[te, s, hb] - tmean(eps_raw[:, s, :], 14)[te, hb])))
        m_28 = np.nanmean(np.abs((eps_raw[te, s, hb] - B[te, hb])))
        line.append(f"{H0[s]:>2}:00 不校({m_raw:.1f}) 选W{Wb}({m_sel:.1f}) 定14({m_14:.1f}) 定28({m_28:.1f})")
    print(f"  {tag}: " + " | ".join(line))

# ---------- 审计3: 相位 ----------
print("\n[审计3] 相位移窗曲线")
for o in (-2, -1, 0, 1, 2):
    Ao = np.full((365, NL), np.nan)
    for h in range(1, NL - 1):
        lo, hi = 6 * h + o, 6 * h + 6 + o
        if lo >= 0 and hi <= 144:
            Ao[:, h] = P[:, lo:hi].mean(axis=1)
    e = np.concatenate([np.abs(F_corr[I0:, s, np.arange(max(H0[s], 5), 20) - H0[s]] - Ao[I0:, max(H0[s], 5):20]).ravel() for s in range(NS)])
    print(f"  o={o:+d}: {np.nanmean(e):7.1f} kW")

# ---------- 审计4: 冷启动 ----------
print("\n[审计4] 冷启动量化")
for e in [0, 1, 5, 14, 27, 28]:
    print(f"  评估日 e={e:>3}: 供体池宽 {e - max(0, e - 28)}")
jan = months < 2
res_jan = (A_nat[jan, :][:, day] - F_corr[jan, 0, :][:, day])       # 1月 附件3单源残差
res_feb = (A_nat[I0:][:, day] - C_adj[:, 0, day])                   # 2月起 混合残差(校正后)
print(f"  1月 附件3单源残差 白天 std={res_jan.std():6.1f} MAE={np.abs(res_jan).mean():6.1f} (n={len(res_jan)})")
print(f"  2月起 混合残差     白天 std={res_feb.std():6.1f} MAE={np.abs(res_feb).mean():6.1f} (n={len(res_feb)})")
print(f"  尺度比 std: {res_jan.std()/res_feb.std():.2f};  MAE 比: {np.abs(res_jan).mean()/np.abs(res_feb).mean():.2f}")

# ---------- 审计5: 追查文档所引 +143/−184/−175 的定义 ----------
print("\n[审计5] δ=A−EGD 月度均值的口径变体 (2月, 6月, 11月)")
egdp_old = np.load(PRED / "pv_pred_ens.npy").reshape(NE, NL, 6).mean(axis=2)
variants = {
    "白天5..19 整点(审计1口径)": (A_e - egdp)[:, 5:20],
    "白天6..20": (A_e - egdp)[:, 6:21],
    "峰段10..15": (A_e - egdp)[:, 10:16],
    "全天0..23": (A_e - egdp),
    "旧版EGD 白天5..19": (A_e - egdp_old)[:, 5:20],
}
# Frame I 口径: 各会话全步长(含跨午夜) 对齐 EGD 与 A
td_spill = (H0[None, :, None] + np.arange(NL)[None, None, :] >= NL).astype(int)
td_day = np.arange(NE)[:, None, None] + td_spill
th_all = (H0[None, :, None] + np.arange(NL)[None, None, :]) % NL
validI = td_day <= 364
tgtI = np.where(validI, A_nat[np.clip(td_day + I0, 0, 364), th_all], np.nan)
egI = np.where(validI, np.concatenate([egdp, egdp[-1:]], axis=0)[np.clip(td_day, 0, NE - 1), th_all], np.nan)
mI = validI & (th_all >= 5) & (th_all <= 19)
variants["FrameI 全步长白天(混跨午夜)"] = None
for name, X in variants.items():
    if X is None:
        vals = [np.nanmean((tgtI - egI)[mI & (np.arange(NE)[:, None, None] >= 0)]["x" == "x"]) if False else None]
        d = (tgtI - egI)[mI].reshape(NE, -1)
        vals = [np.nanmean(d[(mm == m)]) for m in (2, 6, 11)]
    else:
        vals = [X[mm == m].mean() for m in (2, 6, 11)]
    print(f"  {name:>26}: 2月 {vals[0]:+7.1f}  6月 {vals[1]:+7.1f}  11月 {vals[2]:+7.1f}")
print("  文档所引              : 2月  +143.0  6月  -184.0  11月  -175.0")
