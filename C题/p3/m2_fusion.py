# -*- coding: utf-8 -*-
"""
P3-M2 融合层:矫正后附件3 × EGD 的会话相关线性加权
================================================
依据 p3_plan.md §3.2 / D8 / 里程碑 M2:
  光伏中心 C_s = w_s·F'_s + (1−w_s)·EGD, 权重按会话由滚动样本外 MAE 定权;
  负载中心不融合, 直接用 EGD 负载预测(附件3 无负载预报)。
审计修订(2026-09-12, m1_model.md §7): 两源误差相关 ~0.08, "160 量级"仅 6:00 会话成立
  → 权重逐会话, 验收=逐会话融合优于任一单一来源。
帧定义(三层, 勿混):
  Frame I(发布帧, 审计口径)   = 按发布(日,会话)计, 目标=其覆盖的白天小时(含跨午夜次日目标)。
                                仅用于与审计数字(oracle 208.1/166.6/229.5/223.8, 池化 214.8)连续性校验。
  Frame W(权重帧)            = s=0..2 取绑定段∩白天([max(H_s,5),19]); s=3 绑定段无白天, 取完整绑定段。
                                滚动权重在此帧估计。
  Frame D(部署帧)            = 绑定段全天 [H_s,23], 即随机规划节点实际消费的中心范围; 主验收帧。
权重规则: 对评估日 d, 会话 s, 在权重帧的滚动窗 [d−28,d−1] 上取两源 MSE, 逆方差定权
  w = MSE_EGD/(MSE_att3+MSE_EGD) (独立性下 MSE 最优; 逆 MAE² 的稳定解析形式);
  窗内 EGD 有效天数 m<28 时按 m/(m+K0), K0=3 向 0.5 收缩(EGD 自 2.1 才有)。
  弃用两个朴素方案(实测): ①逆 MAE 定权在技能差异大时系统性欠权优势源(会话 2 权重 0.62 vs
  oracle 0.83, 融合劣于单源); ②窗内 MAE 格点 argmin 过拟合窗口噪声(权重 0.12/0.82/0.94 对
  oracle 0.34/0.59/0.83)。逆 MSE 为两者的稳定折中。
口径: 整点级 MAE, 白天=目标小时 5..19, 整点实际=终点读法(ti 6h..6h+5), 与 M1 一致。
中心下沉: 十分钟中心 = 融合整点水平 × EGD 小时内形状(EGD 整点为零时平坦均分), 保能量;
  M3 情景误差下沉用 PLAN §3.4 历史形态曲线, 两者相加组合。
用法: python m2_fusion.py [smoke]   产出: p3/m2_out/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent          # C题/
M1 = Path(__file__).resolve().parent / "m1_out"
PRED = BASE / "p2_part1" / "预测结果"
OUT = Path(__file__).resolve().parent / "m2_out"

NS, NL = 4, 24
H0 = np.array([0, 6, 12, 18])
I0 = 31                                                # 2025-02-01 在附件2 的行号
W2 = 28                                                # 权重滚动窗(预注册, 承 M1 平台中心)
K0 = 3.0                                               # EGD 冷启动收缩
GRID_O = np.arange(0, 1.0001, 0.01)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
NEVAL = 40 if SMOKE else 334

# ---------- 1. 输入 ----------
F_corr = np.load(M1 / "F_corr.npy")                    # (365,4,24) 发布日×会话×步长
A_nat = np.load(M1 / "A_nat.npy")                      # (365,24) 整点实际(终点读法)
assert F_corr.shape == (365, NS, NL) and A_nat.shape == (365, NL)
egdp = np.load(PRED / "pv_pred_ens_v5.npy")            # (334,144) EGD 光伏
assert egdp.shape == (334, 144) and np.isfinite(egdp).all()
A_egd = egdp.reshape(334, NL, 6).mean(axis=2)          # (334,24) 行e ↔ 实际日 31+e
assert np.all(A_egd[:, list(range(0, 6)) + list(range(20, 24))] == 0), "EGD 夜间零值口径变化"

di1 = np.arange(365)
# ---------- 2. Frame I(发布帧, 审计连续性) ----------
tabI = []
daI, deI = [], []
for s in range(NS):
    th_s = (H0[s] + np.arange(NL)) % NL                # (24,) 各步长的目标小时
    spill = ((H0[s] + np.arange(NL)) >= NL).astype(int)
    tdM = di1[:, None] + spill[None, :]                # (365,24) 各(发布日,步长)的目标日
    AtgtI = np.where(tdM <= 364, A_nat[np.clip(tdM, 0, 364), th_s], np.nan)
    day = (th_s[None, :] >= 5) & (th_s[None, :] <= 19) & (tdM <= 364) & (di1 >= I0)[:, None]
    e_att = np.abs(F_corr[:, s, :] - AtgtI)[day]
    egT = np.where(tdM <= 364, A_egd[np.clip(tdM - I0, 0, 333), th_s], np.nan)  # EGD 按目标日配对
    e_egd = np.abs(egT - AtgtI)[day]
    da, de = (F_corr[:, s, :] - AtgtI)[day], (egT - AtgtI)[day]
    errs = [np.abs(w * da + (1 - w) * de).mean() for w in GRID_O]
    i = int(np.argmin(errs))
    tabI.append({"会话": f"{H0[s]:>2}:00", "矫正附件3": round(e_att.mean(), 1),
                 "EGD": round(e_egd.mean(), 1), "融合(oracle)": round(min(errs), 1),
                 "oracle权重": round(GRID_O[i], 2)})
    # 与 M1 分会话表连续性(同对同值)
    m1_ref = [332.9, 212.6, 403.7, 452.4][s]
    assert abs(e_att.mean() - m1_ref) < 0.5, f"Frame I att3 与 M1 不符: {e_att.mean()} vs {m1_ref}"
    daI.append(da.ravel()); deI.append(de.ravel())
daI, deI = np.concatenate(daI), np.concatenate(deI)
errsI = [np.abs(w * daI + (1 - w) * deI).mean() for w in GRID_O]
poI = float(min(errsI))
wI = GRID_O[int(np.argmin(errsI))]
print("===== Frame I(发布帧, 审计连续性; oracle 为 in-sample 上界) =====")
print(pd.DataFrame(tabI).to_string(index=False))
print(f"池化oracle {poI:.1f} (w={wI:.2f}) <- 审计 214.8 (w=0.31)")
# 审计连续性: 四会话 oracle 与池化 oracle 均须逐位复现 m1_model.md §7 的审计值。
_ORACLE_REF = [208.1, 166.6, 229.5, 223.8]
for r, ref in zip(tabI, _ORACLE_REF):
    assert abs(r["融合(oracle)"] - ref) < 0.15, f"Frame I oracle 与审计不符: {r['融合(oracle)']} vs {ref}"
assert abs(poI - 214.8) < 0.15, f"Frame I 池化 oracle 与审计不符: {poI:.2f} vs 214.8"
assert abs(wI - 0.31) < 1e-9, f"Frame I 池化 oracle 权重与审计不符: {wI} vs 0.31"

# ---------- 3. 权重帧与滚动权重 ----------
wlo = [max(int(H0[s]), 5) if s < 3 else int(H0[s]) for s in range(NS)]
whi = [19 if s < 3 else 23 for s in range(NS)]
EA, EE = [], []                                        # 各会话权重帧逐日误差 (365,nh)
for s in range(NS):
    hz = np.arange(wlo[s], whi[s] + 1)
    EA.append(np.abs(F_corr[:, s, hz - H0[s]] - A_nat[:, hz]))
    ee_ = np.full((365, len(hz)), np.nan)
    ee_[I0:] = np.abs(A_egd[:, hz] - A_nat[I0:, hz])
    EE.append(ee_)

w_att = np.full((NEVAL, NS), np.nan)                   # 附件3 权重史
for e in range(NEVAL):
    d = I0 + e
    d0 = max(I0, d - W2)
    m = d - d0                                         # 窗内含 EGD 的天数
    for s in range(NS):
        if m == 0:
            w = 0.5
        else:
            ma2 = float((EA[s][d0:d] ** 2).mean())         # 窗内 MSE
            me2 = float((EE[s][d0:d] ** 2).mean())
            wr = me2 / (ma2 + me2)                         # 逆方差权重(独立性下 MSE 最优)
            w = 0.5 + (m / (m + K0)) * (wr - 0.5)          # EGD 冷启动收缩
        w_att[e, s] = np.clip(w, 0.02, 0.98)
assert np.isfinite(w_att).all()

# ---------- 4. 融合中心(绑定段) ----------
C = np.full((NEVAL, NS, NL), np.nan)
for s in range(NS):
    hb = np.arange(H0[s], NL)
    C[:, s, hb] = (w_att[:, s, None] * F_corr[I0:I0 + NEVAL, s, :NL - H0[s]]
                   + (1 - w_att[:, s, None]) * A_egd[:NEVAL, hb])
assert np.nanmin(C) >= 0, "融合中心出现负值"

# ---------- 5. 评估(Frame W 主验收 + Frame D 部署) ----------
def frame_errs(s, h_lo, h_hi):
    hz = np.arange(h_lo, h_hi + 1)
    An = A_nat[I0:I0 + NEVAL, hz]
    fc, eg, fu = F_corr[I0:I0 + NEVAL, s, hz - H0[s]], A_egd[:NEVAL, hz], C[:, s, hz]
    return np.abs(fc - An), np.abs(eg - An), np.abs(fu - An), fc, eg, An


def oracle_w(fc, eg, An):
    da, de = fc - An, eg - An
    errs = [np.abs(w * da + (1 - w) * de).mean() for w in GRID_O]
    i = int(np.argmin(errs))
    return float(GRID_O[i]), float(errs[i])


tabW, tabD = [], []
pf_all, pa_all, pe_all = [], [], []
for s in range(NS):
    # 报告帧必须与权重估计帧严格同一: 用 wlo/whi, 勿硬编码 19
    # (v1 曾对 s=3 误取 18..19, 与其权重帧 18..23 不一致, 已修)
    e_a, e_e, e_f, fc, eg, An = frame_errs(s, wlo[s], whi[s])
    wo, mo = oracle_w(fc, eg, An)
    tabW.append({"会话": f"{H0[s]:>2}:00", "目标小时": f"{wlo[s]}..{whi[s]}",
                 "矫正附件3": round(e_a.mean(), 1), "EGD": round(e_e.mean(), 1),
                 "融合(滚动w)": round(e_f.mean(), 1), "融合(oracle)": round(mo, 1),
                 "oracle权重": round(wo, 2), "滚动w均值": round(float(w_att[:, s].mean()), 2),
                 "较优单源提升%": round((1 - e_f.mean() / min(e_a.mean(), e_e.mean())) * 100, 1)})
    if s < 3:
        assert e_f.mean() < min(e_a.mean(), e_e.mean()), f"会话{s} 融合未优于单一来源"
    pf_all.append(e_f.ravel()); pa_all.append(e_a.ravel()); pe_all.append(e_e.ravel())

    e_aD, e_eD, e_fD, *_ = frame_errs(s, int(H0[s]), 23)
    win = (e_fD < np.minimum(e_aD, e_eD)).mean()
    tabD.append({"会话": f"{H0[s]:>2}:00", "目标小时": f"{H0[s]}..23",
                 "矫正附件3": round(e_aD.mean(), 1), "EGD": round(e_eD.mean(), 1),
                 "融合(滚动w)": round(e_fD.mean(), 1), "双源皆胜率%": round(win * 100, 1)})

pf, pa, pe = (float(np.concatenate(x).mean()) for x in (pf_all, pa_all, pe_all))
print("\n===== M2 主表: Frame W(绑定∩白天, 334天, 逐会话验收) =====")
print(pd.DataFrame(tabW).to_string(index=False))
print(f"池化: 矫正附件3 {pa:.1f} | EGD {pe:.1f} | 融合(滚动w) {pf:.1f}")
print("(注: 本帧绑定口径不含跨午夜长步长目标, 比 Frame I 的 oracle 上界更易, 不可直接对比)")
print("\n===== M2 部署帧: Frame D(绑定段全天) =====")
print(pd.DataFrame(tabD).to_string(index=False))
print(f"\n滚动权重年均值(0:00→18:00): {[round(float(w_att[:, s].mean()), 3) for s in range(NS)]}")

# ---------- 6. 十分钟中心下沉(EGD 形状, 保能量) ----------
CS = np.full((NS, NEVAL, 144), np.nan)
for s in range(NS):
    for h in range(int(H0[s]), NL):
        sl = np.arange(6 * h, 6 * h + 6)
        egsl = egdp[:NEVAL, sl]
        egs = egsl.sum(axis=1)
        ok = egs > 1e-9
        shape = np.where(ok[:, None], egsl / np.where(ok, egs, 1.0)[:, None], 1.0 / 6)  # 和为1
        CS[s][:, sl] = C[:, s, h][:, None] * shape
        assert np.allclose(CS[s][:, sl].sum(axis=1)[ok], C[ok, s, h], rtol=1e-8), "下沉不保能量"

# ---------- 7. 落盘 ----------
if not SMOKE:
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "C_ses.npy", C)                      # (334,4,24) 行e↔实际日31+e, 绑定段外NaN
    np.save(OUT / "C_slot.npy", CS)                    # (4,334,144) 十分钟中心, 绑定段外NaN
    np.save(OUT / "w_hist.npy", w_att)                 # (334,4) 附件3权重史
    mm = pd.to_datetime(pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="小区负载").iloc[:, 0])
    tab_mw = []
    for mth in range(2, 13):
        rows = np.where(mm.dt.month.values[I0:I0 + NEVAL] == mth)[0]
        tab_mw.append({"月份": f"{mth}月", **{f"{H0[s]:>2}:00": round(float(w_att[rows, s].mean()), 3)
                                              for s in range(NS)}})
    with pd.ExcelWriter(OUT / "m2_结果.xlsx", engine="openpyxl") as w:
        pd.DataFrame(tabI).to_excel(w, sheet_name="FrameI审计连续性", index=False)
        pd.DataFrame(tabW).to_excel(w, sheet_name="FrameW主表", index=False)
        pd.DataFrame(tabD).to_excel(w, sheet_name="FrameD部署帧", index=False)
        pd.DataFrame(tab_mw).to_excel(w, sheet_name="月度权重", index=False)
    print(f"\n结果已保存 -> {OUT}")
else:
    print("\n[smoke] 通过")
