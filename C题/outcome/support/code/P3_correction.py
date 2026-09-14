# -*- coding: utf-8 -*-
"""
P3-M1 矫正层:附件3 光伏预报的(会话×步长)分层去偏
================================================
依据 p3_plan.md §3.1 / D8:
  对每个(会话 s, 步长 ℓ)层用滚动窗口估计系统性偏差 b̂ = mean(F−A), 矫正预报 F' = F − b̂。
  只用目标时刻之前已实现的数据(防泄漏)。前期分析的 566→377 用的是全样本偏差(自带泄漏),
  M1 的增量 = 滚动样本外版本。
指标口径(与前期分析完全一致, 保证 566 锚点可复现):
  整点级 MAE;整点实际 = 附件2 列 ti=6h..6h+5 的均值;白天 = 目标小时 5..19(15 小时);
  主表用 334 天(2025.2.1-12.31, 与 Q2/Q3 落地窗口一致), 另报全年行以对上 566 锚点;
  P1/P2 优化时段口径(ti=6h-1..6h+4)作 ±10min 敏感性行。
关键结构事实: 每会话步长24(预报24小时)的目标块恰在下一发布时刻结束,
  故所有 (s,ℓ) 层的滚动样本池统一为 [d−W, d−1] 的有效残差, 无需逐层特判泄漏边界。
用法: python P3_correction.py [smoke]   产出: p3/m1_out/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent          # C题/
ATT2 = BASE / "附件" / "附件2.xlsx"
ATT3 = BASE / "附件" / "附件3.xlsx"
PRED = BASE / "p2_part1" / "预测结果"
OUT = Path(__file__).resolve().parent / "m1_out"

T = 144
NS, NL = 4, 24
H0 = np.array([0, 6, 12, 18])                          # 各会话发布时刻
I_EVAL0 = 31                                           # 2025-02-01 在附件2 的行号
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
NEVAL = 40 if SMOKE else 334
I_EVAL1 = I_EVAL0 + NEVAL - 1                          # 评估的最后一个发布日

# ---------- 1. 数据 ----------
d2 = pd.read_excel(ATT2, sheet_name="光伏发电实际功率")
dates2 = pd.to_datetime(d2.iloc[:, 0])
P = d2.iloc[:, 1:1 + T].astype(float).values           # (365,144)
assert P.shape == (365, T) and np.isfinite(P).all()

d3 = pd.read_excel(ATT3)
dates3 = pd.to_datetime(d3.iloc[:, 0].ffill(), format="%Y-%m-%d").values
sess_str = d3.iloc[:, 1].astype(str).values
F = d3.iloc[:, 2:2 + NL].astype(float).values.reshape(365, NS, NL)  # 逐日4会话按序
assert np.isfinite(F).all(), "附件3 预报含缺失值"
assert (np.diff(dates3[::NS]).astype("timedelta64[D]").astype(int) == 1).all()
assert sorted(set(sess_str[:4])) == ["0:00", "12:00", "18:00", "6:00"]
# 会话顺序须逐日恒为 (0,6,12,18): 目标映射按位置取 H0[s], 顺序错位即全盘错位。
_sess_expect = np.array([f"{h}:00" for h in H0])
assert (np.asarray(sess_str, dtype=object).reshape(365, NS) == _sess_expect[None, :]).all(), "会话顺序非逐日恒定"

# ---------- 2. 整点实际值 ----------
A_nat = P.reshape(365, NL, 6).mean(axis=2)             # 前期口径: ti 6h..6h+5 (主用)
A_opt = np.empty((365, NL))                            # P1/P2 优化时段口径: ti 6h-1..6h+4
A_opt[1:, 0] = np.nan
for h in range(1, NL):
    A_opt[:, h] = P[:, 6 * h - 1:6 * h + 5].mean(axis=1)
A_opt[1:, 0] = np.concatenate([P[:-1, 143:144], P[1:, 0:5]], axis=1).mean(axis=1)

# ---------- 3. 目标映射与残差 ----------
di = np.arange(365)[:, None, None]
si = np.arange(NS)[None, :, None]
li = np.arange(NL)[None, None, :]
th = (H0[None, :, None] + li) % NL                     # 目标小时 (365,4,24)
td = di + (H0[None, :, None] + li >= NL).astype(int)   # 目标日
valid = td <= 364                                      # 12.31 会话的越年目标无实际值
Atgt = np.where(valid, A_nat[np.clip(td, 0, 364), th], np.nan)
Rraw = np.where(valid, F - Atgt, np.nan)               # 原始残差 F−A (365,4,24)

for s in range(NS):                                    # 对齐自检(前期实证 ~0.97)
    m = np.isfinite(Rraw[:, s, :])
    c = np.corrcoef(F[:, s, :][m], Atgt[:, s, :][m])[0, 1]
    assert c > 0.9, f"会话{s} 对齐异常 corr={c:.3f}"
    print(f"[自检] 会话{H0[s]:>2}:00 预报-实际相关 {c:.4f}")
nz = F[:, 0, :5].mean()
assert nz < 1.0, f"0:00 会话夜间预报非零: {nz}"
print("[自检] 0:00 会话预报1-5小时均值 ≈ 0 (夜间零值模式成立; 预报6小时→5:00 夏季可为正)")

# ---------- 4. 指标掩码(白天 = 目标小时 5..19) ----------
m_day = (th >= 5) & (th <= 19)
m_issued_y = valid & np.ones_like(valid, bool)         # 全年(发布日 0..364)
m_issued_334 = valid & (di >= I_EVAL0) & (di <= I_EVAL1)
M_Y = m_issued_y & m_day                               # 全年白天(复现 566 锚点)
M_334 = m_issued_334 & m_day                           # 334天白天(主口径)


def mae(Fa, mask):
    v = np.abs(Fa - Atgt)[mask]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


# ---------- 5. 滚动去偏 ----------
def trailing_mean(r, W):
    """r:(365,) 含 NaN → t[d]=mean(r[max(0,d-W):d])(只用 d 之前); 返回均值与有效计数。"""
    D = len(r)
    v = np.where(np.isfinite(r), r, 0.0)
    c = np.isfinite(r).astype(float)
    cv = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    if W is None or W <= 0:                            # expanding
        s, n = cv[:D], cc[:D]
    else:
        lo = np.maximum(0, np.arange(D) - W)
        s, n = cv[np.arange(D)] - cv[lo], cc[np.arange(D)] - cc[lo]
    return np.where(n > 0, s / np.maximum(n, 1.0), 0.0), n


def bias_of(W):
    B = np.empty((365, NS, NL))
    cnt = np.empty((365, NS, NL))
    for s in range(NS):
        for l in range(NL):
            B[:, s, l], cnt[:, s, l] = trailing_mean(Rraw[:, s, l], W)
    return B, cnt


WIN_LIST = [7, 14, 21, 28, 42, 56, 90, 0]             # 0 = expanding; 全扫以复核平台
W_FIX = 28                                             # 预注册选窗: 平台中心, 不追极值(见 m1_model.md §3)
rows_win = []
cache = {}
for W in WIN_LIST:
    B, cnt = bias_of(W)
    Fc = np.maximum(F - B, 0.0)
    v, vy = mae(Fc, M_334), mae(Fc, M_Y)
    tag = "expanding" if W == 0 else str(W)
    cache[W] = (B, cnt)
    rows_win.append({"窗口W": tag, "白天MAE_334天": round(v, 2), "白天MAE_全年": round(vy, 2)})
    print(f"[窗口] W={tag:>9}: 白天MAE 334天 {v:7.2f} / 全年 {vy:7.2f}")
arg_v, arg_tag = min((r["白天MAE_334天"], r["窗口W"]) for r in rows_win)
plateau = [r["白天MAE_334天"] for r in rows_win if r["窗口W"] in ("14", "21", "28", "42")]
# 选窗规则: 平台中心而非 argmin。全扫显示 14–42 构成平台, 28 居中且样本量足以撑满 96 层;
# argmin 落在 21, 与 28 仅差零点几 kW, 属噪声级, 追极值没有意义。此规则与模型文档写死一致。
Wstar = W_FIX
Bstar, cnt_star = cache[Wstar]
_gap = abs(mae(np.maximum(F - Bstar, 0.0), M_334) - arg_v)
print(f"[窗口] 全扫 argmin={arg_tag} ({arg_v:.2f}); 平台 14–42 为 {min(plateau):.2f}–{max(plateau):.2f}")
print(f"[窗口] 选定 W* = {Wstar} (平台中心, 不追极值; 与 argmin 差 {_gap:.2f} kW)")

Fc = np.maximum(F - Bstar, 0.0)
clip = float((Fc < F - Bstar - 1e-9)[valid].mean())
print(f"[自检] 矫正后截零比例 {clip * 100:.2f}% (黎明/黄昏)")
res_oos = np.where(valid, Atgt - Fc, np.nan)
print(f"[自检] 矫正后白天残差均值 {np.nanmean((Atgt - Fc)[M_334]):+.2f} kW (应≈0)")
print(f"[自检] 样本池最小有效数 {cnt_star[m_issued_334 & m_day].min():.0f}")

# ---------- 6. 汇总表 ----------
raw_y, raw_334 = mae(F, M_Y), mae(F, M_334)
# EGD 参照(334天, 同为整点级): pred (334,144) → 同口径整点均值
egdp = np.load(PRED / "pv_pred_ens_v5.npy")
A_egd = egdp.reshape(NEVAL, NL, 6).mean(axis=2)
egd_334 = float(np.abs(A_egd - A_nat[I_EVAL0:I_EVAL1 + 1])[:, 5:20].mean())
# ±10min 敏感性: P1/P2 优化时段口径的整点实际
Atgt_opt = np.where(valid, A_opt[np.clip(td, 0, 364), th], np.nan)
sens_raw = float(np.nanmean(np.abs((F - Atgt_opt)[M_334])))
sens_corr = float(np.nanmean(np.abs((Fc - Atgt_opt)[M_334])))

print("\n===== M1 校准(整点级, 白天=5..19) =====")
print(f"原始附件3 全年   : {raw_y:.1f} kW   <- 应≈566.6 (锚点复现)")
print(f"原始附件3 334天  : {raw_334:.1f} kW")
print(f"矫正后(W*={Wstar if Wstar else 'expanding'}) 全年 : {mae(Fc, M_Y):.1f} / 334天: {mae(Fc, M_334):.1f} kW")
print(f"EGD v5 334天整点 : {egd_334:.1f} kW   <- 前期 ~249.5")
print(f"±10min敏感性(334天, P1/P2时段口径): 原始 {sens_raw:.1f} → 矫正 {sens_corr:.1f}")

tab_s = []
for s in range(NS):
    ms_y, ms_334 = M_Y & (si == s), M_334 & (si == s)
    tab_s.append({"会话": f"{H0[s]:>2}:00",
                  "原始_全年": round(mae(F, ms_y), 1), "矫正_全年": round(mae(Fc, ms_y), 1),
                  "原始_334天": round(mae(F, ms_334), 1), "矫正_334天": round(mae(Fc, ms_334), 1)})

bands = [("1-4", 0, 4), ("5-12", 4, 12), ("13-24", 12, 24)]
tab_b = []
for name, l0, l1 in bands:
    mb = M_334 & (li >= l0) & (li < l1)
    tab_b.append({"步长带": name, "原始": round(mae(F, mb), 1), "矫正": round(mae(Fc, mb), 1)})

tab_m = []
for mth in range(1, 13):
    dm = np.where(dates2.dt.month.values == mth)[0]
    mm_y = M_Y & np.isin(di, dm)
    mm_334 = M_334 & np.isin(di, dm)
    tab_m.append({"月份": f"{mth}月", "原始_全年": round(mae(F, mm_y), 1),
                  "矫正_全年": round(mae(Fc, mm_y), 1),
                  "矫正_334天": round(mae(Fc, mm_334), 1)})

tab_l = []                                             # 系统性成分占白天 MSE 份额(334天)
for s in range(NS):
    mb = M_334 & (si == s)
    r2 = np.nansum((F - Atgt)[mb] ** 2)
    b2 = np.nansum(Bstar[mb] ** 2)
    tab_l.append({"会话": f"{H0[s]:>2}:00", "系统成分占MSE": f"{b2 / r2 * 100:.1f}%",
                  "平均|滚动偏差|": round(float(np.nanmean(np.abs(Bstar[mb]))), 1)})

# ---------- 7. 落盘 ----------
if not SMOKE:
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "F_raw.npy", F)
    np.save(OUT / "F_corr.npy", Fc)
    np.save(OUT / "bias_hist.npy", Bstar)
    np.save(OUT / "R_oos.npy", res_oos)
    np.save(OUT / "A_nat.npy", A_nat)
    with pd.ExcelWriter(OUT / "m1_结果.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"口径": "原始附件3 白天 全年(锚点复现)", "MAE_kW": round(raw_y, 2)},
            {"口径": "原始附件3 白天 334天", "MAE_kW": round(raw_334, 2)},
            {"口径": f"矫正(W={Wstar if Wstar else 'expanding'}) 白天 全年", "MAE_kW": round(mae(Fc, M_Y), 2)},
            {"口径": f"矫正(W={Wstar if Wstar else 'expanding'}) 白天 334天", "MAE_kW": round(mae(Fc, M_334), 2)},
            {"口径": "EGD v5 白天 334天(整点)", "MAE_kW": round(egd_334, 2)},
            {"口径": "敏感性: P1/P2时段口径 原始→矫正(334天)", "MAE_kW": f"{sens_raw:.1f} → {sens_corr:.1f}"},
        ]).to_excel(w, sheet_name="校准与总表", index=False)
        pd.DataFrame(rows_win).to_excel(w, sheet_name="窗口对比", index=False)
        pd.DataFrame(tab_s).to_excel(w, sheet_name="分会话", index=False)
        pd.DataFrame(tab_b).to_excel(w, sheet_name="分步长带", index=False)
        pd.DataFrame(tab_m).to_excel(w, sheet_name="分月", index=False)
        pd.DataFrame(tab_l).to_excel(w, sheet_name="层内统计", index=False)
    print(f"\n结果已保存 -> {OUT}")
else:
    print("\n[smoke] 通过")
