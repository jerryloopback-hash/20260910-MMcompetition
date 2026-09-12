# -*- coding: utf-8 -*-
"""
P3-M1 矫正层:附件3 光伏预报的(会话×步长)分层去偏
================================================
依据 p3_plan.md §3.1 / D8:
  对每个(会话 s, 步长 ℓ)层用滚动窗口估计系统性偏差 b̂ = mean(F−A), 矫正预报 F' = F − b̂。
  只用目标时刻之前已实现的数据(防泄漏)。
对齐口径(沿前期实证结论, 脚本内置自检):
  预报k小时 = 发布时刻 + (k−1) 小时;
  整点目标值 ↔ 附件2 实际的整点块均值, 时段沿用 P1/P2 "行标签=时段起点" 口径
  (小时块 h = 优化时段 k=6h+1..6h+6; h=0 块含前一日 ti=143)。
关键结构事实: 每会话步长24(预报24小时)的目标块恰在下一发布时刻结束,
  故所有 (s,ℓ) 层的样本池统一为 [d−W, d−1] 的有效残差, 无需逐层特判泄漏边界。
评估: 2025.2.1–12.31(334 天); 白天 = 目标小时 6..17([6:00,18:00)); MAE 单位 kW。
用法: python m1_correction.py [smoke]   产出: p3/m1_out/
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
P = d2.iloc[:, 1:1 + T].astype(float).values           # (365,144) 行标签=时段起点
assert P.shape == (365, T) and np.isfinite(P).all()

d3 = pd.read_excel(ATT3)
dates3 = pd.to_datetime(d3.iloc[:, 0].ffill(), format="%Y-%m-%d").values
sess_str = d3.iloc[:, 1].astype(str).values
F = d3.iloc[:, 2:2 + NL].astype(float).values.reshape(365, NS, NL)  # 逐日4会话按序
assert np.isfinite(F).all(), "附件3 预报含缺失值"
# 结构自检: 日期逐日递增且每日4行、会话顺序 0/6/12/18
assert (np.diff(dates3[::NS]).astype("timedelta64[D]").astype(int) == 1).all()
sess_ord = sorted(set(sess_str[:4]))
assert sess_ord == ["0:00", "12:00", "18:00", "6:00"], sess_ord

# ---------- 2. 整点实际值(优化时段口径的小时块均值) ----------
# h>=1: 附件2 当日 ti=6h-1 .. 6h+4;  h=0: 前一日 ti=143 + 当日 ti=0..4
A = np.empty((365, NL))
for h in range(1, NL):
    A[:, h] = P[:, 6 * h - 1:6 * h + 5].mean(axis=1)
A[1:, 0] = np.concatenate([P[:-1, 143:144], P[1:, 0:5]], axis=1).mean(axis=1)
A[0, 0] = np.nan                                       # 1.1 无前一日数据

# ---------- 3. 目标映射与残差 ----------
# 预报(l+1)小时 = 发布时刻 + l 小时 → 目标(日, 时)
di = np.arange(365)[:, None, None]
si = np.arange(NS)[None, :, None]
li = np.arange(NL)[None, None, :]
th = (H0[None, :, None] + li) % NL                     # 目标小时 (365,4,24)
td = di + (H0[None, :, None] + li >= NL).astype(int)   # 目标日 (365,4,24)
valid = td <= 364                                      # 12.31 会话的越年目标无实际值
Atgt = np.where(valid, A[np.clip(td, 0, 364), th], np.nan)
Rraw = np.where(valid, F - Atgt, np.nan)               # 原始残差 F−A (365,4,24)

# 对齐自检: 各会话预报-实际相关应很高(前期实证 ~0.98)
for s in range(NS):
    m = np.isfinite(Rraw[:, s, :]) & valid[:, s, :]
    c = np.corrcoef(F[:, s, :][m], Atgt[:, s, :][m])[0, 1]
    assert c > 0.9, f"会话{s} 对齐异常 corr={c:.3f}"
    print(f"[自检] 会话{H0[s]:>2}:00 预报-实际相关 {c:.4f}")
nz = F[:, 0, :5].mean()                                # 0:00 会话预报1-5小时 → 目标小时0-4
assert nz < 1.0, f"0:00 会话夜间预报非零: {nz}"
print(f"[自检] 0:00 会话预报1-5小时均值 {nz:.4f} ≈ 0 (夜间零值模式成立; 预报6小时→5:00 夏季可为正)")

# ---------- 4. 评估掩码与指标 ----------
m_issued = (di >= I_EVAL0) & (di <= I_EVAL1)           # 发布日在评估窗口内
m_day = (th >= 6) & (th < 18)                          # 白天: 目标小时 6..17
m_eval = m_issued & valid
M_H = m_eval & m_day                                   # 白天(整点口径)
M_A = m_eval                                           # 全天


def mae(Fa, mask):
    return float(np.nanmean(np.abs(Fa - Atgt)[mask]))


# 整点广播到 10 分钟(白天): 小时块 h 的预报填充其 6 个优化时段
# 时段 k 的小时 h=k//6; h>=H0[s] 由当日会话步长 h-H0[s] 覆盖, 否则由前一日会话覆盖
Sact = np.empty((365, T))
Sact[1:, 0] = P[:-1, 143]
Sact[1:, 1:] = P[1:, :T - 1]
Sact[0, 0] = np.nan
slot_day = np.zeros((365, T), bool)
for h in range(6, 18):
    slot_day[:, 6 * h:6 * h + 6] = True


def slot_mae(Fa, day0=I_EVAL0, day1=I_EVAL1):
    """Fa:(365,4,24) → 各会话小时值广播到时段后与 Sact 的白天 MAE(会话等权池化)"""
    errs = []
    ks = np.arange(T)
    for s in range(NS):
        shift = (ks // 6 - H0[s]) % NL                 # 时段 k → 该会话的步长
        rowk = np.where(ks // 6 >= H0[s], 0, -1)       # h<H0[s] 的时段由前一日同会话预报覆盖
        idx = np.clip(np.arange(365)[:, None] + rowk[None, :], 0, 364)
        B = Fa[idx, s, shift[None, :]]                 # (365,144)
        errs.append(np.abs(B[day0:day1 + 1] - Sact[day0:day1 + 1])[slot_day[day0:day1 + 1]])
    return float(np.mean(np.concatenate(errs)))


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


WIN_LIST = [28, 56, 90, 0]                             # 0 = expanding
rows = []
Bstar, cnt_star, Wstar = None, None, None
best = np.inf
for W in WIN_LIST:
    B, cnt = bias_of(W)
    Fc = np.maximum(F - B, 0.0)
    mh, ms = mae(Fc, M_H), slot_mae(Fc)
    tag = "expanding" if W == 0 else str(W)
    rows.append({"窗口W": tag, "白天MAE_整点": round(mh, 2), "白天MAE_10min": round(ms, 2)})
    print(f"[窗口] W={tag:>9}: 白天MAE 整点 {mh:7.2f} / 10min {ms:7.2f}")
    if ms < best:
        best, Bstar, cnt_star, Wstar = ms, B, cnt, W
print(f"[窗口] 选定 W* = {Wstar or 'expanding'} (10min 白天 MAE 最小)")

Fc = np.maximum(F - Bstar, 0.0)
clip = float((Fc < F - Bstar - 1e-9)[m_eval].mean())
print(f"[自检] 矫正后截零比例 {clip * 100:.2f}% (黎明/黄昏)")
res_oos = np.where(valid, Atgt - Fc, np.nan)
print(f"[自检] 矫正后白天残差均值 {np.nanmean((Atgt - Fc)[M_H]):+.2f} kW (应≈0)")
print(f"[自检] 样本池最小有效数 {cnt_star[m_eval].min():.0f} (评估日内)")

# ---------- 6. 汇总表 ----------
raw_h, raw_s = mae(F, M_H), slot_mae(F)
egde = np.abs(np.load(PRED / "pv_pred_ens_v5.npy")[:NEVAL] - Sact[I_EVAL0:I_EVAL1 + 1])[slot_day[I_EVAL0:I_EVAL1 + 1]]
print("\n===== M1 校准 =====")
print(f"原始附件3 白天 MAE: 整点 {raw_h:.1f} / 10min {raw_s:.1f} kW")
print(f"EGD v5 白天 10min MAE: {float(np.mean(egde)):.1f} kW (参照 ~250)")
print(f"矫正后({Wstar or 'expanding'}) 白天 MAE: 整点 {mae(Fc, M_H):.1f} / "
      f"10min {slot_mae(Fc):.1f} kW")

# 分会话
tab_s = []
for s in range(NS):
    tab_s.append({"会话": f"{H0[s]:>2}:00",
                  "原始_白天": round(mae(F, m_eval & ((th >= 6) & (th < 18)) & (si == s)), 1),
                  "矫正_白天": round(mae(Fc, m_eval & ((th >= 6) & (th < 18)) & (si == s)), 1),
                  "原始_全天": round(mae(F, m_eval & (si == s)), 1),
                  "矫正_全天": round(mae(Fc, m_eval & (si == s)), 1)})

# 分步长带(白天)
bands = [("1-4", 0, 4), ("5-12", 4, 12), ("13-24", 12, 24)]
tab_b = []
for name, l0, l1 in bands:
    mb = m_eval & (li >= l0) & (li < l1) & m_day
    tab_b.append({"步长带": name, "原始": round(mae(F, mb), 1), "矫正": round(mae(Fc, mb), 1)})

# 分月(白天, 整点口径)
tab_m = []
for mth in range(2, 13):
    dm = np.where(dates2.dt.month.values == mth)[0]
    mm = m_eval & m_day & np.isin(di, dm)
    tab_m.append({"月份": f"{mth}月", "原始": round(mae(F, mm), 1), "矫正": round(mae(Fc, mm), 1)})

# 层内统计: 系统性成分占白天 MSE 份额(按会话汇总)
tab_l = []
for s in range(NS):
    mb = m_eval & m_day & (si == s)
    r2 = np.nansum((F - Atgt)[mb] ** 2)
    b2 = np.nansum(Bstar[mb] ** 2)
    tab_l.append({"会话": f"{H0[s]:>2}:00", "系统成分占MSE": f"{b2 / r2 * 100:.1f}%",
                  "平均|偏差|": round(float(np.nanmean(np.abs(Bstar[mb]))), 1)})

# ---------- 7. 落盘 ----------
if not SMOKE:
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "F_raw.npy", F)
    np.save(OUT / "F_corr.npy", Fc)
    np.save(OUT / "bias_hist.npy", Bstar)
    np.save(OUT / "R_oos.npy", res_oos)
    with pd.ExcelWriter(OUT / "m1_结果.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"口径": "原始附件3 白天 整点", "MAE": round(raw_h, 2)},
                      {"口径": "原始附件3 白天 10min", "MAE": round(raw_s, 2)},
                      {"口径": "EGD 白天 10min", "MAE": round(float(np.mean(egde)), 2)},
                      {"口径": f"矫正(W={Wstar or 'expanding'}) 白天 整点", "MAE": round(mae(Fc, M_H), 2)},
                      {"口径": f"矫正(W={Wstar or 'expanding'}) 白天 10min", "MAE": round(slot_mae(Fc), 2)}]
                     ).to_excel(w, sheet_name="校准与总表", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="窗口对比", index=False)
        pd.DataFrame(tab_s).to_excel(w, sheet_name="分会话", index=False)
        pd.DataFrame(tab_b).to_excel(w, sheet_name="分步长带", index=False)
        pd.DataFrame(tab_m).to_excel(w, sheet_name="分月", index=False)
        pd.DataFrame(tab_l).to_excel(w, sheet_name="层内统计", index=False)
    print(f"\n结果已保存 -> {OUT}")
else:
    print("\n[smoke] 通过")
