# -*- coding: utf-8 -*-
"""
附件4电价特征审计 v2（2026-09-13, 讨论阶段: 只做特征分析, 不建预测模型）
  A1 结构: rank-1 SVD vs 附件1形状;  形状偏差的周周期性;  真噪声下限
  A2 水平过程: 周期/自相关/节假日/漂移/1月冷启动
  A3 噪声结构: 分布/自相关/异方差
  A4 方差分解 + 朴素基准阶梯
"""
import numpy as np
import pandas as pd
from scipy import stats

A4 = r"D:\CUMCM2026Problems\C题\附件\附件4.xlsx"
A1 = r"D:\CUMCM2026Problems\C题\附件\附件1.xlsx"

a4 = pd.read_excel(A4, index_col=0)
a4.columns = range(144)
P = a4.values.astype(float)
dates = pd.to_datetime(a4.index)
dow = dates.dayofweek.values
shape = pd.to_numeric(pd.read_excel(A1)["电价"], errors="coerce").values.astype(float)
shape_mean = shape.mean()
lev = P.mean(axis=1)

# ---------- A1 结构 ----------
print("=== A1 结构 ===")
# (a) rank-1 SVD
Pc = P - P.mean()
U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
v2 = S[1]**2 / (S**2).sum(); v3 = S[2]**2 / (S**2).sum()
rank1 = U[:, :1] @ np.diag(S[:1]) @ Vt[:1] + P.mean()
r1_shape = np.corrcoef(Vt[0], shape - shape.mean())[0, 1]
res_svd = P - rank1
print(f"SVD奇异值前5: {np.round(S[:5],1)}  rank1/2/3方差占比: {S[0]**2/(S**2).sum()*100:.2f}% / {v2*100:.2f}% / {v3*100:.2f}%")
print(f"rank1右奇异向量与附件1形状相关: {r1_shape:.4f}")
print(f"rank1拟合残差MAE={np.abs(res_svd).mean()*1e3:.2f}e-3")
# (b) 附件1形状×逐日水平
num = P @ shape; den = shape @ shape
lev_fit = num / den
Pfit = np.outer(lev_fit, shape)
resid = P - Pfit
r2 = 1 - (resid**2).sum(axis=1) / ((P - P.mean(axis=1, keepdims=True))**2).sum(axis=1)
corr = np.array([np.corrcoef(P[d], shape)[0, 1] for d in range(365)])
print(f"形状×水平: 逐日相关 min={corr.min():.4f} 中位={np.median(corr):.4f}; R² 中位={np.median(r2):.4f}")
print(f"  残差MAE={np.abs(resid).mean()*1e3:.2f}e-3 (rank1 SVD为{np.abs(res_svd).mean()*1e3:.2f}e-3)")
# (c) 形状偏差的周周期性: dev_d ≈ dev_{d-7}?
dev_mae7 = np.abs(resid[7:] - resid[:-7]).mean()
dev_mae0 = np.abs(resid[7:] - 0).mean()
print(f"形状偏差 dev=P-level×shape: |dev_d - dev_(d-7)| MAE={dev_mae7*1e3:.2f}e-3  vs dev自身尺度{dev_mae0*1e3:.2f}e-3  → 压缩率{1-dev_mae7/dev_mae0*1:.2f}")
dev_mae1 = np.abs(resid[1:] - resid[:-1]).mean()
print(f"  对照 |dev_d - dev_(d-1)| MAE={dev_mae1*1e3:.2f}e-3 (压缩率{1-dev_mae1/dev_mae0:.2f})")
# (d) 星期×时段 稳定形状: 逐dow的归一化均值曲线差异
dow_curves = np.array([norm_mean for norm_mean in
                       [ (P[dow==w].mean(axis=0)) / (P[dow==w].mean()) * shape_mean for w in range(7) ]])
print(f"7条星期形状两两最大差异(相对形状均值): {np.max(np.abs(dow_curves - shape))*1e3:.1f}e-3 (点对点)")
# (e) 真噪声下限: 每日 level×shape + 星期×时段偏差均值 都去掉后
dev_dow = np.zeros_like(P)
for w in range(7):
    dev_dow[dow == w] = resid[dow == w].mean(axis=0)
resid2 = P - Pfit - dev_dow
print(f"再扣星期偏差后的残差MAE={np.abs(resid2).mean()*1e3:.2f}e-3  (这才是白噪声下限的估计)")
mon = pd.Series(np.abs(resid).mean(axis=1), index=dates).groupby(dates.month).mean()
print(f"形状×水平残差MAE分月(×1e-3): {np.round(mon.values*1e3,1)}")
res2m = pd.Series(np.abs(resid2).mean(axis=1), index=dates).groupby(dates.month).mean()
print(f"扣星期偏差后残差MAE分月(×1e-3): {np.round(res2m.values*1e3,1)}")

# ---------- A2 水平过程 ----------
print("\n=== A2 每日水平(日均电价)过程 ===")
lv = pd.Series(lev, index=dates)
print(f"水平范围 [{lev.min():.4f}, {lev.max():.4f}], 均值 {lev.mean():.4f}")
acf = [pd.Series(lev).autocorr(lag) for lag in range(1, 22)]
print("水平ACF lag1..21:", np.round(acf, 3))
d7 = lev[7:] - lev[:-7]; d1 = lev[1:] - lev[:-1]
print(f"|Δlag7| 中位={np.median(np.abs(d7)):.5f}  |Δlag1| 中位={np.median(np.abs(d1)):.5f}")
dow_eff = lv.groupby(lv.index.dayofweek).mean()
print("星期0-6均值:", np.round(dow_eff.values, 4))
wk = dow_eff.values
print(f"周末(5,6)/工作日 = {np.mean(wk[5:]) / np.mean(wk[:5]):.4f}")
mm = lv.groupby(dates.month).mean()
print("月度水平:", np.round(mm.values, 4))
err7 = np.abs(lev[7:] - lev[:-7])
e7 = pd.Series(err7, index=dates[7:])
print("lag7水平MAE分月(×1e-3):", np.round(e7.groupby(e7.index.month).mean().values*1e3, 1), " 全年", f"{err7.mean()*1e3:.2f}")
worst = e7.sort_values(ascending=False).head(20)
holidays = ["2025-01-01","2025-01-28","2025-01-29","2025-01-30","2025-01-31","2025-02-01","2025-02-02","2025-02-03","2025-02-04",
            "2025-04-04","2025-04-05","2025-04-06","2025-05-01","2025-05-02","2025-05-03","2025-05-04","2025-05-05",
            "2025-05-31","2025-06-01","2025-06-02","2025-10-01","2025-10-02","2025-10-03","2025-10-04","2025-10-05","2025-10-06","2025-10-07","2025-10-08"]
hol = pd.to_datetime(holidays)
worst_in_hol = sum(any(abs((d - h).days) <= 2 for h in hol) for d in worst.index)
print(f"lag7水平误差最差20天中落在节假日±2天内的: {worst_in_hol} 天")
print("  最差8天:", [(str(d.date()), f"{v*1e3:.1f}") for d, v in worst.head(8).items()])
jan = lv[dates.month == 1]
print(f"1月水平: lag7段MAE={np.abs(jan.values[7:] - jan.values[:-7]).mean()*1e3:.2f}e-3  前7天均值基准={np.abs(jan.values[7:] - jan.values[:7].mean()).mean()*1e3:.2f}e-3")
# 2025-01-01 是周三; 2.1起预测, 历史只有31天 -> dow结构已可学
print(f"1月 dow均值: {[np.round(jan[jan.index.dayofweek==w].mean(),3) for w in range(7)]}")

# ---------- A3 噪声(扣星期偏差后) ----------
print("\n=== A3 噪声结构(残差2 = P - level×shape - 星期×时段偏差) ===")
z = resid2.ravel()
print(f"std={z.std():.5f}  skew={stats.skew(z):.2f}  kurtosis={stats.kurtosis(z):.2f}")
jb = stats.jarque_bera(z[::37])
print(f"Jarque-Bera(1/37抽样): stat={jb.statistic:.0f} p={jb.pvalue:.2e}")
zr = pd.Series(z)
print("池化ACF lag1..6:", np.round([zr.autocorr(l) for l in range(1, 7)], 3))
ac1 = [pd.Series(resid2[:, ti]).autocorr(1) for ti in range(144) if resid2[:, ti].std() > 1e-9]
print(f"逐时段lag1自相关分位[10,50,90]: [{np.percentile(ac1,10):.2f},{np.percentile(ac1,50):.2f},{np.percentile(ac1,90):.2f}]")
sc = pd.Series(np.abs(resid2).mean(axis=1), index=dates)
print(f"噪声尺度 vs 水平相关: {np.corrcoef(lev, sc.values)[0,1]:.3f}")

# ---------- A4 方差分解 + 基准阶梯 ----------
print("\n=== A4 方差分解(对总方差) ===")
tot = ((P - P.mean())**2).mean()
print(f"总方差={tot:.5f}")
print(f"  附件1形状×水平 解释: {1-((resid)**2).mean()/tot:.4f}")
print(f"  + 星期×时段偏差 解释: {1-((resid2)**2).mean()/tot:.4f}")
print(f"  rank-1 SVD 解释: {S[0]**2/(S**2).sum():.4f}")

print("\n=== 朴素基准阶梯(严格day-ahead滚动, 2.2-12.31, MAE×1e-3) ===")
i_feb1 = int(np.where(dates == pd.Timestamp('2025-02-01'))[0][0])
bench = {}
for d in range(i_feb1, 365):
    if d - i_feb1 == 0: continue
    true = P[d]
    e = {}
    e["B0 naive-day(昨日曲线)"] = np.abs(P[d-1] - true).mean()
    if d >= 7:
        e["B1 naive-week(上周同日)"] = np.abs(P[d-7] - true).mean()
        e["B2 shape×lag7水平"] = np.abs(shape * (lev[d-7]/shape_mean) - true).mean()
        if d >= 14:
            dv = lev[d-7] - lev[d-14]
            e["B3 shape×lag7+1阶外推"] = np.abs(shape * ((lev[d-7]+dv)/shape_mean) - true).mean()
        e["B4 B2+上周同dow形状偏差"] = np.abs(shape*(lev[d-7]/shape_mean) + resid[d-7] - true).mean()
    for kk, v in e.items():
        bench.setdefault(kk, []).append(v)
for kk, v in bench.items():
    print(f"  {kk:28s} MAE={np.mean(v)*1e3:6.2f}  RMSE={np.sqrt(np.mean(np.square(v)))*1e3:6.2f}")
print(f"  {'下限(真水平+真星期偏差)':28s} MAE={np.abs(resid2[i_feb1:]).mean()*1e3:6.2f}  RMSE={np.sqrt((resid2[i_feb1:]**2).mean())*1e3:6.2f}")
print("  v6存档对照: ShapeLevel≈?  EGD集成=41.58  LGB=43.56 (电价验证误差报告_按月_v6.xlsx)")
rep = pd.read_excel(r"D:\CUMCM2026Problems\C题\预测结果\电价验证误差报告_按月_v6.xlsx", sheet_name="全期误差")
print(rep.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
