# -*- coding: utf-8 -*-
"""验证队友 v5 分位数交付: 形状/单调性/对齐/覆盖率"""
import numpy as np, pandas as pd

Q = [0.10, 0.50, 0.833, 0.90]
D = r"..\p2_part1\预测结果"

for name in ["load", "pv"]:
    q = np.load(rf"{D}\{name}_quantiles_v5.npy")
    p = np.load(rf"{D}\{name}_pred_ens_v5.npy")
    print(f"[{name}] 分位数 shape={q.shape} dtype={q.dtype}  点预测 shape={p.shape}")
    print(f"  有限值: {np.isfinite(q).all()}  范围 [{q.min():.1f}, {q.max():.1f}]")
    mono = bool(np.all((q[:,:,0] <= q[:,:,1]) & (q[:,:,1] <= q[:,:,2]) & (q[:,:,2] <= q[:,:,3])))
    print(f"  分位单调(P10<=P50<=P83.3<=P90): {mono}")
    print(f"  各分位全期均值: " + "  ".join(f"{qn}={q[:,:,i].mean():.1f}" for i, qn in enumerate(["P10","P50","P83.3","P90"])))
    print(f"  点预测均值: {p.mean():.1f}   P50均值: {q[:,:,1].mean():.1f}   P83.3-P50 全期均差: {(q[:,:,2]-q[:,:,1]).mean():.1f} kW")
    # 与附件2实际值的对齐: 334天均从2025-02-01起
    a = pd.read_excel(r"..\附件\附件2.xlsx", sheet_name="小区负载" if name=="load" else "光伏发电实际功率")
    act = a.iloc[31:365, 1:145].astype(float).values
    gap_up = np.maximum(act - q[:,:,2], 0)   # 真实值超出 P83.3 的量(低估缺口风险)
    print(f"  [对齐检查] 实际334天 vs 分位数334天: 形状匹配 {act.shape==q.shape[:2]}")
    print(f"  [P83.3残余风险] 真实值>P83.3 的平均超出量: {gap_up[gap_up>0].mean() if (gap_up>0).any() else 0:.1f} kW, 超出时段占比 {100*(gap_up>0).mean():.1f}%")

pvq = np.load(rf"{D}\pv_quantiles_v5.npy")
hrs = (np.arange(144)+1)*10/60
night = (hrs <= 6) | (hrs >= 20)
print(f"\n[PV夜间] 分位数最大值(应=0): {pvq[:, night, :].max():.1f}")

cov = pd.read_excel(rf"{D}\分位数覆盖率验证_v5.xlsx")
print("\n[覆盖率验证]")
print(cov.to_string(index=False))

# v5 带日期 xlsx 的日期范围
dx = pd.read_excel(rf"{D}\小区负载预测_2025_02-12_带日期_v5.xlsx", nrows=2)
print(f"\n[v5 xlsx 首列] 列名: {dx.columns[0]}, 首行日期: {dx.iloc[0,0]}")
