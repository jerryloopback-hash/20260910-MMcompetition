# -*- coding: utf-8 -*-
"""P2B 数据侦察：预测残差结构与 5x 风险敞口（供优化思路决策依据）"""
import pandas as pd, numpy as np

BASE = r"c:\Users\adminl\Desktop\20260910-MMcompetition\C题"
ATT2 = BASE + r"\附件\附件2.xlsx"
PRED_L = BASE + r"\p2_part1\预测结果\load_pred_ens.npy"
PRED_P = BASE + r"\p2_part1\预测结果\pv_pred_ens.npy"
ATT1 = BASE + r"\附件\附件1.xlsx"

# 附件1 电价（144 时段）
w1 = pd.ExcelFile(ATT1)
d1 = pd.read_excel(w1, sheet_name=w1.sheet_names[0])
price_raw = pd.to_numeric(d1.iloc[:, 1], errors='coerce').values.astype(float)
# 与 p1_solve 相同的日历口径：I_1 <- 第144行，I_k <- 第 k-1 行
price = np.concatenate([price_raw[143:], price_raw[:143]])
print(f"电价范围 [{price.min():.4f}, {price.max():.4f}] 元/kWh，5x 范围 [{5*price.min():.1f}, {5*price.max():.1f}] 元/kWh")
print(f"电价峰值时段: {int(np.argmax(price))}  (10min slot index, 0=0:00->0:10)")

# 实际值（附件2）
df_l = pd.read_excel(ATT2, sheet_name='小区负载')
df_p = pd.read_excel(ATT2, sheet_name='光伏发电实际功率')
load_mat = df_l.iloc[:, 1:].astype(float).values   # 365 x 144
pv_mat   = df_p.iloc[:, 1:].astype(float).values
print(f"\n附件2 shape {load_mat.shape}（365天 x 144时点）")

# 预测值（334 天 = 2025-02-01..12-31，对应附件2 的第 31..364 天）
pred_l = np.load(PRED_L)   # 334 x 144
pred_p = np.load(PRED_P)
assert pred_l.shape == pred_p.shape == (334, 144)
actual_l = load_mat[31:365]
actual_p = pv_mat[31:365]
print(f"预测 shape {pred_l.shape}，日期索引 附件2[31:365]")

# 残差 = 实际 - 预测（正值=低估；负载低估→缺口，光伏高估→缺口）
res_l = actual_l - pred_l
res_p = (actual_p - pred_p)  # 光伏：预测偏大(高估) → 残差为负 → 缺口来源
# 但光伏夜间预测=0、实际=0，残差集中在白天
hrs = (np.arange(144) + 1) * 10 / 60  # 各时点对应小时
day = (hrs > 6) & (hrs < 20)

print("\n=== 残差统计（kW）===")
for name, r in [("负载", res_l), ("光伏(白天)", res_p[:, day])]:
    print(f"{name}: 均值 {r.mean():+.1f}, 标准差 {r.std():.1f}, "
          f"p10 {np.percentile(r,10):.0f}, p50 {np.percentile(r,50):+.0f}, p90 {np.percentile(r,90):.0f}")

# 缺口触发率：负载低估(low) 与 光伏高估(over)
low_load = (res_l > 0)         # 实际负载比预测高
over_pv  = (res_p < 0)         # 实际光伏比预测低
print(f"\n负载低估率(全天) {low_load.mean()*100:.1f}%，白天 {low_load[:,day].mean()*100:.1f}%")
print(f"光伏高估率(白天) {over_pv[:,day].mean()*100:.1f}%")

# 按搭载 晚高峰时段（λ 峰值前后）低估率
peak_slot = int(np.argmax(price))
for wl in [peak_slot-3, peak_slot, peak_slot+3]:
    if 0 <= wl < 144:
        print(f"  时点{int(hrs[wl])*100+int((hrs[wl]%1)*60):04d}前后: 负载低估率 {low_load[:,wl].mean()*100:.0f}%")
print(f"晚高峰λ峰值时点 {hrs[peak_slot]:.1f} h")

# 月度残差（夏季 vs 其他）
mon = np.repeat(np.arange(1,13), [31,28,31,30,31,30,31,31,30,31,30,31])  # 2025
mon_p = mon[31:365]
print("\n=== 月度负载低估率与残差波动（风险集中季）===")
for m in [2,6,7,9]:
    sel = (mon_p == m)
    print(f"  {m}月: 负载低估率 {low_load[sel].mean()*100:.0f}%, 负载残差std {res_l[sel].std():.0f}, "
          f"光伏高估率 {over_pv[sel][:,day].mean()*100:.0f}%")

# 5x 保险是否划算的粗估：某时段多买 1 kWh 的保险成本=λ，避免的期望罚=5λ·P(缺口)
print("\n=== 保险经济性（报童视角：P(缺口) 超过 20% 就值得多买保险）===")
print("多买 1 kWh 保险成本 = λ；避免的期望紧急罚 = 5λ·P(缺口)。5λ·P > λ ⇔ P > 20%")

# 简化缺口能量定标（忽略储能优化，仅看"点预测直采"的供需缺口的量级上界）
# 计划供电 = 负载预测（直供+充电去向不管）+ 光伏预测；实际需求 = 实际负载
# 缺口 = max(0, 实际负载 − min(光伏实际, 光伏预测) − 预测不是关键…… 这里用最简：
#   计划可供电源(不含储能) = 负载预测 + 光伏预测×η（假设预测当全额计划采购）
#   简化为直接比较缺口成分：
gap_from_pv  = np.maximum(0, -(res_p))            # 光伏高估→缺口（kW，忽略储能调解）
gap_from_load = np.maximum(0, res_l)              # 负载低估→缺口
print("\n=== 简化缺口能量月均值（kWh/天，未计入储能,用于量级) ===")
print("光伏高估源缺口（kW·h/日,白天） vs 负载低估源缺口（kWh/日）:")
for m in [2,3,5,6,7,8,9,10,12]:
    sel = (mon_p == m)
    pv_gapE = (gap_from_pv[sel][:, day] * (1/6)).sum(axis=1).mean()
    ld_gapE = (gap_from_load[sel] * (1/6)).sum(axis=1).mean()
    print(f"  {m:2d}月: 光伏高估缺口 {pv_gapE:7.0f} kWh/日 | 负载低估缺口 {ld_gapE:7.0f} kWh/日")
pv_gapE_all = (gap_from_pv[:, day] * (1/6)).sum(axis=1).mean()
ld_gapE_all = (gap_from_load * (1/6)).sum(axis=1).mean()
print(f"  全期均值: 光伏 {pv_gapE_all:.0f} | 负载 {ld_gapE_all:.0f} kWh/日")
print(f"\n[解读] 若这些缺口全按 5λ 罚（λ≈均值0.8→罚4元/kWh），仅光伏高估源每日≈{pv_gapE_all*5*price.mean():,.0f} 元——"
      f"这就是分位修正可挽回的量级上限（储能可回收一部分，实际更小）。")