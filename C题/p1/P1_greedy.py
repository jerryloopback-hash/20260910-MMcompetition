# -*- coding: utf-8 -*-
"""
P1 对照实验：规则式"谷充峰放"启发式（文献规则法的代表） vs LP 最优。
规则（对应文献中常见的分时电价充放电策略，如王凌云2020、王先齐的调度策略图）：
  R1 光伏优先直供负载；光伏盈余全部充电（受并网侧功率上限与 SOC 上限约束），仍多余则弃光；
  R2 电价 λ_t ≤ 分位阈值 q_lo（全日价格 20% 分位）：电网充电，至功率上限/SOC 上限；
  R3 电价 λ_t ≥ 分位阈值 q_hi（全日价格 80% 分位）：储能放电供负载，至功率上限/SOC 下限；
  R4 其余负载缺口一律购电直供；
  R5 收尾修正：E(24:00) ≠ 6000 时，超出则从最贵的充电段扣减、不足则从最便宜的充电段补足。
该基线只看当段价格分位、不带前瞻，用以量化 LP 全局优化的价值。
"""
from pathlib import Path
import numpy as np
import openpyxl

OUTDIR = Path(__file__).resolve().parent
BASE = OUTDIR.parent

wb = openpyxl.load_workbook(BASE / "附件" / "附件1.xlsx", data_only=True)
rows = list(wb["Sheet1"].iter_rows(min_row=2, values_only=True))
raw_price = np.array([r[1] for r in rows], float)
raw_load = np.array([r[2] for r in rows], float)
raw_pv = np.array([r[3] for r in rows], float)
price = np.concatenate([raw_price[143:], raw_price[:143]])
load = np.concatenate([raw_load[143:], raw_load[:143]])
pv = np.concatenate([raw_pv[143:], raw_pv[:143]])

DT, T = 1 / 6.0, 144
ETA = 0.9
EMIN, EMAX, E0 = 1200.0, 10800.0, 6000.0
PMAX = 5000.0
CH = PMAX * DT          # 每段最大并网侧充电能量 833.33 kWh
DIS = PMAX * DT         # 每段最大并网侧放电能量 833.33 kWh

q_lo, q_hi = np.quantile(price, 0.20), np.quantile(price, 0.80)
print(f"价格分位阈值: q20={q_lo:.4f}, q80={q_hi:.4f} 元/kWh")

b = np.zeros(T); c = np.zeros(T); d = np.zeros(T); a = np.zeros(T); g = np.zeros(T)
q_cur = np.zeros(T)
E = E0
for t in range(T):
    # R1 光伏直供 + 盈余充电
    g[t] = min(pv[t], load[t])
    surplus = pv[t] - g[t]
    headroom = EMAX - E
    c[t] = min(surplus, PMAX, headroom / (ETA * DT)) * DT / DT  # 功率口径：并网侧 ≤ PMAX
    c[t] = min(surplus, PMAX, headroom / (ETA * DT))
    q_cur[t] = surplus - c[t]
    E += ETA * c[t] * DT
    # R3 峰时放电
    if price[t] >= q_hi:
        need = load[t] - g[t]
        d[t] = max(0.0, min(need, PMAX, (E - EMIN) * ETA / DT))
        E -= d[t] * DT / ETA
    # R2 谷时充电
    elif price[t] <= q_lo:
        room = min(PMAX - c[t], (EMAX - E) / (ETA * DT))
        b[t] = max(0.0, room)
        E += ETA * b[t] * DT
    # R4 缺口直购
    a[t] = max(0.0, load[t] - g[t] - d[t])

# R5 收尾修正：闭环到 6000（SOC 感知：维护轨迹，扣减/补充量受其后节点余量约束）
SOC = np.empty(T + 1); SOC[0] = E0
for t in range(T):
    SOC[t + 1] = SOC[t] + ETA * (b[t] + c[t]) * DT - d[t] * DT / ETA
E_end = SOC[-1]
print(f"修正前 E(24:00) = {E_end:,.1f} kWh（目标 6000）")
chg_seg = [t for t in range(T) if b[t] + c[t] > 1e-9]
rem = E_end - E0                        # 储能侧 kWh，正=超充

if rem > 1e-9:                          # 超充：从最贵的充电段扣，受其后节点距下限余量约束
    for t in sorted(chg_seg, key=lambda t: -price[t]):
        if rem <= 1e-9:
            break
        slack = min(SOC[t + 1:]) - EMIN             # 扣减会整体压低 t 之后所有节点
        p = min(b[t], slack / (ETA * DT), rem / (ETA * DT))
        if p > 1e-9:
            b[t] -= p; SOC[t + 1:] -= ETA * p * DT; rem -= ETA * p * DT
        if rem <= 1e-9:
            break
        p = min(c[t], (min(SOC[t + 1:]) - EMIN) / (ETA * DT), rem / (ETA * DT))
        if p > 1e-9:
            c[t] -= p; q_cur[t] += p; SOC[t + 1:] -= ETA * p * DT; rem -= ETA * p * DT
elif rem < -1e-9:                       # 欠充：从最便宜的段补，受其后节点距上限余量约束
    rem = -rem
    cand = sorted(chg_seg, key=lambda t: price[t])
    cand += [t for t in np.argsort(price) if t not in set(cand)]
    for t in cand:
        if rem <= 1e-9:
            break
        room = EMAX - max(SOC[t + 1:])
        p = min(PMAX - c[t] - b[t], room / (ETA * DT), rem / (ETA * DT))
        if p > 1e-9:
            b[t] += p; SOC[t + 1:] += ETA * p * DT; rem -= ETA * p * DT
E_end = E0 + rem
print(f"修正后 E(24:00) = {E_end:,.2f} kWh")

# 修正后可行性复核
assert abs(SOC[-1] - E0) < 1e-6, f"闭环失败: {SOC[-1]}"
assert SOC.min() >= EMIN - 1e-6 and SOC.max() <= EMAX + 1e-6, f"SOC 越界 [{SOC.min():.0f},{SOC.max():.0f}]"
assert np.all(a + g + d >= load - 1e-6), "存在供电缺口"
print(f"修正后 SOC 范围 [{SOC.min():.0f}, {SOC.max():.0f}]，供电与闭环校验通过")

cost = float(np.sum(price * (a + b) * DT))
purchase = float(np.sum((a + b) * DT))
print(f"\n=== 对照结果 ===")
print(f"规则式贪心: 购电费 {cost:,.2f} 元, 全天购电 {purchase:,.1f} kWh")
lp_cost = 35126.85
print(f"LP 最优   : 购电费 {lp_cost:,.2f} 元")
print(f"差距      : +{cost - lp_cost:,.2f} 元 (+{(cost/lp_cost-1)*100:.2f}%)")
np.savez(OUTDIR / "_greedy.npz", a=a, b=b, c=c, d=d, g=g, q=q_cur,
         cost=cost, purchase=purchase)
print(f"轨迹存档 -> {OUTDIR/'_greedy.npz'}")
