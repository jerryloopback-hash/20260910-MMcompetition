# -*- coding: utf-8 -*-
"""
问题1：微网计划购电策略 —— 线性规划求解
模型细节见同目录 q1_model.md
求解器：scipy.optimize.linprog(method='highs')（HiGHS）
"""
import numpy as np
import openpyxl
from scipy.optimize import linprog

BASE = r"d:/devWorkshopForCC/20260910-MMcompetition/C题"

# ---------- 1. 读取附件1 ----------
wb = openpyxl.load_workbook(f"{BASE}/附件/附件1.xlsx", data_only=True)
ws = wb["Sheet1"]
rows = list(ws.iter_rows(min_row=2, values_only=True))          # 144 x (time, price, load, pv)
assert len(rows) == 144
price = np.array([r[1] for r in rows], dtype=float)             # 元/kWh
load  = np.array([r[2] for r in rows], dtype=float)             # kW
pv    = np.array([r[3] for r in rows], dtype=float)             # kW（预测）
DT = 1 / 6.0                                                    # h，时段长
ETA = 0.9                                                       # 充/放效率
T  = 144
EMIN, EMAX = 1200.0, 10800.0
E0 = E0_END = 6000.0                                            # 0:00 与 24:00 电量
PMAX = 5000.0                                                   # 充放功率上限 kW

# ---------- 2. 构造 LP ----------
# 变量（每时段6个）：a 购电供负载, b 购电充电, g 光伏供负载, c 光伏充电, d 放电, q 弃光
# 排列：t 时段内固定顺序 a,b,g,c,d,q
NV = T * 6
idx = { 'a': np.arange(0, NV, 6), 'b': np.arange(1, NV, 6),
        'g': np.arange(2, NV, 6), 'c': np.arange(3, NV, 6),
        'd': np.arange(4, NV, 6), 'q': np.arange(5, NV, 6) }

def cu(v, upto):                 # upto: 前 upto 个时段的累积（1-based 语义, upto<=143）
    return (v[:(upto)]).sum()    # 注：v 是 6 步间隔的索引

# 目标：min sum price*(a+b)*DT
c_obj = np.zeros(NV)
c_obj[idx['a']] = price * DT
c_obj[idx['b']] = price * DT

A_eq, b_eq = [], []

# 负载平衡 / 光伏平衡
for t in range(T):
    rowL = np.zeros(NV); rowL[idx['a'][t]] = 1; rowL[idx['g'][t]] = 1; rowL[idx['d'][t]] = 1
    rowP = np.zeros(NV); rowP[idx['g'][t]] = 1; rowP[idx['c'][t]] = 1; rowP[idx['q'][t]] = 1
    A_eq.append(rowL); b_eq.append(load[t])
    A_eq.append(rowP); b_eq.append(pv[t])

# 终值约束 E143 = 6000  (对前 143 各时段：0.9*(b+c) - (1/0.9)*d 的累积 = 0)
rowE = np.zeros(NV)
for t in range(143):
    rowE[idx['b'][t]] +=  ETA
    rowE[idx['c'][t]] +=  ETA
    rowE[idx['d'][t]] += -1.0 / ETA
A_eq.append(rowE); b_eq.append(0.0)

# 不等式：SOC 上下限（化为累积形式），充/放功率上限
A_ub, b_ub = [], []
for t in range(1, T + 1):                       # 时段1..144 的累积 SOC 界限
    rowU = np.zeros(NV); rowLw = np.zeros(NV)
    for k in range(t):
        rowU[idx['b'][k]] +=  ETA * DT
        rowU[idx['c'][k]] +=  ETA * DT
        rowU[idx['d'][k]] += -DT / ETA
        rowLw[idx['b'][k]] += -ETA * DT
        rowLw[idx['c'][k]] += -ETA * DT
        rowLw[idx['d'][k]] +=  DT / ETA
    A_ub.append(rowU); b_ub.append(EMAX - E0)   # E_t <= 10800
    A_ub.append(rowLw); b_ub.append(E0 - EMIN)  # E_t >= 1200
for t in range(T):
    rowC = np.zeros(NV); rowC[idx['b'][t]] = 1; rowC[idx['c'][t]] = 1
    rowD = np.zeros(NV); rowD[idx['d'][t]] = 1
    A_ub.append(rowC); b_ub.append(PMAX)
    A_ub.append(rowD); b_ub.append(PMAX)

res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
              A_eq=np.array(A_eq), b_eq=np.array(b_eq),
              bounds=[(0, None)] * NV, method="highs")
assert res.success, f"求解失败: {res.message}"
x = res.x

# ---------- 3. 结果后处理 ----------
a = x[idx['a']]; b = x[idx['b']]; g = x[idx['g']]
c = x[idx['c']]; d = x[idx['d']]; q = x[idx['q']]

purchase_kwh = (a + b) * DT            # 各时段购电量 kWh
SOC = np.zeros(T + 1); SOC[0] = E0
for t in range(T):
    SOC[t+1] = SOC[t] + ETA * (b[t] + c[t]) * DT - d[t] * DT / ETA

E_day = float(res.fun)
print("=== 求解成功 ===")
print(f"最优购电费 = {E_day:,.2f} 元")
print(f"全天购电量 = {purchase_kwh.sum():,.1f} kWh  (143段内 {purchase_kwh[:143].sum():,.1f})")
print(f"弃光总量   = {q.sum()*DT:,.2f} kWh（时段数 {int((q>1e-6).sum())}）")
print(f"同段边充边放检查: {(np.minimum(b+c, d)>1e-6).sum()} 个时段")
print(f"SOC 范围 [{SOC.min():.1f}, {SOC.max():.1f}]，边界 E0={SOC[0]:.1f} E143={SOC[143]:.1f}")
print(f"充电功率超限检查: {(b+c).max():.3f}，放电功率超限: {d.max():.3f}")

# 能量守恒校验（前143段，E143=E0）
Eload = load[:143].sum()*DT; Epv = pv[:143].sum()*DT
Eq = q[:143].sum()*DT
Lch = (1-ETA)*(b[:143]+c[:143]).sum()*DT
Ldi = (1/ETA-1)*d[:143].sum()*DT
Ebuy = purchase_kwh[:143].sum()
lhs = Ebuy + Epv; rhs = Eload + Eq + Lch + Ldi
print(f"\n[守恒校验 前143段] 购电{ lhs:,.2f} = 负载{Eload:,.0f}+弃光{Eq:.2f}+充损{Lch:,.1f}+放损{Ldi:,.1f}  →  {rhs:,.2f}，差={lhs-rhs:.2e}")
print(f"光伏自消纳率 = {(Epv-sum(c[:143]*DT)-Eq)/Epv*100:.1f}%（直供负载）/ 总利用率={ (Epv-Eq)/Epv*100:.1f}%")

# ---------- 4. 写入 result1.xlsx （以模板为基础填充） ----------
tpl = openpyxl.load_workbook(f"{BASE}/附件/附件5/result1.xlsx")
ws1 = tpl["计划购电量"]; ws2 = tpl["充放电量"]
for t in range(T):
    ws1.cell(row=2 + t, column=2).value = round(float(purchase_kwh[t]), 2)
# 6 个 4h 块：块 k <- 时段索引 [24k, 24k+24)
for k in range(6):
    sl = slice(24 * k, 24 * (k + 1))
    ws2.cell(row=2 + k, column=2).value = round(float((b[sl] + c[sl]).sum() * DT), 2)   # 充电量
    ws2.cell(row=2 + k, column=3).value = round(float(d[sl].sum() * DT), 2)             # 放电量
ws2.cell(row=2, column=5).value = 6000     # 0:00
ws2.cell(row=3, column=5).value = 6000     # 24:00
out = f"{BASE}/q1/result1.xlsx"
tpl.save(out)
print(f"\n已保存 -> {out}")

# 存档关键数值供讨论
np.save("D:/devWorkshopForCC/20260910-MMcompetition/C题/q1/_traj.npy",
        np.vstack([a, b, g, c, d, q, (a+b), SOC[1:]]))