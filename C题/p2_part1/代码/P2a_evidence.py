# -*- coding: utf-8 -*-
"""计算论文模型池各"因素"的印证统计量，全部来自附件2真实数据"""
import numpy as np
import pandas as pd

PATH = r'D:\CUMCM2026Problems\C题\附件\附件2.xlsx'
xl = pd.ExcelFile(PATH)
load = pd.read_excel(xl, '小区负载'); pv = pd.read_excel(xl, '光伏发电实际功率')
load = load.drop(columns=[load.columns[0]]).values.astype(float)
pv = pv.drop(columns=[pv.columns[0]]).values.astype(float)
dates = pd.date_range('2025-01-01', periods=365, freq='D')
N, T = load.shape
print(f'数据: {N}天 x {T}时点')

def acf(x, lag):
    a, b = x[lag:], x[:-lag]
    return np.corrcoef(a.ravel(), b.ravel())[0,1]

print('\n========== 小区负载 ==========')
print(f'lag1={acf(load,1):.3f}  lag7={acf(load,7):.3f}')
dow = pd.Series(dates).dt.dayofweek.values
# 周内日均
print('\n周内各天日均负荷:')
names = ['周一','周二','周三','周四','周五','周六','周日']
wd = [load[dow==d].mean() for d in range(7)]
for d in range(7):
    print(f'  {names[d]}: {wd[d]:.0f}', end='')
print()
work = np.mean([wd[i] for i in [0,1,2,3,6]]); wknd = np.mean([wd[4],wd[5]])
print(f'周一-四+周日均值={work:.0f}, 周五六均值={wknd:.0f}, 降幅={(1-wknd/work)*100:.1f}%')

# 月度日均 + 峰谷比 + 峰值时点
month = pd.Series(dates).dt.month.values
hrs = (np.arange(T)+1)*10/60
print('\n月度: 日均 / 日内峰谷比 / 峰值时点')
mmean=[]
for m in range(1,13):
    mm = load[month==m]; mmean.append(mm.mean())
    daily_ratio = mm.max(axis=1)/(mm.min(axis=1)+1e-9)
    peak_t = hrs[mm.mean(axis=0).argmax()]
    print(f'  {m:2d}月: 日均{mm.mean():6.0f}  峰谷比{daily_ratio.mean():.2f}  峰值{peak_t:.1f}h')
mmean=np.array(mmean)
print(f'月度日均 max/min = {mmean.max()/mmean.min():.2f} (最高{mmean.max():.0f}/最低{mmean.min():.0f})')

# 7月 vs 11月 平均日内曲线相对差异
jul = load[month==7].mean(axis=0); nov = load[month==11].mean(axis=0)
diff = np.abs(jul-nov).mean()/nov.mean()
print(f'\n7月vs11月平均日内曲线相对差异 = {diff*100:.1f}%')

# kNN印证：同dow随机两日距离 vs 最近3邻距离（归一化欧氏，查询=昨日曲线）
def norm_curve(c):
    s = c.std()+1e-9
    return (c-c.mean())/s
Lnorm = np.array([norm_curve(load[i]) for i in range(N)])
rand_d=[]; nn_d=[]
for i in range(31, N):
    q = Lnorm[i-1]; target_dow = dow[i]
    cands = [j for j in range(max(0,i-60), i-1) if dow[j+1]==target_dow and j+1<i]
    if len(cands)<4: continue
    ds = np.array([np.sqrt(((q-Lnorm[j])**2).mean()) for j in cands])
    rand_d.append(ds.mean()); nn_d.append(np.sort(ds)[:3].mean())
print(f'\nkNN印证(负载,限同dow): 同dow候选平均距离={np.mean(rand_d):.3f}, 最近3邻距离={np.mean(nn_d):.3f}, 近邻比平均近{(1-np.mean(nn_d)/np.mean(rand_d))*100:.1f}%')

print('\n========== 光伏 ==========')
print(f'lag1={acf(pv,1):.3f}  lag7={acf(pv,7):.3f}')
# 非零比例、非零时段
nz = (pv>1).mean()
daymask = (hrs>6)&(hrs<20)
print(f'非零比例={nz*100:.1f}%, 白天(6-20h)均值={pv[:,daymask].mean():.0f}')
# 峰值时点
peak_hours = [hrs[pv[i].argmax()] for i in range(N) if pv[i].max()>100]
print(f'峰值时点: 中位数={np.median(peak_hours):.1f}h, 落在11.5-12.5h占比={np.mean([abs(h-12)<=0.5 for h in peak_hours])*100:.1f}%')
# 月度
print('月度日均:')
pm=[]
for m in range(1,13):
    mm=pv[month==m]; pm.append(mm.mean())
    print(f'  {m:2d}月: {mm.mean():.0f}', end='')
pm=np.array(pm)
print(f'\n月度 max/min={pm.max()/pm.min():.2f} (最高{pm.max():.0f} {pm.argmax()+1}月/最低{pm.min():.0f} {pm.argmin()+1}月)')

# FPCA印证：白天曲线SVD主成分方差解释
D = pv[:, daymask]
Dc = D - D.mean(axis=0)
U,S,Vt = np.linalg.svd(Dc, full_matrices=False)
ev = S**2; evr = ev/ev.sum()
print(f'\nFPCA印证(光伏白天曲线SVD)累计方差解释: PC1={evr[0]*100:.1f}%  前2={evr[:2].sum()*100:.1f}%  前3={evr[:3].sum()*100:.1f}%')

# 负载也做FPCA主成分（对比，说明负载低秩性弱于光伏）
Dl = load - load.mean(axis=0)
_,Sl,_ = np.linalg.svd(Dl, full_matrices=False)
evl = Sl**2; evlr=evl/evl.sum()
print(f'负载曲线SVD: PC1={evlr[0]*100:.1f}%  前2={evlr[:2].sum()*100:.1f}%  前3={evlr[:3].sum()*100:.1f}% (对比:负载形态更多样)')

# kNN印证（光伏不限dow）
Pnorm = np.array([norm_curve(pv[i]) for i in range(N)])
rand_p=[]; nn_p=[]
for i in range(31,N):
    q=Pnorm[i-1]
    cands=list(range(max(0,i-60),i-1))
    ds=np.array([np.sqrt(((q-Pnorm[j])**2).mean()) for j in cands])
    rand_p.append(ds.mean()); nn_p.append(np.sort(ds)[:3].mean())
print(f'kNN印证(光伏,不限dow): 候选平均距离={np.mean(rand_p):.3f}, 最近3邻={np.mean(nn_p):.3f}, 近邻近{(1-np.mean(nn_p)/np.mean(rand_p))*100:.1f}%')

# 光伏日总量变异（天气噪声）：相邻周日总量的波动
daily_total = pv.sum(axis=1)
print(f'\n光伏日总量: 总体变异系数={daily_total.std()/daily_total.mean()*100:.1f}%')
# 同月份内日总量变异（剔除季节后的天气波动）
within=[]
for m in range(1,13):
    dt=daily_total[month==m]
    if len(dt)>3: within.append(dt.std()/dt.mean())
print(f'月内日总量变异系数(天气波动)平均={np.mean(within)*100:.1f}%')
print('\n完成')
