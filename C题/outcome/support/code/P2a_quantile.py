# -*- coding: utf-8 -*-
"""
分位数预测模块（v5.1新增）
方法：基于EGD集成点预测的残差，滚动经验分位数（按ti分别计算）
论文依据：Wang 2019 CQRA (IEEE TSG)；Gneiting 2023 分位数评估；报童模型经典库存理论
报童最优分位：Cu/(Cu+Co) = 5/(5+1) = 0.833（低估代价5倍电价，高估代价1倍电价）
输出：P10/P50/P83.3/P90分位数预测 + 区间覆盖率验证
"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

exec(open(r'D:\CUMCM2026Problems\C题\P2a_forecast.py', encoding='utf-8').read().split('# ---------------- 9.')[0])

OUT = r'D:\CUMCM2026Problems\C题\预测结果'
QUANTILES = [0.10, 0.50, 0.833, 0.90]  # P10, P50, 报童最优P83.3, P90
Q_NAMES = ['P10', 'P50', 'P83.3(报童最优)', 'P90']
ROLLING_WINDOW = 30  # 滚动残差窗口

def ti_to_hour(ti):
    return (ti + 1) * 10.0 / 60.0

def is_pv_night(ti_arr):
    hrs = ti_to_hour(ti_arr)
    return (hrs <= 6.0) | (hrs >= 20.0)

def rolling_quantile_forecast(point_pred, true_mat, dates, pred_days_idx, is_pv=False):
    """
    滚动分位数预测：对每个预测日，用过去ROLLING_WINDOW天的残差（按ti）计算经验分位数
    point_pred: (n_pred, 144) EGD集成点预测
    true_mat: (365, 144) 完整真实值
    pred_days_idx: 预测日在true_mat中的索引列表
    返回: (n_pred, 144, n_quantiles) 分位数预测
    """
    n_pred = len(pred_days_idx)
    n_q = len(QUANTILES)
    q_pred = np.zeros((n_pred, N_SLOT, n_q))
    night_mask = is_pv_night(np.arange(N_SLOT)) if is_pv else np.zeros(N_SLOT, dtype=bool)

    for k, i_day in enumerate(pred_days_idx):
        # 过去ROLLING_WINDOW天的残差（真实-预测）
        lo = max(0, i_day - ROLLING_WINDOW)
        # 点预测只对预测期有，残差需要用已有的点预测
        # 对于预测日之前的日子，用真实值和"如果当时用EGD预测"的值
        # 简化：用point_pred中对应日的预测（如果在预测期内），否则用persistence近似
        residuals = []
        for j in range(lo, i_day):
            # 找j在pred_days_idx中的位置
            if j in pred_days_idx:
                jk = pred_days_idx.index(j)
                pred_j = point_pred[jk]
            else:
                pred_j = true_mat[j-1] if j > 0 else true_mat[j]  # 冷启动用persistence
            residuals.append(true_mat[j] - pred_j)
        residuals = np.array(residuals)  # (n_days, 144)

        # 按ti计算经验分位数
        for ti in range(N_SLOT):
            res_ti = residuals[:, ti]
            for qi, q in enumerate(QUANTILES):
                q_val = np.percentile(res_ti, q * 100)
                q_pred[k, ti, qi] = point_pred[k, ti] + q_val

        # 光伏夜间置0
        if is_pv:
            q_pred[k, night_mask, :] = 0.0
        # 分位数非负（功率不能为负）
        q_pred[k] = np.clip(q_pred[k], 0, None)

        if (k+1) % 50 == 0:
            print(f'  分位数预测第{k+1}/{n_pred}天')

    return q_pred

def verify_coverage(q_pred, true_mat, pred_days_idx, is_pv=False):
    """验证P10-P90区间覆盖率（应≈80%）和P83.3超限率（应≈16.7%）"""
    n_pred = len(pred_days_idx)
    night_mask = is_pv_night(np.arange(N_SLOT)) if is_pv else np.zeros(N_SLOT, dtype=bool)
    cover_80 = []  # P10-P90
    exceed_833 = []  # 真实值 > P83.3的比例
    for k, i_day in enumerate(pred_days_idx):
        t = true_mat[i_day]
        p10 = q_pred[k, :, 0]
        p90 = q_pred[k, :, 3]
        p833 = q_pred[k, :, 2]
        if is_pv:
            valid = ~night_mask
            t = t[valid]; p10 = p10[valid]; p90 = p90[valid]; p833 = p833[valid]
        cover_80.append(np.mean((t >= p10) & (t <= p90)))
        exceed_833.append(np.mean(t > p833))
    return np.mean(cover_80), np.mean(exceed_833)

def save_quantile_excel(q_pred, dates, pred_days, name_suffix):
    """保存分位数预测Excel：每个分位一个sheet，格式与点预测一致（日期+144时点）"""
    time_cols = [f'{ti_to_hour(ti):.2f}h' for ti in range(N_SLOT)]
    with pd.ExcelWriter(OUT + f'\\{name_suffix}.xlsx') as writer:
        for qi, qname in enumerate(Q_NAMES):
            df = pd.DataFrame(q_pred[:, :, qi], columns=time_cols)
            df.insert(0, '日期', [pd.Timestamp(d).strftime('%Y-%m-%d') for d in pred_days])
            df.to_excel(writer, sheet_name=qname, index=False)
    print(f'  已保存 {name_suffix}.xlsx（{len(Q_NAMES)}个分位sheet）')

# ==================== 主流程 ====================
print('加载数据与v5.1点预测...')
dates, load_mat, pv_mat = load_data()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
pred_days = pd.date_range('2025-02-01', '2025-12-31', freq='D')
pred_idx = [date_to_idx[pd.Timestamp(d)] for d in pred_days]

load_point = np.load(OUT + r'\load_pred_ens_v5.npy')  # (334,144)
pv_point = np.load(OUT + r'\pv_pred_ens_v5.npy')
print(f'负载点预测: {load_point.shape}, 光伏: {pv_point.shape}')

print('\n===== 负载分位数预测 =====')
load_q = rolling_quantile_forecast(load_point, load_mat, dates, pred_idx, is_pv=False)
np.save(OUT + r'\load_quantiles_v5.npy', load_q)
print(f'  已保存 load_quantiles_v5.npy {load_q.shape}')

print('\n===== 光伏分位数预测 =====')
pv_q = rolling_quantile_forecast(pv_point, pv_mat, dates, pred_idx, is_pv=True)
np.save(OUT + r'\pv_quantiles_v5.npy', pv_q)
print(f'  已保存 pv_quantiles_v5.npy {pv_q.shape}')

print('\n===== 覆盖率验证 =====')
l_cov, l_exc = verify_coverage(load_q, load_mat, pred_idx, is_pv=False)
p_cov, p_exc = verify_coverage(pv_q, pv_mat, pred_idx, is_pv=True)
print(f'负载: P10-P90区间覆盖率={l_cov*100:.1f}% (目标≈80%), 真实值>P83.3比例={l_exc*100:.1f}% (目标≈16.7%)')
print(f'光伏: P10-P90区间覆盖率={p_cov*100:.1f}% (目标≈80%), 真实值>P83.3比例={p_exc*100:.1f}% (目标≈16.7%)')

print('\n===== 报童模型最优购电量（P83.3分位）=====')
l_p833 = load_q[:, :, 2]  # (334,144)
p_p833 = pv_q[:, :, 2]
# 净负载 = 负载 - 光伏，购电量 = max(净负载, 0)
net_load_p833 = np.clip(l_p833 - p_p833, 0, None)
print(f'P83.3分位下日均购电量: {net_load_p833.sum(axis=1).mean():.0f} kWh')
print(f'点预测下日均购电量: {np.clip(load_point - pv_point, 0, None).sum(axis=1).mean():.0f} kWh')
print(f'报童策略多购: {(net_load_p833.sum(axis=1).mean() - np.clip(load_point - pv_point, 0, None).sum(axis=1).mean()):.0f} kWh/天 (保守高估以避免5倍电价缺口)')

# 保存Excel
print('\n===== 保存分位数Excel =====')
save_quantile_excel(load_q, dates, pred_days, '小区负载分位数预测_v5')
save_quantile_excel(pv_q, dates, pred_days, '光伏功率分位数预测_v5')

# 保存覆盖率报告
cov_df = pd.DataFrame([
    {'对象': '小区负载', 'P10-P90覆盖率%': round(l_cov*100,1), '真实值>P83.3比例%': round(l_exc*100,1)},
    {'对象': '光伏发电', 'P10-P90覆盖率%': round(p_cov*100,1), '真实值>P83.3比例%': round(p_exc*100,1)},
])
cov_df.to_excel(OUT + r'\分位数覆盖率验证_v5.xlsx', index=False)
print('\n分位数预测全部完成。')
