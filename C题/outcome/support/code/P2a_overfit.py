# -*- coding: utf-8 -*-
"""
v5过拟合诊断：时序切分下LightGBM训练/测试误差比
其他模型无参数训练，不存在过拟合
对比：v1无正则过拟合比4.83
"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
exec(open(r'D:\CUMCM2026Problems\C题\P2a_forecast.py', encoding='utf-8').read().split('# ---------------- 9.')[0])

dates, load_mat, pv_mat = load_data()
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
N = len(dates)
split = int(N * 0.7)  # 255
print(f'时序切分：训练1-{split}天({pd.Timestamp(dates[0]).date()}~{pd.Timestamp(dates[split-1]).date()})')
print(f'          测试{split+1}-{N}天({pd.Timestamp(dates[split]).date()}~{pd.Timestamp(dates[-1]).date()})')

def lgb_overfit(mat, name):
    # 在训练集末尾(split-1)训练LGB
    X_tr, y_tr, X_va, y_va = build_train_matrix(dates, mat, split-1, date_to_idx)
    model = train_lgb(X_tr, y_tr, X_va, y_va)
    # 训练期误差：训练集最后30天（滚动预测，每7天重训）
    train_preds = []; train_true = []
    m = None
    for i in range(split-30, split):
        if (i - (split-30)) % 7 == 0 or m is None:
            Xt, yt, Xv, yv = build_train_matrix(dates, mat, i-1, date_to_idx)
            if Xt is not None and len(Xt) > 100:
                m = train_lgb(Xt, yt, Xv, yv)
        if m is not None:
            df = build_pred_matrix(dates[i], date_to_idx, mat)
            p = lgb_predict(m, df)
            train_preds.append(p); train_true.append(mat[i])
    train_mae = mean_absolute_error(np.array(train_true), np.array(train_preds))
    # 测试期误差：用训练集末尾的模型，不更新，直接预测测试集
    test_preds = []; test_true = []
    for i in range(split, N):
        df = build_pred_matrix(dates[i], date_to_idx, mat)
        p = lgb_predict(model, df)
        test_preds.append(p); test_true.append(mat[i])
    test_mae = mean_absolute_error(np.array(test_true), np.array(test_preds))
    ratio = test_mae / train_mae
    print(f'{name}: 训练MAE(最后30天滚动)={train_mae:.1f}, 测试MAE(后30%不更新)={test_mae:.1f}, 过拟合比={ratio:.2f}')
    return train_mae, test_mae, ratio

print('\n===== LightGBM过拟合诊断（强正则+早停）=====')
l_tr, l_te, l_r = lgb_overfit(load_mat, '小区负载')
p_tr, p_te, p_r = lgb_overfit(pv_mat, '光伏发电')

print('\n===== 对比v1无正则 =====')
print(f'v1无正则(历史): 负载过拟合比=4.83')
print(f'v5强正则: 负载={l_r:.2f}, 光伏={p_r:.2f}')
print(f'强正则降低过拟合比 {(4.83-l_r)/4.83*100:.0f}%（负载）')

rows = [
    {'对象': '小区负载', '模型': 'LightGBM(v5强正则)', '训练MAE': round(l_tr,1), '测试MAE': round(l_te,1), '过拟合比': round(l_r,2)},
    {'对象': '光伏发电', '模型': 'LightGBM(v5强正则)', '训练MAE': round(p_tr,1), '测试MAE': round(p_te,1), '过拟合比': round(p_r,2)},
    {'对象': '小区负载', '模型': 'LightGBM(v1无正则,历史)', '训练MAE': None, '测试MAE': None, '过拟合比': 4.83},
]
pd.DataFrame(rows).to_excel(r'D:\CUMCM2026Problems\C题\预测结果\过拟合诊断_v5.xlsx', index=False)
print('\n已保存到 过拟合诊断_v5.xlsx')
