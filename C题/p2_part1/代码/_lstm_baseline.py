# -*- coding: utf-8 -*-
"""
LSTM baseline：单模型日前预测，作为消融对照（不进EGD池）
输入=历史7天144维曲线，输出=下一天144维曲线
扩展窗口训练，每7天重训，滚动origin预测
"""
import sys
sys.path.insert(0, r'D:\CUMCM2026Problems\C题')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time, os

# 复用v5的数据加载和时点映射
src = open(r'D:\CUMCM2026Problems\C题\预测_v5_扩展模型池.py', encoding='utf-8').read()
exec(src.split('# ---------------- 3.')[0])

N_SLOT = 144
SEQ_LEN = 7
RETRAIN_EVERY = 7
HIDDEN = 32
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 32
LR = 0.001
TRAIN_WINDOW = 90  # 固定训练窗口（天），加速CPU训练
DEVICE = 'cpu'

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=144, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 144)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def make_dataset(mat, end_idx, seq_len=7, val_days=7, train_window=90):
    """固定窗口构造(前seq_len天→下一天)样本"""
    start = max(seq_len, end_idx - train_window)
    X, y = [], []
    for i in range(start, end_idx + 1):
        X.append(mat[i-seq_len:i])
        y.append(mat[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if len(X) <= val_days + 5:
        return X, y, None, None
    X_tr, y_tr = X[:-val_days], y[:-val_days]
    X_va, y_va = X[-val_days:], y[-val_days:]
    return X_tr, y_tr, X_va, y_va

def get_norm_stats(y_tr):
    mu = float(y_tr.mean())
    sd = float(y_tr.std()) + 1e-6
    return mu, sd

def train_one(X_tr, y_tr, X_va, y_va, mu, sd):
    model = LSTMForecaster().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    Xt = torch.from_numpy((X_tr - mu) / sd).to(DEVICE)
    yt = torch.from_numpy((y_tr - mu) / sd).to(DEVICE)
    Xva = torch.from_numpy((X_va - mu) / sd).to(DEVICE) if X_va is not None else None
    yva = y_va
    best_val = float('inf'); best_state = None; bad = 0
    n = len(Xt)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = crit(pred, yt[idx])
            loss.backward()
            opt.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                vp = model(Xva).cpu().numpy() * sd + mu
                va_loss = mean_absolute_error(yva.reshape(-1), vp.reshape(-1))
            if va_loss < best_val:
                best_val = va_loss; best_state = {k:v.clone() for k,v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def rolling_lstm(mat, dates, start, end):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    pred_days = pd.date_range(start, end, freq='D')
    n_pred = len(pred_days)
    preds = np.full((n_pred, N_SLOT), np.nan)
    model = None; cur_mu = 0.0; cur_sd = 1.0
    t0 = time.time()
    for k, day in enumerate(pred_days):
        i_day = date_to_idx[pd.Timestamp(day)]
        train_end = i_day - 1
        need_retrain = (k % RETRAIN_EVERY == 0) or (model is None)
        if need_retrain and train_end >= 35:
            X_tr, y_tr, X_va, y_va = make_dataset(mat, train_end, SEQ_LEN, train_window=TRAIN_WINDOW)
            if len(X_tr) > 10:
                cur_mu, cur_sd = get_norm_stats(y_tr)
                model = train_one(X_tr, y_tr, X_va, y_va, cur_mu, cur_sd)
                model.eval()
        if model is not None and i_day >= SEQ_LEN:
            seq = (mat[i_day-SEQ_LEN:i_day].astype(np.float32) - cur_mu) / cur_sd
            with torch.no_grad():
                inp = torch.from_numpy(seq[None]).to(DEVICE)
                p = model(inp).cpu().numpy()[0] * cur_sd + cur_mu
            preds[k] = np.clip(p, 0, None)
        else:
            preds[k] = mat[i_day-1]
        if (k+1) % 20 == 0:
            print(f'[{time.time()-t0:6.1f}s] {pd.Timestamp(day).date()} 第{k+1}/{n_pred}天 LSTM MAE={mean_absolute_error(mat[i_day], preds[k]):.1f}')
    print(f'LSTM滚动完成，耗时{time.time()-t0:.1f}s')
    return preds, pred_days

if __name__ == '__main__':
    dates, load_mat, pv_mat = load_data()
    OUT = r'D:\CUMCM2026Problems\C题\预测结果'
    print('=== 负载 LSTM baseline ===')
    lp, pdays = rolling_lstm(load_mat, dates, '2025-02-01', '2025-12-31')
    np.save(os.path.join(OUT, 'lstm_load_pred.npy'), lp)
    print('=== 光伏 LSTM baseline ===')
    pp, _ = rolling_lstm(pv_mat, dates, '2025-02-01', '2025-12-31')
    night = (ti_to_hour(np.arange(N_SLOT)) <= 6.0) | (ti_to_hour(np.arange(N_SLOT)) >= 20.0)
    pp[:, night] = 0.0
    np.save(os.path.join(OUT, 'lstm_pv_pred.npy'), pp)
    # 全期误差
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    lt = np.vstack([load_mat[date_to_idx[pd.Timestamp(d)]] for d in pdays])
    pt = np.vstack([pv_mat[date_to_idx[pd.Timestamp(d)]] for d in pdays])
    print(f'\nLSTM负载全期 MAE={mean_absolute_error(lt, lp):.1f} RMSE={np.sqrt(mean_squared_error(lt, lp)):.1f}')
    print(f'LSTM光伏全期 MAE={mean_absolute_error(pt, pp):.1f} RMSE={np.sqrt(mean_squared_error(pt, pp)):.1f}')
    print('LSTM baseline完成')
