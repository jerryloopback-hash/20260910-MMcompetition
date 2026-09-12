# -*- coding: utf-8 -*-
"""
P3-M3 误差分布与情景生成:终偏校正 + 混合中心残差路径库 + 整日自助抽样器 + 形态下沉
================================================
依据 p3_plan.md §3.3–3.5 / D4–D5 / 里程碑 M3; M2 审计接口指令(m2_model.md §8):
  树的中心取 C_s(混合), 条件分布必须对 A−C_s 估, 不得用 M1 单源残差。
坐标系: D4 误差链的混合坐标实现——路径状态 (ε0, G1, G2, G3):
  ε0 = A − C_0 (0:00 混合中心残差, 24 维整日);
  G_s = C_s − C_{s−1} (混合中心修正量, 附件3 修正量 R_s 的混合坐标实现, 绑定段 18/12/6 维);
  恒等式 ε_s = ε0 − Σ_{k≤s} G_k 逐日逐点成立(脚本 assert)。
终偏校正层(本版新增, 探针检验催生):
  诊断实测两源残差均有月尺度漂移偏差(δ=A−EGD 白天月均 +143(2月)→−184(6月)→−175(11月);
  M1 去偏后的 η0 仍有 ±150 月度漂移), 融合中心继承 ±100–170 kW 慢漂移
  (M2 审计的 −4.1 kW 是全年白天 pooled 均值, 掩盖了该结构)。
  处理: 对混合残差按(会话×目标小时)做尾随去偏, 候选窗 {7,10,14,28} 逐会话按样本外 MAE 择优
  (实测采纳 14/14/10/7; 预注册固定 W=14 与择优几乎等价, 最大差 0.46 kW, 故选窗不承重); 负载侧同法
  检验但全部窗口均劣化(28 样本均值劣于偏差信号), 不采纳。
二级去偏检验: 对一级残差再作 28 天均值的二级校正, 样本外 MAE 恶化(123.92→127.24),
  即池均值漂移不可预测 → 情景侧按"再中心化"丢弃该均值(而非并入中心),
  保证"情景中心 ≡ 采纳中心"(校验清单第 1 条)严格成立; 丢弃幅度如实报告。
生成机制: PLAN §3.3 取**整日路径自助**(弃低秩因子模型——高斯尾低估 5λ 联合极端):
  一个历史日 = 一条完整路径(光伏混合残差 60 维 + 负载残差 144 维, 整日绑定抽取),
  非参数保留跨小时/跨阶段/跨源相关与联合尾部。落盘供体库(334×60 + 334×144),
  M4/M5 按日自助抽取(池=[e−W,e−1], W=28; N=200–500 为下游参数)。
下沉: PLAN §3.4 历史形态曲线(尾随 91 天, 实际小时均值<20kW 取平 1), M5 组装用。
相位定案(M1 遗留): 连续移窗曲线定 ±10 分钟读法。
用法: python m3_scenarios.py [smoke]   产出: p3/m3_out/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent          # C题/
M1 = Path(__file__).resolve().parent / "m1_out"
M2 = Path(__file__).resolve().parent / "m2_out"
PRED = BASE / "p2_part1" / "预测结果"
OUT = Path(__file__).resolve().parent / "m3_out"

NS, NL = 4, 24
H0 = np.array([0, 6, 12, 18])
I0 = 31                                                # 2025-02-01
W_POOL = 28                                            # 情景供体池宽(预注册)
W_CAND = (7, 10, 14, 28)                               # 终偏校正候选窗
N_PATHS = 300                                          # 探针日抽样数(PLAN 200–500)
W_SHAPE = 91                                           # 形态曲线尾随窗
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"

# ---------- 1. 输入 ----------
A_nat = np.load(M1 / "A_nat.npy")                      # (365,24) 整点实际(终点读法)
F_corr = np.load(M1 / "F_corr.npy")                    # (365,4,24)
C_ses = np.load(M2 / "C_ses.npy")                      # (334,4,24) 行e↔实际日31+e
assert C_ses.shape == (334, NS, NL)
P = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:145].astype(float).values
ActL = pd.read_excel(BASE / "附件" / "附件2.xlsx", sheet_name="小区负载").iloc[:, 1:145].astype(float).values
Lpred = np.load(PRED / "load_pred_ens_v5.npy")         # (334,144)
assert P.shape == (365, 144) and ActL.shape == (365, 144) and Lpred.shape == (334, 144)
egdp = np.load(PRED / "pv_pred_ens_v5.npy")            # (334,144) EGD 光伏, 供中心十分钟下沉
assert egdp.shape == (334, 144)


def trailing_nanmean(X, W):
    """X:(E,·) 含 NaN → out[e] = nanmean(X[max(0,e−W):e])(只用 e 之前);
    e=0 无历史 → 0(不校正); 全 NaN 列(绑定段外)→ 0, 由 C_ses 的 NaN 保持掩码。"""
    E = X.shape[0]
    out = np.zeros_like(X, dtype=float)
    for e in range(1, E):
        w = X[max(0, e - W):e]
        v = np.isfinite(w).any(axis=0)
        out[e, v] = np.nanmean(w[:, v], axis=0)
    return out


# ---------- 2. 相位定案(±10 分钟读法, M1 遗留) ----------
print("===== 相位研究: 整点预报 vs 移窗小时实际 (o=0 为终点读法) =====")
phase_rows = []
for o in (-2, -1, 0, 1, 2):
    Ao = np.full((365, NL), np.nan)
    for h in range(1, NL - 1):
        lo, hi = 6 * h + o, 6 * h + 6 + o                  # o=0: ti 6h..6h+5 = 终点读法(A_nat)
        if lo >= 0 and hi <= 144:
            Ao[:, h] = P[:, lo:hi].mean(axis=1)
    errs = []
    for s in range(NS):
        hz = np.arange(max(H0[s], 5), 20)
        errs.append(np.abs(F_corr[I0:, s, hz - H0[s]] - Ao[I0:, hz]).ravel())
    mae_o = float(np.concatenate(errs).mean())
    tag = {0: "终点读法(M1 主口径)", -1: "起点读法(P1/P2 声明)"}.get(o, f"移窗{o:+d}")
    phase_rows.append({"移窗o": o, "白天MAE": round(mae_o, 1), "口径": tag})
    print(f"  o={o:+d}: {mae_o:7.1f} kW   {tag}")
o_best = min(phase_rows, key=lambda r: r["白天MAE"])["移窗o"]
assert o_best == 0, f"相位研究不支持终点读法: argmin o={o_best}"
print("[定案] 终点读法(附件2 列标签=时段终点)与整点预报对齐最优, 下沉口径由此锁定\n")

# ---------- 3. 终偏校正层 ----------
eps_raw = A_nat[I0:][:, None, :] - C_ses               # (334,4,24) 混合中心原始残差
b_pv = np.zeros((334, NS, NL))
mae_raw = np.array([np.nanmean(np.abs(eps_raw[:, s, H0[s]:])) for s in range(NS)])
print("===== 终偏校正(混合中心, 候选窗 %s, 逐会话按样本外 MAE 择优) =====" % (W_CAND,))
for s in range(NS):
    best = (np.zeros((334, NL)), mae_raw[s], None)
    for W in W_CAND:
        b = trailing_nanmean(eps_raw[:, s, :], W)      # (334,24)
        m = np.nanmean(np.abs((eps_raw[:, s, :] - b)[:, H0[s]:]))
        print(f"  会话{H0[s]:>2}:00  W={W:>2}: MAE {mae_raw[s]:7.2f} → {m:7.2f}")
        if m < best[1]:
            best = (b, m, W)
    b_pv[:, s, :], mae_new, w_used = best
    assert mae_new <= mae_raw[s] + 1e-6, f"会话{s} 终偏校正劣化"
    print(f"  会话{H0[s]:>2}:00 采纳 W={w_used} (MAE {mae_raw[s]:.2f}→{mae_new:.2f})")
C_adj = C_ses + b_pv                                    # 终偏校正后的中心(M5 消费)
# 负载侧: 各窗口去偏均需优于原始才采纳
rL_raw = ActL[I0:] - Lpred                             # (334,144) 原生十分钟
mae_L_raw = float(np.abs(rL_raw).mean())
b_L = np.zeros_like(rL_raw)
print(f"  负载: 原始 MAE {mae_L_raw:.2f};", end=" ")
for W in (28, 56, 91):
    m = float(np.abs(rL_raw - trailing_nanmean(rL_raw, W)).mean())
    print(f"W={W}: {m:.2f};", end=" ")
b_L = trailing_nanmean(rL_raw, 28)
mae_L_new = float(np.abs(rL_raw - b_L).mean())
if mae_L_new < mae_L_raw:
    print(f"采纳 W=28")
else:
    b_L = np.zeros_like(b_L)
    print(f"全部劣化 → 不去偏(28 样本均值噪声 > 偏差信号), 情景侧按再中心化保持一致性")
L_adj = Lpred + b_L
print(f"  终偏幅度: max|b̂_pv| = {np.nanmax(np.abs(b_pv)):.1f} kW, "
      f"max|b̂_L| = {np.abs(b_L).max():.1f} kW\n")

# ---------- 4. 供体路径库(校正坐标) ----------
eps0 = A_nat[I0:] - C_adj[:, 0, :]                     # (334,24)
G = [C_adj[:, s, H0[s]:] - C_adj[:, s - 1, H0[s]:] for s in range(1, NS)]
donor_pv = np.concatenate([eps0] + G, axis=1).astype(np.float32)     # (334,60)
donor_load = (ActL[I0:] - L_adj).astype(np.float32)                  # (334,144)
eps_ses = (A_nat[I0:][:, None, :] - C_adj).astype(np.float32)        # (334,4,24)
cum = np.zeros((334, NL))
for s in range(1, NS):
    cum[:, H0[s]:] += G[s - 1]
    assert np.allclose(eps0[:, H0[s]:] - cum[:, H0[s]:], eps_ses[:, s, H0[s]:], atol=1e-3)
print(f"===== 供体路径库: 光伏 {donor_pv.shape}, 负载 {donor_load.shape} "
      f"(校正坐标; D4 恒等式 ε_s=ε0−ΣG 逐点成立) =====")

# ---------- 5. 小时内形态曲线库(PLAN §3.4, 供 M5 下沉) ----------
shape = np.ones((334, NL, 6), dtype=np.float32)
for e in range(334):
    r0, r1 = max(I0, I0 + e - W_SHAPE), I0 + e         # 实际日行号窗
    if r1 <= r0:
        continue
    for h in range(NL):
        blk = P[r0:r1, 6 * h:6 * h + 6]
        ah = blk.mean()
        if ah >= 20.0:
            shape[e, h] = blk.mean(axis=0) / ah
print(f"===== 形态曲线库 {shape.shape} (尾随{W_SHAPE}天, 实际小时均值<20kW 取平1) =====")

# ---------- 5b. 中心的十分钟下沉(对 C_adj 重算; M2 的 C_slot.npy 基于校正前中心, 已过期) ----------
# 公式同 M2 §5: 十分钟中心 = 融合整点水平 × EGD 小时内形状(和为1), 保能量。
C_slot_adj = np.full((NS, 334, 144), np.nan, dtype=np.float32)
for s in range(NS):
    for h in range(int(H0[s]), NL):
        sl = np.arange(6 * h, 6 * h + 6)
        egsl = egdp[:334, sl]
        egs = egsl.sum(axis=1)
        ok = egs > 1e-9
        shp = np.where(ok[:, None], egsl / np.where(ok, egs, 1.0)[:, None], 1.0 / 6)
        C_slot_adj[s][:, sl] = C_adj[:, s, h][:, None] * shp
        assert np.allclose(C_slot_adj[s][:, sl].sum(axis=1)[ok], C_adj[ok, s, h], rtol=1e-6)
print(f"===== 十分钟中心(校正后) {C_slot_adj.shape} 保能量; 取代 M2 的 C_slot.npy 供 M5 =====")

# ---------- 6. 协方差健康(活子空间)与 PCA 预览 ----------
blocks = {"eps0(24)": donor_pv[:, :24], "G1(18)": donor_pv[:, 24:42],
          "G2(12)": donor_pv[:, 42:54], "G3(6)": donor_pv[:, 54:60]}
tab_cov, tab_pca = [], []
for name, X in blocks.items():
    live = X.std(axis=0) > 1e-6 * X.std(axis=0).max()  # 结构性零方向(夜间)剔除
    Xl = X[:, live]
    C = np.cov(Xl.T)
    ridge = 1e-8 * np.trace(C) / C.shape[0]
    cond = float(np.linalg.cond(C + ridge * np.eye(C.shape[0])))
    w = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    gap = float(w[0] / w[1]) if w[1] > 0 else np.inf   # 谱隙 λ1/λ2
    tab_cov.append({"块": name, "活维数": int(live.sum()), "总维数": X.shape[1],
                    "条件数(活子空间)": f"{cond:.2e}", "λ1/λ2": f"{gap:.1f}"})
    tab_pca.append({"块": name, "PC1占比": round(float(w[0] / w.sum()), 3),
                    "PC1+PC2占比": round(float(w[:2].sum() / w.sum()), 3)})
print("\n===== 协方差健康(供体库全体 334 日; 夜间结构性零方向剔除后; 抽样为自助, 不经高斯) =====")
print(pd.DataFrame(tab_cov).to_string(index=False))
print(pd.DataFrame(tab_pca).to_string(index=False))

# ---------- 7. 抽样器与探针日检验 ----------
def sample_day(e, N=N_PATHS, W=W_POOL, seed=2026):
    """评估日 e 的情景集合: 供体池 [max(0,e−W), e−1] 整日自助 + ε0 再中心化。
    返回 (光伏路径, 负载路径, 供体索引, 再中心化移位)。e=0 返回 None(冷启动, 见文档)。"""
    lo = max(0, e - W)
    if e - lo <= 0:
        return None
    rng = np.random.default_rng(seed + e)
    idx = rng.integers(lo, e, size=N)
    pv, ld = donor_pv[idx].astype(np.float64), donor_load[idx].astype(np.float64)
    shift = pv[:, :24].mean(axis=0)                    # 丢弃不可预测的池均值漂移(二级检验否决并入)
    pv[:, :24] -= shift[None, :]                       # 仅 ε0 再中心化; G 块是期望修正量信号, 不动
    return pv, ld, idx, shift


probes = [28, 120, 210, 302] if not SMOKE else [28]
tab_probe = []
for e in probes:
    pv, ld, _, sh = sample_day(e)
    m_sh = float(np.abs(sh).max())
    m_ld = float(np.abs(ld.mean(axis=0)).max())
    ens_ok = float(np.abs(pv[:, :24].mean(axis=0)).max())
    tab_probe.append({"探针日e": e, "池宽": e - max(0, e - W_POOL),
                      "再中心化移位max": round(m_sh, 1), "负载池均值max": round(m_ld, 1),
                      "再中心化后|均值|": f"{ens_ok:.1e}"})
    assert ens_ok < 1e-6, f"探针日 {e} 再中心化失败"
print("\n===== 探针日检验(N=%d; 情景中心≡采纳中心由再中心化保证, 移位=被丢弃的不可预测漂移) =====" % N_PATHS)
print(pd.DataFrame(tab_probe).to_string(index=False))

# ---------- 8. 落盘 ----------
if not SMOKE:
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "donor_pv.npy", donor_pv)            # (334,60) ε0|G1|G2|G3 (校正坐标)
    np.save(OUT / "donor_load.npy", donor_load)        # (334,144)
    np.save(OUT / "eps_ses.npy", eps_ses)              # (334,4,24) 混合中心残差(审计指令交付)
    np.save(OUT / "C_adj.npy", C_adj.astype(np.float32))          # (334,4,24) 终偏校正后中心(M5 消费)
    np.save(OUT / "C_slot_adj.npy", C_slot_adj)                   # (4,334,144) 校正后十分钟中心(M5 消费, 取代 M2 C_slot)
    np.save(OUT / "L_adj.npy", L_adj.astype(np.float32))          # (334,144) 负载中心(=L_pred)
    np.save(OUT / "shape_hour.npy", shape)             # (334,24,6)
    np.save(OUT / "bias_pv_hist.npy", b_pv.astype(np.float32))
    with pd.ExcelWriter(OUT / "m3_结果.xlsx", engine="openpyxl") as w:
        pd.DataFrame(phase_rows).to_excel(w, sheet_name="相位研究", index=False)
        pd.DataFrame(tab_cov).to_excel(w, sheet_name="协方差健康", index=False)
        pd.DataFrame(tab_pca).to_excel(w, sheet_name="PCA占比", index=False)
        pd.DataFrame(tab_probe).to_excel(w, sheet_name="探针日检验", index=False)
    print(f"\n结果已保存 -> {OUT}")
else:
    print("\n[smoke] 通过")
