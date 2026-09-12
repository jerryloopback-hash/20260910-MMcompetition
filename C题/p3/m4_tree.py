# -*- coding: utf-8 -*-
"""
P3-M4 情景树构造:逐日(供体池, 条件聚类 3x3x3, 节点注册) + 执行重解集合
================================================
依据 p3_plan.md §四 / D10 / 里程碑 M4; M3 审计两条硬约束(m3_model.md §10):
  ①执行层节点中心必须条件在已实现前缀上, 使会话 s 的执行根中心 = C_adj(e,s);
  ②聚类输入 = 池内 ≤28 条不同整日路径(有效情景数 = 池宽, 不做 N 次重抽样)。
架构(两层分离, 见 m4_model.md §3):
  规划树(供 M5 阶段1 计划 LP 的扩展型): 场景 = 池内全部供体(等权 1/K),
    真值路径 A^ω = C_adj0(e) + ε̃0(d') − shift (ε0 再中心化, 十分钟经形态下沉);
    决策共享 = 阶段2 按 G̃1 聚 3 簇、阶段3 在簇内按 G̃2 聚 3、阶段4 在簇内按 G̃3 聚 3
    (逐阶段全局 PCA top-2 + kmeans, 固定种子); 节点 (a,b) 只覆盖自己块的时段。
  执行重解(供 M5 会话 s 的 D2 全量追索): 根 = (C_adj(e,s), 实测 SOC) —— D10 "条件在实测 SOC"
    在此成立; 场景默认取全池(中心已带一阶条件, 全池谱宽比实测 6:00=0.91/12:00=0.83, 影响小),
    cond=True 则用已实现前缀路由出的节点成员(硬条件; 18:00 中位仅 1 条, 会把晚间负载残差
    std 253kW 锁死为 0, 故不作默认, 留作 M7 消融臂)。两条路径都自检 assert。
SOC: 规划 LP 中为逐场景连续状态(由计划执行物理推出, 不作节点维度——决策依赖不确定性的
  实现层在执行重解, 其根 SOC 即实测值); 硬约束①在执行层逐会话成立(脚本 assert)。
冷启动: e=0(2.1) 无供体 → 无树(M5 走确定性单日补丁); e≥1 池宽 = min(28, e)。
用法: import m4_tree  (M5 消费); 直接运行 = 自检与探针日报告 → m4_out/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent          # C题/
M3 = Path(__file__).resolve().parent / "m3_out"
OUT = Path(__file__).resolve().parent / "m4_out"

NS, NL = 4, 24
H0 = np.array([0, 6, 12, 18])
I0 = 31
W_POOL = 28
K_LEAF = 3                                             # 每阶段分支数(3x3x3)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
PROBES = [28, 120, 210, 302] if not SMOKE else [28]

# ---------- 数据 ----------
A_nat = np.load(M3.parent / "m1_out" / "A_nat.npy")    # (365,24) 整点实际(终点读法)
C_adj = np.load(M3 / "C_adj.npy").astype(np.float64)   # (334,4,24) 行e↔实际日31+e
C_slot_adj = np.load(M3 / "C_slot_adj.npy").astype(np.float64)   # (4,334,144)
L_adj = np.load(M3 / "L_adj.npy").astype(np.float64)   # (334,144)
shape_hour = np.load(M3 / "shape_hour.npy").astype(np.float64)   # (334,24,6)
donor_pv = np.load(M3 / "donor_pv.npy").astype(np.float64)       # (334,60) ε0|G1|G2|G3
donor_load = np.load(M3 / "donor_load.npy").astype(np.float64)   # (334,144)

HRS = [np.arange(H0[s], NL) for s in range(NS)]        # 各会话绑定小时 6/12/18/24 维
BLK = [(H0[s] * 6, (H0[s] + 1) * 6 if s < 3 else 144) for s in range(NS)]  # 各块 slot 范围
# donor_pv 列切分: ε0(24) | G1(18, hours 6..23) | G2(12, hours 12..23) | G3(6, hours 18..23)
G_BLOCKS = [(24, 42), (42, 54), (54, 60)]

# ---------- 工具 ----------
def pca_top2(X):
    """X:(n,d) → 投影(n,2), 载荷(d,2), 活维掩码, 前2主成分方差占比, 活维中心。
    返回中心 mu 与载荷 P 是为了让外部点(new point)也能按同一基投影——路由必需。"""
    live = X.std(axis=0) > 1e-9
    Xl = X[:, live]
    mu = Xl.mean(axis=0)
    Xc = Xl - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:min(2, len(S))].T
    Z = Xc @ P
    share = float((S[:len(P)] ** 2).sum() / (S ** 2).sum()) if S.sum() > 0 else 1.0
    return Z, P, live, share, mu


def kmeans_fixed(X, k, seed):
    """scipy kmeans2 包装: 固定种子; 返回标签与投影空间质心。"""
    k = max(1, min(k, len(X)))
    if k == 1:
        return np.zeros(len(X), dtype=int), X.mean(axis=0, keepdims=True)
    cent, lab = kmeans2(X, k, iter=30, minit="++", seed=seed)
    return lab.astype(int), cent


def cluster_stage(X, k, seed):
    """PCA top-2 → kmeans; 返回标签、方差占比、切分模型(供路由)。"""
    Z, P, live, share, mu = pca_top2(X)
    lab, cent = kmeans_fixed(Z, k, seed)
    model = dict(live=live, mu=mu, P=P, cent=cent)
    return lab, share, model


def assign_node(model, x):
    """把一点 x 按某个切分模型路由到簇号(最近质心, PCA 空间)。"""
    z = (x[model["live"]] - model["mu"]) @ model["P"]
    return int(np.argmin(((model["cent"] - z[None, :]) ** 2).sum(axis=1)))


def pool_of(e):
    """评估日 e 的供体池(不同整日路径, 全部 < e)。"""
    lo = max(0, e - W_POOL)
    return np.arange(lo, e), e - lo


# ---------- 每日构造 ----------
def build_scenarios(e):
    """评估日 e 的情景集合(供 M5 规划 LP):
    PV 整点/十分钟真值路径、负载十分钟路径、权重、供体索引。"""
    idx, K = pool_of(e)
    assert K >= 1 and idx.max() < e, "供体池泄漏"
    eps0 = donor_pv[idx, :24]                              # (K,24)
    shift = eps0.mean(axis=0)                              # ε0 再中心化移位
    A_hour = C_adj[e, 0, :] + eps0 - shift                 # (K,24) 场景整点真值(不截零, 保恒等式)
    # 十分钟: 中心 slot + (整点残差)·形态剖面 (剖面均值=1, 保小时均值)
    r_hour = A_hour - C_adj[e, 0, :]                       # (K,24) 截零后的整点残差
    sh = shape_hour[e]                                     # (24,6) 剖面, 均值=1
    PV_slot = C_slot_adj[0, e, :][None, :] + np.repeat(r_hour, 6, axis=1) * sh.ravel()[None, :]
    # 不变量(截零前): 情景十分钟光伏的**小时均值**必须等于情景整点真值
    # —— 挡住"中心下沉尺度"类错误(如形状误用和为1 使中心缩 6 倍)
    assert np.allclose(PV_slot.reshape(K, NL, 6).mean(axis=2), A_hour, atol=1e-6), \
        "十分钟情景未守住小时均值(检查 C_slot_adj 的下沉口径)"
    PV_slot = np.maximum(PV_slot, 0.0)
    # 负载: 逐slot再中心化
    rl = donor_load[idx]                                   # (K,144)
    L_slot = L_adj[e][None, :] + rl - rl.mean(axis=0)[None, :]
    w = np.full(K, 1.0 / K)
    return dict(e=e, idx=idx, K=K, w=w, A_hour=A_hour, PV_slot=PV_slot, L_slot=L_slot,
                shift=shift, shiftL=rl.mean(axis=0))


def build_tree(e, scen):
    """3x3x3 条件聚类树(供 M5 规划 LP 的决策共享结构 + 执行重解路由)。
    节点注册: 每阶段节点 = (成员场景号, 父节点号); 阶段 s 节点的 (a,b) 只覆盖块 s 的 slot。
    每个父节点同时存下切分模型 model 与子映射 child_map, 使当日已实现前缀能被路由(route_node)。"""
    idx = scen["idx"]
    K = scen["K"]
    Gs = [donor_pv[idx, a:b] for a, b in G_BLOCKS]          # G̃1/G̃2/G̃3
    nodes = []                                             # nodes[lev] = list of dict
    lab = np.zeros(K, dtype=int)                           # 当前前缀标签(全局节点号)
    lev_info = []
    root = None
    for lev in range(3):
        G = Gs[lev]
        lab_new = np.full(K, -1, dtype=int)
        lev_nodes = []
        parents = np.unique(lab) if lev > 0 else [0]     # lev=0 的 pi=0 只用于种子, 保持与 v1 同种子
        for pi in parents:
            mem = np.where(lab == pi)[0] if lev > 0 else np.arange(K)
            k = min(K_LEAF, len(mem))
            sub, _sh, model = cluster_stage(G[mem], k, seed=1000 * (e + 1) + 10 * lev + pi)
            child_map = {}
            for j in range(k):
                mm = mem[sub == j]
                if len(mm) == 0:
                    continue                             # kmeans2 可能给出空簇, 不注册
                child_map[int(j)] = len(lev_nodes)
                lab_new[mm] = len(lev_nodes)
                lev_nodes.append(dict(lev=lev, parent=int(pi), members=mm,
                                      n=len(mm), rep=float(len(mm))))
            if lev == 0:
                root = dict(model=model, child_map=child_map)
            else:
                nodes[lev - 1][pi]["model"] = model      # 父节点的切分模型(供路由)
                nodes[lev - 1][pi]["child_map"] = child_map
        lab = lab_new
        nodes.append(lev_nodes)
        lev_info.append([n["n"] for n in lev_nodes])
    return dict(nodes=nodes, leaf=lab, lev_sizes=lev_info, root=root)


def route_node(tree, e, s):
    """把当日已实现前缀 (G_1(e),…,G_s(e)) 路由到阶段 s+1 的节点号(s∈{1,2,3})。
    这是 D10"节点条件在修正量上"在执行层的落点: M5 用所落节点的成员做条件场景。"""
    assert 1 <= s <= 3, "s 取 1..3"
    Gs = realized_prefix(e)                                # Gs[k-1] 定义在 hours H0[k]..23
    nid = tree["root"]["child_map"][assign_node(tree["root"]["model"], Gs[0])]
    for lev in range(1, s):
        par = tree["nodes"][lev - 1][nid]
        nid = par["child_map"][assign_node(par["model"], Gs[lev])]
    return nid


def realized_prefix(e):
    """当日已实现修正量 G_k(e) = C_adj(e,k) − C_adj(e,k−1), 及恒等式自检。"""
    G = [C_adj[e, s, HRS[s]] - C_adj[e, s - 1, HRS[s]] for s in range(1, NS)]
    # 恒等式: C_adj(e,s) = C_adj(e,0) + Σ_{k≤s} G_k(e) (在 HRS[s] 小时上; G_k 取自 H0[s] 起的尾部)
    for s in range(1, NS):
        rec = C_adj[e, s, HRS[s]]
        acc = C_adj[e, 0, HRS[s]] + sum(G[k - 1][H0[s] - H0[k]:] for k in range(1, s + 1))
        assert np.allclose(rec, acc, atol=1e-6), f"日{e} 阶段{s} 前缀恒等式破坏"
    return G


def build_exec_ensemble(e, s, tree=None, cond=False):
    """执行重解集合(D2 全量追索, 硬约束①): 根中心 = C_adj(e,s)。
    cond=False(默认): 场景 = 全池(≤28 条整日路径)。
      —— 中心 C_adj(e,s) 已把当日至 s 的全部信息(含修正量)带进一阶条件, 全池只让二阶
         (谱宽)略宽; 实测条件/全池谱宽比 6:00=0.91、12:00=0.83, 对成本相关块影响小。
    cond=True: 场景 = 当日已实现前缀所落节点的成员子集(route_node 路由), 即 D10 节点的
      硬条件版。仅在 6:00 有 ~10 条; 12:00 中位 3(15.9% 为 1); 18:00 中位 1(66.7% 为 1)——
      硬条件会把晚间负载残差(std 253 kW)一并锁死为 0, 有欠套保风险, 故不作默认, 留作消融臂。
    场景 = 所选子集的 ε̃_s = ε̃0 − Σ_{k≤s} G̃_k (绑定小时), 再中心化(信任采纳中心)。
    返回 hour 路径(K', hours) 与 slot 路径(K', slots from H_s)。"""
    idx_all, K = pool_of(e)
    assert K >= 1
    if cond:
        if tree is None:
            tree = build_tree(e, build_scenarios(e))
        nid = route_node(tree, e, s)
        members = tree["nodes"][s - 1][nid]["members"]
        idx = idx_all[members]
        node_id = int(nid)
    else:
        idx, node_id = idx_all, -1
    eps0 = donor_pv[idx, :24]
    Gs = [donor_pv[idx, a:b] for a, b in G_BLOCKS]
    hrs = HRS[s]
    # ε̃_s[h] = ε̃0[h] − Σ_{k≤s} G̃_k[h](h ∈ HRS[s] ⊆ 各 k 的绑定段, k≤s 时恒成立)
    eps_s = eps0[:, hrs].copy()
    for k in range(1, s + 1):
        eps_s = eps_s - Gs[k - 1][:, H0[s] - H0[k]:]
    mu = eps_s.mean(axis=0)
    paths_hour = np.maximum(C_adj[e, s, hrs][None, :] + eps_s - mu[None, :], 0.0)
    # slot 版: 根中心的 slot 版 + (小时残差 − 根中心小时残差)·形态剖面
    root_slot = C_slot_adj[s, e][BLK[s][0]:]
    res_hour = paths_hour - C_adj[e, s, hrs][None, :]
    sh = shape_hour[e][H0[s]:]                             # (nh,6)
    paths_slot = root_slot[None, :] + np.repeat(res_hour, 6, axis=1) * sh.ravel()[None, :]
    paths_slot = np.maximum(paths_slot, 0.0)
    # 负载: 根 = L_adj, 残差 = 供体 r̃L − 均值(全 144 slot, 取本会话及以后)
    rl = donor_load[idx]
    L_slot = L_adj[e][None, :] + rl - rl.mean(axis=0)[None, :]
    return dict(e=e, s=s, idx=idx, node=node_id, cond=cond, n_cond=len(idx),
                root_hour=C_adj[e, s, hrs], paths_hour=paths_hour,
                paths_slot=paths_slot, L_slot=L_slot[:, BLK[s][0]:],
                shift_exec=mu)


# ---------- 自检与探针报告 ----------
def check_day(e):
    scen = build_scenarios(e)
    tree = build_tree(e, scen)
    K = scen["K"]
    # 场景中心恒等: mean(A_hour) = C_adj0(e)(ε0 再中心化)
    assert np.allclose(scen["A_hour"].mean(axis=0), C_adj[e, 0, :], atol=1e-9), "再中心化失败"
    # 嵌套一致: 阶段3 成员 ⊂ 阶段2 成员, 阶段4 ⊂ 阶段3
    for lev in (1, 2):
        for n in tree["nodes"][lev]:
            par = tree["nodes"][lev - 1][n["parent"]]
            assert np.all(np.isin(n["members"], par["members"])), "树嵌套破坏"
    # 占用: 每场景恰好一个叶
    assert np.all(tree["leaf"] >= 0) and len(np.unique(tree["leaf"])) == len(tree["nodes"][2])
    # 执行重解: 根中心 = C_adj(e,s)(硬约束①; 前缀恒等式已在 realized_prefix 内断言)
    G = realized_prefix(e)
    nconds = []
    for s in range(1, NS):
        for cd in (False, True):                           # 默认(全池) 与 条件臂 都必须自检通过
            ee = build_exec_ensemble(e, s, tree=tree, cond=cd)
            assert np.allclose(ee["root_hour"], C_adj[e, s, HRS[s]], atol=1e-9), "执行根中心错位"
            assert ee["paths_hour"].min() >= 0 and ee["n_cond"] >= 1
            if cd and s > 1:                               # 路由出的节点必须嵌套在父节点内
                par = tree["nodes"][s - 2][tree["nodes"][s - 1][ee["node"]]["parent"]]
                assert np.all(np.isin(tree["nodes"][s - 1][ee["node"]]["members"], par["members"]))
        nconds.append(build_exec_ensemble(e, s, tree=tree, cond=True)["n_cond"])
    # 场景整点真值均值即采纳中心的一致性核对(十分钟后均值不必等于中心, 因截零)
    return dict(e=e, K=K, sizes=[[n["n"] for n in tree["nodes"][lev]] for lev in range(3)],
                nconds=nconds)


if __name__ == "__main__":
    rows = []
    print("===== M4 自检(逐日树构造 + 执行根恒等式) =====")
    days = PROBES if SMOKE else range(334)
    for e in days:
        if pool_of(e)[1] == 0:
            rows.append({"e": e, "池宽": 0, "簇2": "冷启动", "占用2": 0,
                         "占用3": 0, "占用4": 0, "最大叶": 0})   # 2.1 确定性单日补丁(M5)
            continue
        info = check_day(e)
        sizes = info["sizes"]
        occ = [sum(1 for x in s if x > 0) for s in sizes]
        leaf_sizes = sorted(sizes[2], reverse=True)
        rows.append({"e": e, "池宽": info["K"],
                     "簇2": "/".join(map(str, sizes[0])), "占用2": occ[0],
                     "占用3": occ[1], "占用4": occ[2],
                     "最大叶": leaf_sizes[0] if leaf_sizes else 0,
                     "条件集6/12/18": "/".join(map(str, info["nconds"]))})
        if SMOKE or (e % 60 == 0):
            print(f"  e={e:3d} 池{info['K']:2d} 簇2[{sizes[0]}] 占用2/3/4={occ[0]}/{occ[1]}/{occ[2]} "
                  f"条件集6/12/18={info['nconds']}")
    df = pd.DataFrame(rows)
    print("\n占用节点数均值: 阶段2 %.1f / 阶段3 %.1f / 阶段4 %.1f" % (
        df["占用2"].mean(), df["占用3"].mean(), df["占用4"].mean()))
    print("最大叶规模均值: %.1f (叶≈单供体时阶段4 非预期性退化, 见文档)" % df["最大叶"].mean())
    if not SMOKE:
        OUT.mkdir(exist_ok=True)
        with pd.ExcelWriter(OUT / "m4_结果.xlsx", engine="openpyxl") as w:
            df.to_excel(w, sheet_name="逐日树占用", index=False)
        print(f"\n自检报告已保存 -> {OUT}")
    else:
        print("\n[smoke] 通过")
