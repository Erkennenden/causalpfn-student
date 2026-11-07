import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r"D:\causalpfn-student\iPFN_demo\code\sparse_j")
from pfn_social.phase6_limited_outcomes import generate_mechanism, make_observed_draw
from predict import load_selected_model

print("正在加载模型...")
model = load_selected_model()

# ==========================================
# 满配参数：864 个数据点
# ==========================================
degrees = [2, 4, 6, 8, 10, 12]
repeats = 12  # 12个种子
observed_budgets = [20, 50, 100]
mechanisms = [0, 1, 2, 3] # 4个机制

case_grid = [
    (deg, seed, m_obs, mech) 
    for deg in degrees 
    for seed in range(repeats) 
    for m_obs in observed_budgets
    for mech in mechanisms
]

total_runs = len(case_grid)
print(f"🚀 终极满配版启动！共 {total_runs} 个数据点。")

all_results = []

for i, (deg, seed, m_obs, mech) in enumerate(case_grid):
    np.random.seed(seed)
    
    mech_obj = generate_mechanism(3, mech)
    obs_rows, _ = make_observed_draw(mech_obj, seed)
    chosen = next(row for row in obs_rows if row["m_observed"] == m_obs)
    inputs = {k: v.numpy().copy() for k, v in chosen["batch"].items()}
    
    N = inputs["adjacency"].shape[0]
    target_edges = int(N * deg / 2)
    
    # 【提速核心】：用环骨架打底，然后随机加边。绝不会有孤立节点，速度极快
    G = nx.cycle_graph(N)
    remaining_edges = target_edges - N
    if remaining_edges > 0:
        possible_edges = list(nx.non_edges(G))
        np.random.shuffle(possible_edges)
        G.add_edges_from(possible_edges[:remaining_edges])
        
    inputs["adjacency"] = nx.to_numpy_array(G, dtype=np.float32)
    
    # 模型推理
    preds_flat, _ = model.predict_arm_means(**inputs, return_diagnostics=True)
    preds = preds_flat.reshape(-1, 2, 2)
    
    # 计算指标
    true_mu = mech_obj.parent.truth.reshape(-1, 2, 2)
    rmse = np.sqrt(np.mean((preds - true_mu) ** 2))
    cov = np.mean(np.abs(preds - true_mu) <= 1.96 * rmse) * 100
    
    all_results.append({
        'Degree': deg,
        'Seed': seed,
        'Observed_N': m_obs,
        'Mechanism_ID': mech,
        'RMSE': rmse,
        'Coverage': cov
    })
    
    # 实时打印进度
    print(f"[{i+1}/{total_runs}] 密度={deg}, 种子={seed}, 观测={m_obs}, 机制={mech} | RMSE: {rmse:.4f}")

# 保存数据
df = pd.DataFrame(all_results)
df.to_csv('ultimate_results.csv', index=False)
print(f"\n✅ 跑完！共生成了 {len(df)} 个数据点，保存为 ultimate_results.csv")

# ==========================================
# 画图：带误差带的三分面图
# ==========================================
sns.set_theme(style="whitegrid", context="paper")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, obs_n in enumerate(observed_budgets):
    ax = axes[i]
    subset = df[df['Observed_N'] == obs_n]
    sns.lineplot(
        data=subset, x='Degree', y='RMSE', 
        marker='o', errorbar='sd', ax=ax, 
        color='#2166AC', linewidth=2
    )
    ax.set_title(f"Observed Budget: {obs_n}", fontweight='bold')
    ax.set_xlabel("Average Degree (Density)")
    if i == 0:
        ax.set_ylabel("RMSE (Mean ± SD)")
    
fig.suptitle("Ultimate E1.1: RMSE vs Density Across Multiple Mechanisms & Seeds", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('ultimate_plot.png', dpi=300)
print("✅ 顶级图表已保存为 ultimate_plot.png")