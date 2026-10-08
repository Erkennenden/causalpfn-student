> **当前为 V3 无 IV 版，使用三种 f 形式；本文件保留首版设计说明，当前接口以 [V3_先看这里.md](../V3_先看这里.md) 为准。**

# 随机结果函数生成模块（CEPO 版）

完整 DGP、效应定义和训练说明见根目录 README_CN.md。来源与许可证见本目录 SOURCE_MANIFEST.json / LICENSE。

每份任务抽取一个确定性 f_theta，输入 `[X, neighbor_X, t, e]`，生成 `Y_obs=f_theta+epsilon`。f 在该任务所有节点及所有反事实计算中共享。噪声仅在观测输出端加入。对 K~Binomial(d,.5) 在多数状态内逐支持点计算 f 并精确加权，构造无噪声 CEPO 标签。

| 字段 | 单任务形状 | 用途 |
|---|---|---|
| tokens | [1,N,5] | 事实 X/T/Y/E/标准化度数 |
| queries | [1,N,3,2] | 状态坐标 (0,0)、(1,0)、(1,1) |
| cepo_target | [1,N,3] | 三个无噪声聚合 mu 标签 |
| oracle_arm_means | [1,N,2,2] | 四个 arm 的精确均值，轴为节点/t/s |
| oracle_ite | [1,N,3] | Direct / Spillover / Total 真值 |
| oracle_ate | [1,3] | 三种真值的节点平均 |

模型只读 tokens、queries、adjacency。oracle 字段用于监督/评估，不能作为输入。旧的 `[t_a,e_a,t_b,e_b]` 点暴露查询和 query_effect loss 已不用于本版。

从项目根目录导出两份可读样例（不训练模型）：
```bash
python -m causalfm_dgp_v1.demo --tasks 2 --seed 2026 --output runs/dgp_example
```

输出 nodes.csv、batch.pt、mechanisms.json、summary.json，包含 f 的实际权重用于复现。正式训练使用 `python -m causalfm_experiment train`，默认 lazy task cache 不为每份任务重复保存图。
