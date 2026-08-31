# Interference-CausalPFN：ER Topology Embedding + 标准注意力

这是只保留 PFN 主模型的版本。

当前模型包含：

- 只生成 ER 网络图；
- factual observed information + causal query 作为模型输入；
- network/topology-aware embedding；
- Transformer 使用普通全局 self-attention，邻接矩阵只用于 network/topology-aware embedding；
- Direct / Spillover / Total 三个 GMM effect heads；
- 只评估 PFN 的 ITE / ATE，不再包含任何比较 baseline。

代码里的 `structural_baseline`、`baseline_sd`、`*_baseline_coefficient` 属于结果生成 DGP 的基线项，不是比较算法，因此保留。

旧模型结构保存的 checkpoint 与本版本不兼容，请重新训练后再使用 `--test-only`。

## 8 张 MUSA 训练

```bash
bash run_8musa_train.sh
```

测试输出：

- `algorithm2_summary.json`
- `algorithm2_dataset_metrics.csv`
- `algorithm2_unit_estimates.csv`

## 测试

```bash
python -m pytest -q
```
