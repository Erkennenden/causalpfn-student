# E1.6 全连接图实验

本目录中的 E1.6 实现只替换网络拓扑，保留 demo 的节点数、Bernoulli(0.5) 处理分配、连续结果 DGP、四状态真值、观测掩码、效应定义、冻结 iPFN J checkpoint 和基线接口。

## 需要阅读的已有文件

- `notebooks/current_ipfn_student_walkthrough.html`：任务流程与输入输出解释。
- `code/sparse_j/pfn_social/phase3_hierarchical.py`：连续结果 DGP 与结构结果函数。
- `code/sparse_j/pfn_social/phase5_large_network.py`：机制、随机流和四状态真值。
- `code/sparse_j/pfn_social/phase6_limited_outcomes.py`：20/50/100 个观测结果的嵌套掩码。
- `code/sparse_j/benchmark.py`：统一模型/基线加载和预测接口。
- `code/sparse_j/pfn_social/intervals.py`：冻结 iPFN-J 的 ATE 边际后验区间。
- `code/sparse_j/evaluate.py`：ATE 点估计、95% 区间和覆盖率统一评估。
- `code/sparse_j/make_figures.py`、`make_table.py`：老师提供的绘图、制表模板。
- `code/sparse_j/models/iPFN_J.pt`：本轮验证使用的冻结 checkpoint。

## E1.6 新增文件

- `code/sparse_j/e1_6_complete_graph.json`：固定节点数、随机种子、效应顺序和默认方法。
- `code/sparse_j/pfn_social/e1_6_complete.py`：无自环全连接图、真值重算和经验支持诊断。
- `code/sparse_j/benchmark_e1_6_complete.py`：数据生成、iPFN/基线评测、RMSE 汇总和作图。
- `code/sparse_j/tests/test_e1_6_complete.py`：拓扑、可复现性、数据泄漏与支持性质测试。

## 运行

从仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r iPFN_demo/code/sparse_j/requirements.txt
python -m pip install pytest

cd iPFN_demo/code/sparse_j
pytest -q tests/test_e1_6_complete.py
python benchmark_e1_6_complete.py --cohort smoke --output ../../../e1_6_smoke
python make_figures.py --input ../../../e1_6_smoke --output ../../../e1_6_smoke_figures
python make_table.py --input ../../../e1_6_smoke --output ../../../e1_6_smoke_tables
```

正式评测将 `--cohort smoke` 改为 `--cohort full`，并使用新的输出目录。`full` 包含 8 个机制、每个机制 12 次处理/噪声重复、3 个观测数量，共 288 个上下文。

输出包括：

- `task_metrics.csv`：逐任务的 oracle、iPFN 与基线 ITE RMSE、ATE 误差、运行时间和支持标记；
- `aggregate_metrics.csv`：任务平均及任务间标准差；
- `node_differences.csv`：iPFN 节点误差分布；
- `rmse_comparison.png`：iPFN 与基线的 ITE/ATE RMSE 图；
- `run_manifest.json`：配置、随机种子范围与 checkpoint SHA256；
- `conclusions.md`：全连接图下的结构和支持结论。
- `evaluation/context_metrics.csv`、`summary*.csv`：ATE RMSE、bias、95% 区间覆盖率和区间宽度；
- `figures/ate_accuracy.{pdf,svg,png}`：使用 `make_figures.py` 生成的论文式 ATE RMSE 图；
- `tables/ate_m*.{csv,tex}`：使用 `make_table.py` 生成的 bias、方差、RMSE 和覆盖率表。

正式提交结果位于 `e1_6_results/final/`。完整 288 个上下文中，冻结 iPFN-J 的 95% ATE 区间全部成功生成；按观测数量与效应分组后的实际覆盖率为 94.79%–100%，全部达到至少 90% 的验收线。覆盖率按 `covered / all requested intervals` 计算，因此若区间计算失败也会进入分母，不会被静默剔除。

## 全连接图必须单独说明的支持问题

在 $K_N$ 中，节点 $i$ 的已处理邻居数等于全网已处理总数减去 $T_i$。因此一次处理分配最多只会出现四个 `(t,s)` 单元格中的两个。总体随机化分布下四个目标状态仍都有正概率，但一份有限样本中的经验支持可能缺失。

代码同时保存 `empirical_effect_supported` 和模型自身的 `predictor_*_supported`。如果 iPFN 对经验缺失的对比仍返回有限值，该结果属于外推，不能当作普通有覆盖的 plug-in 估计。全连接图所有节点度数都为 199，节点差异来自协变量和 DGP，而不是度数差异。
