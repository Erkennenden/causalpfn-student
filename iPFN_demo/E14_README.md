# E1.4 RGG 实验运行说明

张子瑜。本包是 causalpfn-student 的增量代码，需合并到原仓库使用。
基底：https://github.com/Erkennenden/causalpfn-student
原实验基底版本：b3c008c3cf178b929681afd6be803a132eee5434。

将代码包和结果包中的 iPFN_demo 合并到原仓库同名目录。原模型权重、原有场景、make_figures.py 和 make_table.py 由原仓库提供，不重复打包。实验结果见结果包，结论见 E14_简要说明.docx。

## 设置与口径

- 原实验：均匀、双簇两种坐标分布 × 半径 0.12、0.18、0.24，共六种场景（family 7–12）。每场景 20 个机制，每机制 12 次重复，N=200，M=20/50/100。种子 2026102704。
- 双簇中心为 (0.3,0.3)、(0.7,0.7)，等概率选簇，簇内标准差 0.12；坐标截断在单位正方形。出现孤立节点时整张点云重抽，不补边，不强制连通。实验对象是条件于无孤立点的 RGG。
- 独立复验：均匀 r=0.24，100 个新机制；主要端点为 M=50 的溢出效应覆盖率。种子 2026102714。原样本不并入复验。
- 匹配实验：50 对均匀/双簇图（family 13/14），每图 1,800 条边、平均度 18；半径由坐标距离确定，配对内共享特征、系数、处理、噪声和观测掩码。种子 2026102724。平均度相同不代表其他图统计量相同。
- 处理、响应方程与观测设计沿用原示例；模型和区间算法冻结。补充实验均重复 12 次，M=20/50/100，比较 J、HT、Hajek 和两个 kernel baseline。具体协议见 e14_followup_protocol.json。
- 高暴露为被处理邻居比例严格大于 1/2。Direct=μ(1,0)−μ(0,0)，spillover=μ(1,1)−μ(1,0)，total=μ(1,1)−μ(0,0)。ITE 为节点效应，ATE 对全部 200 个节点平均。
- 模板 Bias 为各机制内平均误差取绝对值后跨机制平均；方差使用机制内 R−1 分母；RMSE 为汇总平方误差后开根。带符号 bias 是另一口径。
- Bootstrap 为 10,000 次、种子 2026102734；按整个机制或机制对重抽样，原始六场景汇总按场景分层。差值比较用共同有效上下文，区间未作多重比较校正。失败区间留在覆盖率分母，不支持区间的 baseline 记为不适用。
- oracle 使用额外真实信息，仅作诊断，不参与实际 baseline 排名。空间位置与特征独立，本实验不包含空间混杂。

## 安装与运行

推荐 Python 3.12。在 iPFN_demo 下安装原仓库 requirements-notebook.txt；已有能运行原 notebook 的环境可复用。requirements-e14-tested.txt 记录实验使用的依赖版本。

进入 iPFN_demo/code/sparse_j，确保 python 指向已安装依赖的环境。下列目录使用 _rerun 后缀，避免覆盖随附结果；再次运行时请换新目录名。

```bash
python -m unittest -v test_e14 test_e14_followup test_e14_analysis
python benchmark_e14.py --mechanisms 20 --repeats 12 --output ../../results/e14_rgg_rerun
python make_figures.py --input ../../results/e14_rgg_rerun --output ../../results/e14_rgg_rerun/figures
python make_table.py --input ../../results/e14_rgg_rerun --output ../../results/e14_rgg_rerun/tables
python report_e14.py --input ../../results/e14_rgg_rerun
python benchmark_e14.py --design replication --mechanisms 100 --repeats 12 --methods j ht hajek kernel-old kernel-new --output ../../results/e14_replication100_rerun
python benchmark_e14.py --design degree-matched --mechanisms 50 --repeats 12 --methods j ht hajek kernel-old kernel-new --output ../../results/e14_degree_matched50_rerun
python analyze_e14_followup.py --original ../../results/e14_rgg_rerun --replication ../../results/e14_replication100_rerun --matched ../../results/e14_degree_matched50_rerun --output ../../results/e14_followup_analysis_rerun
python report_e14_followup.py --input ../../results/e14_followup_analysis_rerun --replication ../../results/e14_replication100_rerun --matched ../../results/e14_degree_matched50_rerun
python verify_e14_followup.py --replication ../../results/e14_replication100_rerun --matched ../../results/e14_degree_matched50_rerun --output ../../results/e14_followup_analysis_rerun/audit.json
```

原有场景检查可另运行 `python benchmark.py --cohort pilot84 --output ../../results/original_rerun`。快速试跑可运行 `python benchmark_e14.py --mechanisms 1 --repeats 2 --families 7 10 --output ../../results/e14_smoke_new`，不能用其估计正式覆盖率。

notebooks/e14_rgg_walkthrough.ipynb 展示单份数据的生成、预测与效应核对；为便于 PR 审阅，已清除单元输出，运行后即可重新显示。

原始图表直接使用 make_figures.py、make_table.py；补充图表由 report_e14_followup.py 调用这两个模板文件的函数，按图族分别输出。本次只删除该脚本中的旧 HTML 报告和 PR 文案生成部分，保留绘图和模板调用。

结果包为汇总版，不含逐次输入、逐节点预测和完整 context_metrics.csv。完整重算需先运行上述实验，或恢复完整原始数据；仅查看图表无需重跑。代码中的模拟、估计和统计计算未改动，本次未重新运行正式实验。
