# causalpfn-student

## 95% 区间覆盖率评测

评测入口在 `iPFN_demo/code/sparse_j`。在 Python 3.12 环境中安装该目录的
`requirements.txt` 后运行：

```bash
cd iPFN_demo/code/sparse_j
python -m pip install -r requirements.txt

# 84 个 context：7 个机制族 × 2 个机制实例 × 2 次重复 × 3 个观测样本量。
# 输出路径必须尚不存在。j 是冻结的 J。
python benchmark.py --cohort pilot84 --methods j --output /tmp/ipfn-coverage95-pilot

# 完整开发集：7 × 8 × 12 × 3 = 2016 个 context。
python benchmark.py --cohort full --methods j --output /tmp/ipfn-coverage95-full

# 对已经保存了区间的实验重新汇总，无须重新预测。
python evaluate.py --input /tmp/ipfn-coverage95-full
```

默认区间水平为 `0.95`，可用 `--interval-level` 修改。只运行一个案例可用
`--cohort one --family 0 --mechanism 0 --repeat 0 --observed 50`。
两个评测规模均使用原有数据生成机制和随机种子。

### 输出与评价标准

`evaluation/` 中自动生成以下 CSV：

| 文件 | 内容 |
| --- | --- |
| `context_metrics.csv` | 每个 context、方法、效应的原始计数和指标 |
| `summary.csv` | 按方法、观测样本量、ITE/ATE、效应类型汇总 |
| `summary_by_family.csv` | 在上述分组上，进一步区分机制族 |
| `summary_by_mechanism.csv` | 进一步区分独立机制实例 |

每张表分别报告 direct、spillover、total 三种效应的 ITE 和 ATE。
四个 arm 的顺序为 `(00, 01, 10, 11)`，对比分别为 `10−00`、`11−10`、`11−00`。
ATE 是全部查询节点的平均效应，不限于已观测到结果的节点。

- `coverage`：有效区间中包含真实效应的比例。ITE 先对每个节点判断；ATE 每个 context 判断一次。
- `coverage_including_failures`：支持区间的所有请求中，成功覆盖真值的比例；失败、缺失、非有限或反向区间也保留在分母中，按未覆盖计。
- `mean_width`：有效区间的平均宽度。
- `rmse`、`bias`：有效点估计的误差和偏差。
- `interval_requested`、`interval_valid`、`interval_success_rate`、`interval_failure_contexts`：用于检查是否存在失败结果被筛掉的问题。
- `coverage_se`、`coverage_including_failures_se`：按独立机制实例聚类的近似 Monte Carlo 标准误；同一机制中的节点和重复不被当作独立样本。少于两个有贡献的机制时为 `nan`。

覆盖率需结合区间宽度、点估计误差和失败率查看，并分别检查各机制族。
`nan` 表示不支持或无法估计，不能解释为零误差或零覆盖率。
没有已知真值的真实数据只能输出区间，无法直接计算覆盖率。

### 区间的统计含义与适用范围

区间适用于 `predict.py` / benchmark 的**冻结 J 模型**。
计算使用两个表征、四个噪声方差分量的后验权重，对八分量高斯混合的边际 CDF
求 2.5% 和 97.5% 分位数。ITE 的跨 arm 协方差、ATE 的跨节点协方差均保留。
ATE 区间由平均效应的后验分布计算。

这些是条件于冻结模型和当前图的 **95% 边际后验可信区间**；
实际覆盖率由模拟评测计算，名义水平为 95% 不保证实际覆盖率达到 95%。
区间针对生成器定义的因果均值对比，不额外加入新结果的观测噪声，
也不是所有节点同时覆盖的区间。不包含重新训练参数带来的不确定性。

`predict_current.py` 中的区域集成及其他基线不支持区间。
benchmark 中不支持区间的方法标记为 `*_interval_supported=False`，保留点估计评测。
预测接口只接收五个观测数组；模拟真值另存于 `truth/`，由评测器读取。
`execution.json` 分别记录点预测失败与区间失败、运行时间及支持标记。
旧实验如果未保存区间，需要重新运行 benchmark 才能得到覆盖率。

### 单独生成区间

```bash
python predict.py --input /tmp/ipfn-coverage95-pilot/factual/f0_m0_r0_observed20.npz \
  --output /tmp/ipfn-prediction.npz --interval-level 0.95
```

输出包含 `arm_means`、`ite`、`ate`，以及区间端点
`ite_lower`、`ite_upper`（`N×3`）、`ate_lower`、`ate_upper`（`3`）
和区间标记 `interval_level`、`interval_kind`。
不传 `--interval-level` 时，只输出原有的点预测结果。
