# E1.5 Star v2 — Formal Test 实验说明

## 1. 项目概述

E1.5 Star v2 是针对 Star 网络拓扑开展的因果效应估计与节点响应预测实验。

本实验旨在比较 iPFN-J、两种 Kernel 方法及传统统计基线在 Star 拓扑下的估计表现，重点考察三种因果效应：

- Direct Effect：直接效应
- Spillover Effect：溢出效应
- Total Effect：总效应

实验不仅比较总体因果效应估计误差，还考察中心节点与叶节点的预测差异、不同 Support 条件下的响应预测表现，以及 iPFN-J 的区间覆盖率。

本项目采用冻结的 Candidate v0.3.2 科学协议。正式实验、统计分析及展示产物均已完成独立验收。

### 1.1 v1 与 v2 的关系

v1 是此前的 E1.5 Star 实验版本，其历史材料涉及 400 contexts。v2 是基于 Candidate v0.3.2 FROZEN 建立的独立正式实验版本，正式规模为 200 mechanisms、600 pairs、1,200 contexts。

两个版本的实验身份、协议及统计口径不能混用。v1 历史结果不能作为 v2 Formal Test 的统计证据，也不属于本次 v2 正式比较中的基线；本次七种比较方法见第4节。

## 2. 冻结协议与实验身份

| 项目 | 正式身份 |
|---|---|
| Protocol | Candidate v0.3.2 FROZEN |
| Protocol SHA256 | `fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b` |
| Source Snapshot ID | `99c8b9c9a1e75a9fac4d86d491aaf4505a47cdc5518e11525fc3092e1f73113d` |
| Run ID | `e1_5_star_v2_test_20261031_01` |
| Root Seed | `20261030` |
| Test Cohort Code | `2` |
| RNG | NumPy SeedSequence + PCG64 |

Candidate v0.3.2 是本次实验唯一有效的科学协议基线。实验过程中不修改冻结协议、估计目标、统计阈值或方法比较规则。

## 3. 实验设计

### 3.1 网络结构

采用 Star 拓扑：

- 1 个中心节点（Center）。
- 199 个叶节点（Leaf）。
- 每个 Context 共 200 个节点。
- 每个 Context 有 50 个 Observed Outcomes。

### 3.2 实验规模

| 项目 | 数量 |
|---|---:|
| Mechanisms | 200 |
| Pairs per Mechanism | 3 |
| Total Pairs | 600 |
| Total Contexts | 1,200 |
| Nodes per Context | 200 |
| Observed Outcomes per Context | 50 |
| Data Tasks | 1,800 |
| Requests | 10,200 |
| Failed Attempts | 0 |

每个 Mechanism 对应三个 Pair。统计推断以 Mechanism Block 为独立统计单位，而不是将 Context、Pair 或节点查询直接视为独立机制。

正式实验共完成 10,200 次请求，未出现失败 Attempt。

## 4. 比较方法与估计目标

### 4.1 七种普通比较方法

| 方法 | 说明 |
|---|---|
| State Mean | 状态均值基线 |
| HT | Horvitz–Thompson 估计方法 |
| F-z2 | 冻结的 F-z2 基线 |
| Lin-z2 | 冻结的 Lin-z2 基线 |
| iPFN-J | iPFN-J 模型 |
| kernel-old | 旧版 Kernel 方法 |
| kernel-new | 新版 Kernel 方法 |

七种方法在同一正式实验设计和相同数据条件下进行比较。

### 4.2 三种因果效应

- Direct：直接效应。
- Spillover：溢出效应。
- Total：总效应。

每种普通方法、每种效应均包含 600 个有效 Pair 结果。

Oracle-context 和 Oracle-pair 作为独立参考方法报告，不参与普通七方法 Ranking。

Hájek 仅作为诊断方法，不参与正式 Ranking。

### 4.3 误差指标与比较口径

Pair ATE 误差定义为估计值减去真值。各指标按冻结实现读取和报告：

- RMSE：先计算每个机制内的均方误差，再对机制等权平均并开平方。本次每个机制均有3个有效 Pair。
- Signed Bias：各机制平均误差的均值，保留误差方向。
- Absolute Bias：整体 Signed Bias 的绝对值。
- ACB（Absolute Conditional Bias）：各机制平均误差绝对值的均值，不等同于 Absolute Bias。
- Within Variance：各机制内重复误差的样本方差，再对机制取均值；机制内方差分母为 R−1，本次 R=3。
- Between Variability：机制平均误差之间的样本方差，分母为 K−1，本次 K=200。

RMSE、Bias 和 ACB 使用结果变量的单位；两项 Variance 使用其平方单位。第5节保留正式结果的原有显示精度，不据此重新计算指标。

Ranking 资格按效应分别检查。正式 `rankability.csv` 中的 RANKABLE 表示满足冻结的比较资格，不是综合评分或自动生成的序数排名；下文最低 RMSE 的表述仅针对相应效应的已保存点估计，统计支持另以逐项 Bootstrap CI 说明。

## 5. 七种方法的主要实验结果

### 5.1 Direct Effect

| 方法 | RMSE | Signed Bias | ACB | Within Variance | Between Variability |
|---|---:|---:|---:|---:|---:|
| State Mean | 0.3135 | 0.0091 | 0.1444 | 0.0994 | 0.0321 |
| HT | 0.3758 | 0.0169 | 0.1589 | 0.1506 | 0.0407 |
| F-z2 | 0.3103 | 0.0082 | 0.1449 | 0.0957 | 0.0326 |
| Lin-z2 | 0.3076 | 0.0115 | 0.1429 | 0.0945 | 0.0316 |
| **iPFN-J** | **0.2833** | 0.0027 | 0.1917 | 0.0350 | 0.0572 |
| kernel-old | 0.3112 | -0.0664 | 0.2272 | 0.0255 | 0.0758 |
| kernel-new | 0.2959 | -0.0061 | 0.2114 | 0.0319 | 0.0666 |

iPFN-J 在 Direct Effect 上取得七种方法中最低的 RMSE（0.2833）。

其优势不仅体现在点估计上，也获得了六组逐项 Bootstrap 比较的支持。iPFN-J 相比六种对照方法的 RMSE 均更低，对应的 95% Bootstrap CI 均不包含 0。

需要注意，iPFN-J 的整体 Signed Bias 接近 0，但 ACB 为 0.1917，高于部分传统基线。这说明总体偏差接近零，并不代表所有条件下的条件偏差都较小。

### 5.2 Spillover Effect

| 方法 | RMSE | Signed Bias | ACB | Within Variance | Between Variability |
|---|---:|---:|---:|---:|---:|
| State Mean | 0.0738 | -0.0033 | 0.0308 | 0.0053 | 0.0019 |
| HT | 0.1554 | 0.0016 | 0.0626 | 0.0244 | 0.0079 |
| F-z2 | 0.0725 | -0.0037 | 0.0304 | 0.0050 | 0.0019 |
| **Lin-z2** | **0.0501** | -0.0019 | 0.0216 | 0.0025 | 0.0008 |
| iPFN-J | 0.6210 | 0.0489 | 0.4999 | 0.0086 | 0.3794 |
| kernel-old | 0.6446 | -0.1578 | 0.5090 | 0.0136 | 0.3835 |
| kernel-new | 0.6316 | 0.0494 | 0.5102 | 0.0122 | 0.3903 |

Spillover Effect 是本次实验中传统基线优势最明显的效应类型。

Lin-z2 的 RMSE 为 0.0501，明显低于 iPFN-J 的 0.6210。

四种传统基线与 iPFN-J 的 RMSE 差异均获得逐项 Bootstrap CI 支持。

iPFN-J 与 kernel-old、kernel-new 的差异则没有得到同样明确的统计支持，其对应区间均包含 0。

另外，iPFN-J 及两种 Kernel 方法的 Between Variability 明显较高，提示其跨 Mechanism 的误差表现存在较大差异。

### 5.3 Total Effect

| 方法 | RMSE | Signed Bias | ACB | Within Variance | Between Variability |
|---|---:|---:|---:|---:|---:|
| State Mean | 0.3265 | 0.0058 | 0.1515 | 0.1060 | 0.0361 |
| HT | 0.4421 | 0.0185 | 0.1883 | 0.2061 | 0.0580 |
| F-z2 | 0.3156 | 0.0045 | 0.1467 | 0.0978 | 0.0345 |
| **Lin-z2** | **0.3139** | 0.0096 | 0.1436 | 0.0992 | 0.0325 |
| iPFN-J | 0.6585 | 0.0517 | 0.5104 | 0.0675 | 0.3879 |
| kernel-old | 0.7154 | -0.2242 | 0.5343 | 0.0585 | 0.4246 |
| kernel-new | 0.6788 | 0.0433 | 0.5251 | 0.0700 | 0.4143 |

Total Effect 的表现与 Spillover Effect 相似，传统四种基线均优于 iPFN-J。

Lin-z2 与 F-z2 的 RMSE 分别为 0.3139 和 0.3156，但二者的 Bootstrap CI 包含 0，因此不能认定 Lin-z2 在统计比较上优于 F-z2。

iPFN-J 的 RMSE 则低于两种 Kernel 方法，对应的逐项 Bootstrap CI 均不包含 0。

### 5.4 综合发现

本次实验没有出现某一种方法在所有效应下全面占优的情况。

- Direct：iPFN-J 的 Pair ATE RMSE 最低。
- Spillover：Lin-z2 的 Pair ATE RMSE 最低。
- Total：Lin-z2 的 Pair ATE RMSE 点估计最低，F-z2 次之；二者差异的逐项95% Bootstrap CI 包含0。

这表明方法表现与目标因果效应类型密切相关。Direct Effect 上的优势不能直接推广到 Spillover 和 Total Effect。

## 6. Bootstrap 统计比较

正式分析采用 10,000 次 Mechanism Block Bootstrap。

- 抽样单位：200 个 Mechanisms。
- 每个机制包含 3 个 Pair。
- 每次重抽样以 Mechanism 为整体抽样。
- 七种方法共享机制抽样索引。
- 采用 Percentile 95% CI。

共形成：

- 21 组方法配对比较。
- 63 组 Effect-level 比较。

差值方向定义为：

`RMSE(method_a) - RMSE(method_b)`

正值表示 method_b 的 RMSE 更低。

所有报告的区间均为逐项 95% Bootstrap CI，没有进行多重比较调整。

因此，这些结果不应解释为经过家族错误率控制的整体检验。本项目不额外添加 p-value。

## 7. iPFN-J 的区间覆盖率

### 7.1 总体 Coverage

| 效应 | Covered / Requested | Coverage | 95% Bootstrap CI | Mean Interval Width |
|---|---:|---:|---:|---:|
| Direct | 1166 / 1200 | 97.17% | 96.00%–98.25% | 1.3581 |
| Spillover | 1122 / 1200 | 93.50% | 90.58%–96.00% | 2.5318 |
| Total | 1141 / 1200 | 95.08% | 92.83%–97.00% | 2.8387 |

三种效应的总体 Coverage 点估计均超过冻结协议规定的 90% 门槛。

### 7.2 Center_t 分层诊断

| 效应 | Center_t=0 | Center_t=1 |
|---|---:|---:|
| Direct | 96.83% | 97.50% |
| Spillover | 95.67% | 91.33% |
| Total | 95.17% | 95.00% |

Center_t=1 的 Spillover Coverage 为 91.33%，对应 Bootstrap CI 为 87.50%–94.67%。

该结果作为分层诊断披露，不构成冻结规则下的失败，因为 90% 硬性门槛适用于三个总体 Coverage 指标，而不是分层 CI 下界。

### 7.3 区间解释

必须区分两种区间：

1. Context-level 95% posterior interval：iPFN-J 对单个 Context 效应给出的后验区间。
2. Coverage 95% Bootstrap CI：对经验覆盖率估计不确定性的区间。

Mean Interval Width 仅统计成功返回的区间，本次九项 Coverage 的区间成功率均为 100%。

Mean Interval Width 没有额外 Bootstrap CI。

覆盖率较高不等于点估计误差较低，也不意味着区间越宽越好。

## 8. Center 与 Leaf 节点级诊断

### 8.1 节点级 ITE 预测

目前节点输出能力仅覆盖 iPFN-J、kernel-old 和 kernel-new。四种传统基线不支持相同的节点输出，因此不纳入节点级预测比较。

以下为节点级 pooled RMSE：

| 效应 | 方法 | Center | Leaf | All |
|---|---|---:|---:|---:|
| Direct | iPFN-J | 0.3882 | 0.4119 | 0.4118 |
| Direct | kernel-new | 0.3953 | 0.4329 | 0.4327 |
| Direct | kernel-old | 0.3978 | 0.4483 | 0.4481 |
| Spillover | iPFN-J | 0.0392 | 0.7469 | 0.7451 |
| Spillover | kernel-new | 0.0401 | 0.7650 | 0.7631 |
| Spillover | kernel-old | 0.0410 | 0.7951 | 0.7931 |
| Total | iPFN-J | 0.3978 | 0.8443 | 0.8426 |
| Total | kernel-new | 0.4044 | 0.8569 | 0.8552 |
| Total | kernel-old | 0.4068 | 0.8907 | 0.8890 |

iPFN-J 在三个效应下的节点级 RMSE 均低于两种 Kernel 方法。

但 Spillover 的 Center 与 Leaf 之间存在明显差异。

以 iPFN-J 为例：

- Center RMSE：0.0392。
- Leaf RMSE：0.7469。

Leaf 的 Spillover 预测误差远高于 Center。

Oracle-context 也出现了相似的中心—叶节点差异，说明该现象并非仅由 iPFN-J 架构产生。

以上节点级指标均为描述性结果，没有对应的 Bootstrap CI。

### 8.2 Center-minus-Leaf 效应差异

该指标衡量中心节点效应与平均叶节点效应之差的预测误差。

| 方法 | Direct | Spillover | Total |
|---|---:|---:|---:|
| iPFN-J | 0.3074 | 0.6316 | 0.6493 |
| kernel-new | 0.3231 | 0.6497 | 0.6524 |
| kernel-old | 0.3201 | 0.6814 | 0.6864 |

iPFN-J 在三个效应上的 Center-minus-Leaf RMSE 均为三种模型中的最低值。

该指标与单独的 Center 或 Leaf RMSE 不同，也没有 Bootstrap CI，因此仅作为描述性比较。

## 9. Support 分层诊断

Support 分层使用 `mu_state` 响应预测误差，反映模型在不同节点—状态支持条件下的预测表现。

它不是 Direct、Spillover 或 Total 的因果效应估计误差。

### 9.1 全节点 Support RMSE

| Support 类别 | iPFN-J | kernel-new | kernel-old |
|---|---:|---:|---:|
| Covered | 0.3206 | 0.3443 | 0.3509 |
| Assignment Zero | 0.3400 | 0.3701 | 0.3766 |
| Observation Sampling Zero | 0.3417 | 0.3724 | 0.3785 |
| Conditional Structural Zero | 0.7657 | 0.7772 | 0.8091 |

Conditional Structural Zero 是三个模型误差最高的 Support 类别。

该类别的误差尤其集中在 Leaf 节点：

| iPFN-J 分组 | Covered | Conditional Structural Zero |
|---|---:|---:|
| Center | 0.4627 | 0.5171 |
| Leaf | 0.3197 | 0.7667 |

这说明叶节点的 Conditional Structural Zero 响应预测存在明显困难。

按每种方法的请求数量：

- Covered：60,000。
- Assignment Zero：240,000。
- Observation Sampling Zero：180,000。
- Conditional Structural Zero：480,000。

这些是节点—状态查询数量，不是独立统计样本量。独立统计单位仍然是 200 个 Mechanisms。

Support 结果全部属于描述性诊断，没有 Bootstrap CI。不能据此宣称某一 Support 类别导致了 Spillover 估计误差。

## 10. Oracle 参考结果

Oracle-context 与 Oracle-pair 分别使用不同的信息条件，独立报告，不参与普通七方法 Ranking。

### 10.1 Pair ATE 对比

| 效应 | Oracle-context RMSE | Oracle-pair RMSE |
|---|---:|---:|
| Direct | 0.2734 | 0.232015 |
| Spillover | 0.5991 | 0.000052 |
| Total | 0.6316 | 0.232016 |

Oracle-context 分别对两个 Context 进行推断，再组合得到 Pair ATE。

Oracle-pair 则利用配对层面的共享噪声结构，直接进行联合推断。

两种参考方法的信息条件不同，因此不能将它们视为同一种估计方法的简单版本比较。

Oracle-context 的节点级结果同样显示出明显的 Center–Leaf Spillover 差异：

- Center：0.0384。
- Leaf：0.7247。

Oracle-pair 的 Spillover Pair ATE RMSE 接近零，是本次冻结配对结构下的特殊参考结果。

**Oracle-pair 不是普通方法的无条件理论性能上界。** 它利用了额外的配对信息，其结果不能直接作为普通模型在其他实验条件下应达到的目标。

Oracle 结果均独立报告，不参与普通方法排名。

## 11. 图表与展示产物

正式分析包含十组图表：

1. Pair ATE
2. Method Comparisons
3. J Coverage
4. Node Error
5. Group ATE
6. Center–Leaf
7. Support Strata
8. Runtime
9. Oracle-context
10. Oracle-pair

每组图表均具有对应的来源 metadata，记录数据来源、行身份、分母和实验身份。

在正式分析之外，项目还新增了独立的 v2 Presentation Adapter，用于补充：

- 七方法三效应主结果表。
- J 区间宽度摘要及展示。
- Center、Leaf 与 Center-minus-Leaf 的真实值—预测值对比。

Presentation Adapter 读取已经独立验收的统计结果，不重新运行预测、DGP、正式统计分析或 Bootstrap。

它复用了原始 `make_figures.py` 和 `make_table.py` 的部分样式及辅助函数，但没有调用旧模板的完整主流程，因为原模板的数据接口与 v2 schema 不兼容。

展示层使用独立的 Presentation Acceptance 验收，不替代正式统计分析验收。

## 12. 可复现性与工程验收

### 12.1 正式验收状态

| 验收项目 | 状态 |
|---|---|
| Formal Test Runner | COMPLETE |
| Independent Test Acceptance | PASS |
| Formal Statistical Analysis | COMPLETE |
| Independent Analysis Acceptance | PASS |
| Presentation Acceptance | PASS |

Analysis Independent Acceptance：

- Verified Files：72。
- Method Comparisons：21。
- Effect-level Comparisons：63。
- J Coverage Distributions：9。
- Oracle Separate：True。
- Hájek Ranked：False。
- Scientific Performance Pass Claimed：False。

最后一项表示工程验收没有擅自宣称科学性能达标，并不意味着科学性能验收失败。

Gate 1.6 Presentation Acceptance：

- Targeted Tests：12 PASS。
- FAIL：0。
- ERROR：0。
- SKIP：0。
- Main Results：21 rows / 126 metric values。
- J Interval Summary：9 rows。
- Original Effect Records：10,800。
- Display Points：32,400。
- Presentation Files：16。

以上 Gate 1.6 数量对应外部原始 Presentation 目录及其历史验收。公开交付采用第14节的目录映射，并使用公开脱敏派生 Manifest；该派生文件不是原始 Manifest，也不替代原始 Presentation Acceptance。

公开 Adapter 和对应测试已进行可移植性修复。历史验收收据及派生 Manifest 中的实现哈希仍保留 Gate 1.6 当时的版本身份，不改写为修复后版本，也不据此重新生成展示结果。

### 12.2 运行环境

正式运行环境采用 Python 3.12.10。

主要依赖版本：

```text
numpy==2.1.3
torch==2.8.0
scipy==1.15.3
networkx==3.4.2
torch-geometric==2.6.1
threadpoolctl==3.5.0
matplotlib==3.10.0
```

完整运行身份、源码快照、冻结模型资产及验收收据应与结果文件一起保留，以支持后续独立复验。

原始 `acceptance_report.json` 保留其原始状态；Analysis 独立 PASS 由单独的外部验收收据补充证明，不通过修改原始报告实现。

## 13. 科学结论与研究限制

### 13.1 核心发现

本次实验揭示了 Star 拓扑下不同因果效应类型之间明显的估计性能差异：

1. iPFN-J 在 Direct Effect 上取得最低 Pair ATE RMSE，并获得六组逐项 Bootstrap 比较的支持。
2. 四种传统基线在 Spillover 和 Total Effect 上均优于 iPFN-J。
3. iPFN-J 的节点级预测表现优于两种 Kernel 方法，但节点级预测优势没有转化为所有因果效应上的 Pair ATE 优势。
4. Center 与 Leaf 的 Spillover 节点预测误差存在明显差异。
5. Conditional Structural Zero 的响应预测误差较高，尤其集中在 Leaf 节点。
6. iPFN-J 的三项总体 Coverage 均达到冻结的 90% 目标。

这些结果说明，节点响应预测能力、因果效应估计精度和区间覆盖率是不同的评价维度，不能相互替代。

### 13.2 研究限制

- 本次实验针对冻结的 Star 拓扑和 Candidate v0.3.2 设计，不能直接推广到所有网络结构或真实应用场景。
- Bootstrap CI 为逐项区间，没有多重比较调整。
- 不额外报告未经冻结方案要求的 p-value。
- Node Error、Center–Leaf 和 Support 结果主要为描述性统计，没有额外 Bootstrap CI。
- Support 查询数量不等于独立统计样本量。
- Oracle-context 和 Oracle-pair 的信息条件不同，不应作为普通方法的无条件理论性能上界。
- 四种传统基线不支持与 iPFN-J、Kernel 相同的节点级输出，因此不能进行对应的节点预测比较。
- 高 Coverage 不等于低 RMSE，较宽的区间也不能直接解释为更高的预测精度。

## 14. 结果文件与交付结构

### 14.1 当前公开目录

以下结构依据当前审核工作区的实际文件整理。图表及正式统计 CSV 的逐一文件名在树中省略；正式统计与展示材料分别存放。

```text
iPFN_demo/
├── E1_5_STAR_V2_README.md
├── code/
│   └── sparse_j/
│       ├── star_v2/
│       ├── pfn_social/
│       ├── scripts/
│       ├── models/
│       ├── presentation_v2_adapter.py
│       └── tests/
│           └── test_presentation_v2_adapter.py
└── results/
    └── e1_5_star_v2_test_20261031_01/
        ├── analysis/
        │   ├── tables/
        │   ├── figures/
        │   ├── bootstrap/
        │   │   └── metadata.json
        │   ├── ANALYSIS_REPORT.md
        │   ├── analysis_manifest.json
        │   ├── input_provenance.json
        │   └── acceptance_report.json
        ├── tables/
        │   ├── main_results.csv
        │   ├── main_results.tex
        │   └── j_interval_summary.csv
        ├── figures/
        ├── sources/
        │   └── effects_true_vs_predicted.csv
        ├── reports/
        │   └── PRESENTATION_README.md
        ├── acceptance/
        │   ├── formal_test_independent_acceptance.json
        │   ├── analysis_independent_acceptance.json
        │   ├── presentation_acceptance.json
        │   ├── execution_receipt.json
        │   └── presentation_test_results.json
        ├── provenance/
        │   └── presentation_manifest.public.json
        └── source_delivery/
            ├── source_manifest.json
            ├── source_snapshot_public_subset.zip
            └── SOURCE_SUBSET_MANIFEST.json
```

各目录的用途如下，路径均相对于本 README 所在的 `iPFN_demo/`：

| 目录 | 内容 |
|---|---|
| [analysis/tables/](results/e1_5_star_v2_test_20261031_01/analysis/tables/) | 正式分析公开子集的13份统计结果 CSV，包括 Pair ATE、比较、Coverage、节点、Support 和 Oracle 结果。 |
| [analysis/figures/](results/e1_5_star_v2_test_20261031_01/analysis/figures/) | 十组正式分析图，每组保留 SVG、PNG 和来源 metadata。 |
| [tables/](results/e1_5_star_v2_test_20261031_01/tables/) | 展示用主结果 CSV、LaTeX 表及 J 区间摘要。 |
| [figures/](results/e1_5_star_v2_test_20261031_01/figures/) | 两组展示图：真实值—预测值效应对比及 J 区间宽度，保留 SVG、PNG、PDF 和 metadata。 |
| [sources/](results/e1_5_star_v2_test_20261031_01/sources/) | 展示数据提取副本，保留10,800条原始效应记录及其来源指针。 |
| [reports/](results/e1_5_star_v2_test_20261031_01/reports/) | 原始 Presentation README 的原样副本。 |
| [acceptance/](results/e1_5_star_v2_test_20261031_01/acceptance/) | Formal Test、Analysis、Presentation 的验收记录、正式执行收据及历史展示测试记录。 |
| [provenance/](results/e1_5_star_v2_test_20261031_01/provenance/) | 公开脱敏 Presentation Manifest，保留原始 Manifest 哈希及来源映射。 |
| [source_delivery/](results/e1_5_star_v2_test_20261031_01/source_delivery/) | 原始来源清单、137项公开源码 ZIP 和独立子集清单。 |

### 14.2 完整冻结快照与公开源码子集

| 项目 | 文件数 | 用途 |
|---|---:|---|
| 完整冻结快照 | 140 | 外部归档及完整来源验证。 |
| 公开源码子集 | 137 | 随 PR 公开。 |
| 排除历史文档 | 3 | 涉及本地路径及环境信息，原件保留于外部完整快照。 |

[source_snapshot_public_subset.zip](results/e1_5_star_v2_test_20261031_01/source_delivery/source_snapshot_public_subset.zip) 直接从完整原始快照 ZIP 的成员生成，保留原始相对路径和内容字节，不从 Git 工作树重新构造。仅排除以下三份历史文档：`STAGE2_ACCEPTANCE.md`、`STAGE3_ACCEPTANCE.md`、`STAGE4_SOURCE_AUDIT.md`；它们未进入公开源码子集，原件仍保留于外部完整原始快照。

[source_manifest.json](results/e1_5_star_v2_test_20261031_01/source_delivery/source_manifest.json) 是完整140项原始快照的来源清单原件，记录原始 Snapshot ID 和归档身份，不是改写后的137项清单，也不表示完整原始 ZIP 已公开。

[SOURCE_SUBSET_MANIFEST.json](results/e1_5_star_v2_test_20261031_01/source_delivery/SOURCE_SUBSET_MANIFEST.json) 记录原始快照与公开子集之间的身份映射，包括原始及公开归档 SHA256、排除项，以及137项原始成员、公开成员和 Git 内容的独立身份。Git 文本规范化仅允许严格 CRLF→LF；公开 ZIP 成员始终保留原始字节。

**137项公开子集不具备完整冻结快照验收资格。** 完整来源核验仍依赖外部保存的140项原始快照和匹配 Runtime，不能通过修改冻结 provenance 或历史测试使公开子集冒充完整快照。

### 14.3 统计来源、验收记录与公开边界

| 内容 | 实际文件 |
|---|---|
| 七方法 Pair ATE 指标 | [pair_ate.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/pair_ate.csv) |
| 方法配对比较及逐项 CI | [comparisons.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/comparisons.csv) |
| Ranking 资格 | [rankability.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/rankability.csv) |
| J Coverage、分母及区间宽度 | [j_coverage.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/j_coverage.csv) |
| 节点、中心—叶节点及 Support 诊断 | [node_error.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/node_error.csv)、[center_leaf.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/center_leaf.csv)、[support_strata.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/support_strata.csv) |
| 独立 Oracle 参考 | [oracle_context.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/oracle_context.csv)、[oracle_pair.csv](results/e1_5_star_v2_test_20261031_01/analysis/tables/oracle_pair.csv) |
| 分析解释及 Bootstrap 元数据 | [ANALYSIS_REPORT.md](results/e1_5_star_v2_test_20261031_01/analysis/ANALYSIS_REPORT.md)、[bootstrap/metadata.json](results/e1_5_star_v2_test_20261031_01/analysis/bootstrap/metadata.json) |
| Formal Test 独立验收 | [formal_test_independent_acceptance.json](results/e1_5_star_v2_test_20261031_01/acceptance/formal_test_independent_acceptance.json) |
| Analysis 独立验收 | [analysis_independent_acceptance.json](results/e1_5_star_v2_test_20261031_01/acceptance/analysis_independent_acceptance.json) |
| 历史 Gate 1.6 Presentation 验收 | [presentation_acceptance.json](results/e1_5_star_v2_test_20261031_01/acceptance/presentation_acceptance.json) |
| 正式执行收据 | [execution_receipt.json](results/e1_5_star_v2_test_20261031_01/acceptance/execution_receipt.json) |
| 公开脱敏展示来源记录 | [presentation_manifest.public.json](results/e1_5_star_v2_test_20261031_01/provenance/presentation_manifest.public.json) |

公开结果是完整 Formal Test、Analysis 和 Presentation 原始产物的公开子集与映射交付，不等于完整外部运行目录。完整 T、72文件的验证副本 V、原始 Presentation 目录 P 及未脱敏的历史证据继续在外部归档保留。

公开的 `analysis_manifest.json` 和 `acceptance_report.json` 均保留原始字节；前者登记完整 Analysis 来源集合，不能据此把公开子集视为完整72文件验证副本。独立 Analysis PASS 由单独验收收据证明。公开脱敏 Presentation Manifest 通过原始哈希连接外部原件，不替代历史验收收据。

最终公开文件清单已生成并通过独立完整性核验，详见 `results/e1_5_star_v2_test_20261031_01/PUBLIC_FILE_MANIFEST.json`。源码交付以冻结 Source Snapshot 及匹配 Runtime 为依据；工程验收 PASS 不等于某一方法在全部科学评价维度上优越。

---

**本 README 的核心定位：** 如实呈现 E1.5 Star v2 的正式实验设计、七方法比较结果、节点与 Support 诊断、区间覆盖率及 Oracle 参考结果。所有结论均遵循 Candidate v0.3.2 冻结协议，不将工程验收通过等同于科学性能全面优越。