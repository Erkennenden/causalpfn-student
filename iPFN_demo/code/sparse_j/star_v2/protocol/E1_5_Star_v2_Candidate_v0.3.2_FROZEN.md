# E1.5 Star v2 实验协议 — Candidate v0.3.2（FROZEN／正式冻结版）

**版本**：Candidate v0.3.2（科学协议版本未变；冻结状态更新）  
**日期**：2026-10-30  
**状态**：`FROZEN` — 2026-10-30 经用户明确批准冻结；Stage 1 只读工程审查已通过，不等于代码/数值测试已通过  
**冻结前原稿 SHA256（仅用于版本溯源，不是本文件的 SHA256）**：`8b209fed4602a8310fc16c704f688fc1fddcd2e19a0fab19706b9c25b22c52e1`  
**研究对象**：固定节点数 Star 网络上的四状态因果响应估计  
**固定源码基点**：`fbc0d9e09789248139ff9a201b8f89bf71ae57cf`  
**PR #3 Oracle 参考版本**：`ae82985f7126e220e4c7ce284c7c149473ec8a0b`  
**PR #2 Configuration 参考版本**：`b732fe5f0a5e74facf7ea854956fc84fd6f87881`

> **重要权限声明**：本文件是经用户明确批准的 **FROZEN 科学协议**，其中的科学定义、实验范围、随机身份、统计规则与验收目标均不得未经批准修改。**协议冻结不等于执行授权**：后续代码实施和确定性测试须遵守各阶段单独批准的范围；模型预测、Smoke、Pilot、Formal Test、训练、校准及 Git/PR 操作不能仅凭本文件获得授权，Pilot/Test 尤须分别批准。旧 v1 数据、结果和模型资产保持不变。

------------------------------------------------------------------------

## 0. Candidate v0.3.2 的核心方案

本版本以 Candidate v0.2 为主体，完整保留其 Star DGP、四状态因果目标、配对设计、方法规格、Oracle、Support、评价指标、接口契约、验收矩阵、Pilot/Test 计划和 Codex 实施要求；在此基础上整合 B6 Step 1–5、R3–R7 记录及用户已批准的最终决议。除明确列出的修订外，不改变既有科学定义。

### 0.1 核心实验

| 项目                     | Candidate v0.3.2                                                                                                 |
|--------------------------|----------------------------------------------------------------------------------------------------------------|
| 拓扑                     | 无向、无自环 Star                                                                                              |
| 节点数                   | `N=200`                                                                                                        |
| 中心节点                 | `node 0`                                                                                                       |
| 叶子节点                 | `node 1–199`                                                                                                   |
| 每个上下文观测结果数     | `M=50`                                                                                                         |
| 原始协变量               | `X[200,20]`                                                                                                    |
| 叶子处理                 | iid Bernoulli(0.5)                                                                                             |
| 中心处理                 | 每个 pair 固定为 `T=0` 与 `T=1` 两臂                                                                           |
| 配对共享量               | X、图、机制、叶子处理、float64 原始观测噪声结果、观测节点索引                                                  |
| 主要比较                 | 七种普通方法的配对级 ATE 点估计；每种方法 600 个 Test pair 请求                                                |
| 节点级分析               | J、kernel-old、kernel-new；Oracle-context 单独展示；F-z2/Lin-z2 不要求节点级主结果                             |
| 区间分析                 | 冻结 J 的单上下文三个总体 ATE 边际 95% 后验可信区间                                                            |
| Oracle                   | Oracle-context 与 Star-specific Oracle-pair，均独立报告、不参与普通方法排名                                    |
| Pilot                    | 5 mechanisms × 2 pairs = 10 pairs = 20 contexts                                                                |
| Test                     | 200 mechanisms × 3 pairs = 600 pairs = 1,200 contexts                                                          |
| RNG                      | root seed `20261030`；protocol code `2026103001`；Pilot/Test cohort code `1/2`；NumPy `SeedSequence` + `PCG64` |
| 正式 Star pair HT 权重   | `8`（`r=0.25`、`q=0.5`）                                                                                       |
| 四节点 Fixture A HT 权重 | `4`（`r=0.5`、`q=0.5`；48 种穷举组合，仅作确定性测试）                                                         |
| 原始 factual outcomes    | float64 保存；J adapter 内部转换为 float32；v1 不修改                                                          |
| J 区间验收               | Direct、Spillover、Total 三种总体 ATE 覆盖率分别至少 `90%`；失败区间计入分母                                   |
| 独立统计单位             | mechanism；Bootstrap 按完整机制块重抽样                                                                        |
| 排除项                   | 不新增 Mechanism OOD、人工 Support pressure 或 Calibration cohort                                              |
| 重新训练                 | 不允许                                                                                                         |
| 状态                     | `FROZEN`；Stage 1 只读审查通过；未执行 Pilot/Test                                                              |

### 0.2 七种普通方法

1.  State Mean
2.  HT
3.  F-z2
4.  Lin-z2
5.  kernel-old
6.  kernel-new
7.  iPFN-J

Hájek 只作为 State Mean 的等价性诊断，不进入正式七方法排名。Oracle-context 和 Oracle-pair 不进入七种普通方法排名，也不能被称为必然性能上界。

### 0.3 本版最重要的最终决议

1.  **v2 使用独立随机身份。** `root_seed=20261030`、`protocol_code=2026103001`、Pilot cohort code `1`、Test cohort code `2`，使用 NumPy `SeedSequence` 与 `PCG64`。
2.  **正式 Star pair HT 权重为 8。** `r=50/200=0.25`、pair-state inclusion `q=0.5`；四节点 Fixture A 的 48 种穷举组合中 HT 权重为 4，不得把 8 写成通用常数。
3.  **原始 factual outcomes 使用 float64 保存。** J 只在自己的 adapter 边界转换为 float32；不得改写 float64 原始文件，也不得修改 v1 数据。
4.  **七种普通方法保持不变。** State Mean、HT、F-z2、Lin-z2、kernel-old、kernel-new、iPFN-J；Hájek 只作 State Mean 等价性诊断。
5.  **F-z2/Lin-z2 仅参与 Pair ATE 点估计比较。** 不强制它们输出或参与节点级 `mu`/ITE 主结果。
6.  **Oracle-context 与 Star-specific Oracle-pair 分开报告。** 两者均不进入七种普通方法排名。
7.  **J 的三个总体 ATE 区间分别验收。** Direct、Spillover、Total 的单上下文名义 95% 区间覆盖率各自至少为 90%；失败区间留在各自覆盖率分母中。
8.  **mechanism 是独立统计单位。** 每个机制内 3 个 pair 是嵌套重复；Bootstrap 必须按完整机制块重抽样。
9.  **失败任务全部保留。** 不静默删除失败，不以成功子集指标冒充完整任务表现；任一必要 pair arm 失败时，该 pair 的相应估计失败。
10. **不新增 Mechanism OOD、人工 Support pressure 或 Calibration cohort。** 主实验仅执行注册的固定 Star 新机制 IID-task 设计；自然出现的 Support 仍须完整记录。
11. **模型 provenance 限制继续保留。** 资产哈希证明字节身份，不证明训练/验证/选择机制独立；不得过度声称严格 unseen-topology generalization。

### 0.4 本版依据与来源完整性

Candidate v0.3.2 以已检索并读取的以下资料为依据：

- `E1_5_Star_v2_Candidate_v0.2.md`：原始主体文档，完整读取（1,718 行）；
- `B6_Step1-5_R3-R7_审查与决议记录.md`：完整读取；其中 B6 Step 1–5 是设计审查记录，不是已执行的代码测试报告；
- `estimation任务分配.pdf`：原始 PDF 全部 3 页已直接读取；本版未以 TXT 提取件冒充原始 PDF。

本版保留原任务要求、固定源码基点及其文件哈希审计记录。当前资料没有提供模型训练/验证/选择机制的完整 provenance，因此该限制继续明确披露，不作推断补全。

**学长两次发言的证据等级：** 目前没有取得可逐字核对的两次发言原始截图/完整文字。依据此前对发言截图的对话解读记录，项目实施交付需关注 RMSE、absolute Bias、Variance、95% ATE 区间覆盖率目标（至少 90%）、`make_figures.py`/`make_table.py`、GitHub PR、代码/结果/说明与复现。本版按这些**已转述要求**补足规格，但不把这些概括冒称为学长逐字原话；若日后原始发言与转述冲突，应先做差异审计，不得在冻结后私自改动科学结果。原始任务 PDF 的正式要求为一级核对依据。

------------------------------------------------------------------------

## 1. 研究目的、问题与声明边界

### 1.1 研究目的

在固定 N=200 的 Star 网络中，沿用共同项目的 DGP、处理设计、连续结果结构方程和四状态真值定义，评估冻结 iPFN-J 及多种 baseline 对以下目标的估计表现：

- 四状态平均响应 `mu_i(t,s)`；
- Direct、Spillover、Total 三种节点效应；
- 全图 ATE；
- 中心节点与叶子节点的节点级响应误差和效应误差；
- 固定中心处理造成的 context-level Support 差异；
- 冻结 J 的名义 95% ATE 后验可信区间的经验覆盖表现。

### 1.2 研究问题

1.  在相同 Star DGP 和配对事实数据下，J 的全图 ATE 点估计误差与 State Mean、HT、F-z2、Lin-z2、kernel-old、kernel-new 相比如何？
2.  J 和两个 kernel 对中心节点与叶子节点的四状态响应、ITE 误差是否不同？
3.  固定中心处理时，中心与叶子的设计支持和经验支持有何差异？支持缺失与模型误差之间呈现什么描述性关系？
4.  冻结 J 的三个名义 95% 边际 ATE 后验可信区间，在新生成 Star 机制上的经验覆盖率是否分别达到 90% 目标？

### 1.3 声明边界

本实验是固定 Star 拓扑上的估计评测，不自动支持以下更强结论：

- 不证明模型在所有未见图拓扑上均能泛化；
- 不证明 Support 本身对误差具有因果作用；
- 不证明 Oracle 是理论性能上界；
- 不把 posterior credible interval 的名义 95% 解释成频率学意义上的 95% 覆盖保证；
- 不把 1,200 个上下文、600 个 pair 或 199 个叶子当作 1,200、600 或 199 个独立机制。

若模型训练/验证/选择来源仍不能核实，结果应表述为“在独立注册的 Star v2 新生成机制 cohort 上对冻结模型进行评估”，不得声称训练数据与测试机制的完全独立性已得到证明。

### 1.4 与原始任务及共同实验切分的对应关系

原始 `estimation任务分配.pdf` 的共同实验标准列出 IID-task、结构 OOD、机制 OOD 和 Support 压力测试；但 E1.5 的具体责任描述是：**在固定节点数下构造星图，沿用现有 DGP 与训练评测程序，比较中心节点与叶子节点的估计误差、效应差异及支持情况。** 本 Candidate 严格围绕这一 E1.5 责任范围制定，不擅自把项目级共同切分全部扩展成 E1.5 的额外实验 cohort。

| 共同实验概念     | E1.5 Candidate v0.3.2 的处理                                                                                                               | 是否新增独立 cohort                      |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| IID-task         | 固定 N=200 Star、共同 DGP 先验、独立注册的新机制和随机流；构成本协议主实验                                                               | **是：仅 Pilot 与 Test 两个既定 cohort** |
| 结构 OOD         | Star 是 E1.5 被分配的拓扑；是否相对模型训练图结构构成严格 OOD，取决于尚未完全核实的模型 provenance。跨拓扑结论须结合 E1 项目其他拓扑结果 | 否；不另造结构扰动 cohort                |
| 机制 OOD         | 改变 Direct、Spillover 或噪声机制分布不属于本版批准范围                                                                                  | **否**                                   |
| Support 压力测试 | 不人为删除状态、不降低指定状态概率、不增加人工 observation mask；仅记录固定设计自然产生的 population/design/empirical Support            | **否**                                   |
| Calibration      | 不新增 Calibration cohort，不新增 `j-calibrated`                                                                                         | **否**                                   |

因此，本版的正式实验规模只有：Pilot `5 mechanisms × 2 pairs`，Test `200 mechanisms × 3 pairs`。不得将机制 OOD、人工 Support pressure 或 Calibration 设计混入这些任务数，也不得利用 Test 结果事后校准、改种子或改变阈值。

固定 Star 主实验的 Support 分析仍是必要组成部分：报告自然设计下的 `population_support`、`design_support`、`empirical_support_mask`、Support 原因及中心/叶子分组计数。这里的 Support 分析不是人为 Support 压力实验，也不改变处理分配、观测抽样或目标参考分布。

------------------------------------------------------------------------

## 2. 原始任务要求与固定源码审计

### 2.1 原始任务要求映射

| 原始要求                                 | v2 对应设计                                          |
|------------------------------------------|------------------------------------------------------|
| 固定节点数构造 Star，沿用现有 DGP        | N=200；复用共同机制、隐藏特征、系数先验和结构方程    |
| 比较中心与叶子的误差、效应和 Support     | 单独报告 center、leaf、all 三组                      |
| 按完整模拟任务划分数据                   | mechanism 是独立因果世界；不拆分节点或 pair          |
| 固定图实验使用独立生成任务               | 新 Star v2 RNG cohort，与历史 dev、pilot 分离        |
| 简单基线与带 X 结果回归                  | State Mean、HT、F-z2、Lin-z2                         |
| 图/超图结果模型                          | kernel-old、kernel-new                               |
| 主结果包含 Oracle、简单基线、PFN         | 七种普通方法 + 独立 Oracle 参考                      |
| 相同 target_id、节点顺序、`[t,s]` 轴顺序 | 统一 TaskSpec / EstimationRequest / EstimateBundle   |
| Test 真值不得进入预测或调参              | 预测输入严格隔离，truth 只交给 evaluator             |
| empirical support 不足要标外推           | 独立保存 population/design/empirical support         |
| 交付依赖、命令、配置、随机种子、原始表格 | manifest、source snapshot、CSV/NPZ、README、验收报告 |

### 2.2 固定代码基点

仓库：`Erkennenden/causalpfn-student`  
个人 Fork：`imcoraczy-ship-it/causalpfn-student`  
固定基点：`fbc0d9e09789248139ff9a201b8f89bf71ae57cf`

本次只读审查了以下固定版本源码：

| 文件                                                            | 固定版本文件 SHA                           |
|-----------------------------------------------------------------|--------------------------------------------|
| `iPFN_demo/code/sparse_j/scripts/e1_5_star.py`                  | `3f5abcbdc4a90180134decfecfac8af143e2c657` |
| `iPFN_demo/code/sparse_j/scripts/e1_5_star_analysis.py`         | `fc0f615d17d965a485039b8626c9f4906d44a24b` |
| `iPFN_demo/code/sparse_j/pfn_social/phase3_hierarchical.py`     | `3cd41a5d0a862419463f4935dbdfd6fb7c16d69b` |
| `iPFN_demo/code/sparse_j/pfn_social/phase5_large_network.py`    | `7f1eb78eb4f86d39b98c661cca9cf21617ee5f2a` |
| `iPFN_demo/code/sparse_j/pfn_social/phase6_limited_outcomes.py` | `fd91412ceb0e0c8999ff6801d53dc08be478a166` |
| `iPFN_demo/code/sparse_j/pfn_social/phase6_sparse_classical.py` | `370cea08b0743edbb93619563ff945ab2e3f62a4` |
| `iPFN_demo/code/sparse_j/pfn_social/phase5_meta_bayes.py`       | `42c7b8f4547a32197dd270f886094c75865cb0c6` |
| `iPFN_demo/code/sparse_j/pfn_social/sparse_frozen_gaussian.py`  | `292fb9789310d4623699adcd521015ce3df4e0ea` |
| `iPFN_demo/code/sparse_j/predict.py`                            | `8acffe9ce6219358a8134e8842ec1d07839fab54` |
| `iPFN_demo/code/sparse_j/pfn_social/intervals.py`               | `e6827ebb736ee3db1a911b1334361cd49cbfa4d1` |
| `iPFN_demo/code/sparse_j/experiment.json`                       | `3d68d61f098b770f36ea7d85c5a4425ef3ce134f` |
| `iPFN_demo/code/sparse_j/regional_selected.json`                | `9bbb79e249587d7ced9eef06ddd4452bc5af0056` |
| `iPFN_demo/code/sparse_j/regional_inference.py`                 | `e4959a2520795a63b422bcd043bb2c214534e997` |
| `iPFN_demo/code/sparse_j/scripts/phase4_strong_controls.py`     | `eeb458ab568489cb793fdbadb6f787d34e4fbbd6` |
| `iPFN_demo/code/sparse_j/scripts/phase5_frozen_controls.py`     | `8b91070f9299b2fef7ce3db0c0464a26c1a116ce` |
| `iPFN_demo/code/sparse_j/evaluate.py`                           | `faa5b43a27d23f543d7fddbaff818661c89c5287` |

### 2.3 发现一：当前 E1.5 生成器仍使用旧 dev split

固定版本的 `e1_5_star.py` 调用 `generate_mechanism(0, mechanism_id)`。该函数来自 `phase6_limited_outcomes.py`，其 `configuration()` 使用：

- `base_setting_id=0`
- `N=200`
- `split="dev"`
- `split_seed=331029`

固定版本 `LargeNetworkConfig` 只允许 `smoke/train/dev`，并明确提示 final split 需要新注册协议。

历史 PR #4 的 E1.5 结果采用 20 mechanisms × 10 pairs × 2 center contexts = 400 contexts；J 的历史覆盖率为 Direct 96.50%、Spillover 89.75%、Total 93.25%，其中 Spillover 未达到 90% 目标。该结果属于旧 dev 机制生成路径下的历史证据，不能充当 v2 独立 Test，也不能用于事后调节 v2 的阈值或随机种子。

因此：

- 不能仅更换 mechanism ID 就声称已创建独立 Test；
- 不能继续把旧 `dev` 生成流包装成 v2 独立 cohort；
- 不能覆盖或修改 v1 结果；
- v2 应建立独立 `StarV2Config`、独立 RNG 命名空间，并复用共同 DGP 的纯函数。

v2 应复用 `BASE_SETTINGS[0]` 的 DGP 参数、`common_projection()`、`hidden_features()`、`coefficient_prior()`、`_components()`、`structural_outcome()` 和 `arm_mean_exposure()`，但不再通过旧 `generate_mechanism()` 的 dev 配置来创建新机制。

### 2.4 发现二：旧版 DesignBased 不能直接用于固定中心配对设计

固定版本 `phase6_sparse_classical.py` 的 HT/Hájek/F-z2/Lin-z2 使用单上下文 propensity：

`majority_arm_propensities(degree, treatment_prob=0.5)`

并按单次全节点 iid Bernoulli 处理设计构造权重。

但 v2 每个 pair 中中心处理固定为 0 和 1，两个上下文共享叶子处理、噪声和观测索引。单上下文 propensity 不能直接代表 pair 的联合 inclusion probability。

此外，旧代码的整数状态编码为 `state = 2 * exposure + treatment`，其顺序是 `00,10,01,11`，并使用 `CONTRASTS=((1,0),(3,1),(3,0))`。v2 的标准 `mu[N,2,2]` flatten 顺序则是 `00,01,10,11`。适配器必须显式转换，不能直接复用索引。

因此 v2 必须实现 pair-level adapter。旧代码可以作为特征设计、SVD、OLS 和输入校验的参考，但不能不加修改地直接调用旧单上下文 HT。

### 2.5 发现三：Oracle-context 的可用参考实现

PR #3 参考 commit：`ae82985f7126e220e4c7ce284c7c149473ec8a0b`

参考文件：`iPFN_demo/code/sparse_j/pfn_social/e14_oracle.py`  
该文件 SHA：`6551caa25bf7ad9e3837f5a2951c645db0f2e8e0`

该实现使用真实隐藏表征、已知系数先验和已知噪声方差，但不使用真实抽取的 beta、真值响应或未观测结果。它可作为 Oracle-context 的参考实现，不要求 v2 原样照搬。

### 2.6 发现四：当前固定代码并未提供完整统一 TaskSpec API

原始任务文档规定统一调用链：

`estimate_effects(task, identification, model) -> EstimateBundle`

但固定 commit 中可见的 baseline 入口主要是 `phase6_sparse_classical.py` 等适配器，并没有一个已经完整实现的通用 TaskSpec / EstimationRequest / EstimateBundle 接口。

因此 v2 必须建立自己的明确适配契约，或者对接项目后续提供的统一接口；不得在报告中声称当前旧 Notebook/benchmark 已经完整接入统一识别模块。

### 2.7 发现五：Oracle-pair 需要 float64 原始事实结果

当前 v1 `make_pair()` 将两臂事实 Y 分别转为 float32。由于分别量化，两个结果相减后不能保证共享噪声严格抵消。

v2 的模拟器应以 float64 生成和保存原始事实 `Y_obs`，并在需要时由 J/kernel 适配器内部转为 float32。Oracle-pair 的差分约束必须使用原始 float64 数据并通过预先规定的数值容差检查。

------------------------------------------------------------------------

## 3. DGP、网络与因果目标

### 3.1 Star 网络

固定无向、无自环 Star：

- 节点集合：`V={0,...,199}`
- 中心节点：`0`
- 叶子节点：`1,...,199`
- `A[0,i]=A[i,0]=1`，对所有叶子 i 成立
- 其他邻接元素为 0

必须满足：

- `A=A.T`
- `diag(A)=0`
- `A` 为二元矩阵
- `degree[0]=199`
- `degree[i]=1`，i=1,...,199
- 节点顺序固定，不能重排

### 3.2 处理与暴露

对节点 i：

\[ E_i(T)=\frac{(AT)_i}{d_i} \]

\[ S_i(T)=\mathbf 1\{E_i(T)>0.5\} \]

其中 `T_i` 是节点自身处理，`S_i` 是二元同伴暴露状态。

Star 下：

- 中心节点的暴露是 199 个叶子处理比例是否严格大于 0.5；
- 每个叶子的暴露等于中心节点处理，即 `S_i=T_0`。

### 3.3 共同 DGP

沿用固定基点 `phase3_hierarchical.py` 的结构方程。记：

- `h_i`：节点隐藏表征；
- `hbar_i=(A h)_i/d_i`：邻居隐藏表征均值；
- `e_i=(A T)_i/d_i`：邻居处理比例；
- `j_i=(A(T⊙h))_i/d_i`：邻居处理加权隐藏表征；
- `b_i,tau_i,gamma_i,eta_i`：由共同 25 维系数生成的节点结构分量。

连续结果：

\[ Y_i=b_i+T_i\tau_i+(\gamma_i+T_i\eta_i)e_i +j_i^\top\beta_J+T_i j_i^\top\beta_{TJ}+\epsilon_i \]

其中 `beta_J=beta[19:22]`，`beta_TJ=beta[22:25]`。

v2 复用 base setting 0（`shared_nonlinear`、noise_sd 由固定配置给出，当前源定义为 1.0），保留共同的 20 维原始 X、隐藏表征映射、系数先验与结构方程，只把图替换成 Star。

### 3.4 四状态目标

保留共同项目的参照分配分布：

\[ T_{-i}\overset{iid}{\sim}\operatorname{Bernoulli}(0.5) \]

目标：

\[ \mu_i(t,s)= E_\pi\left[ Y_i(t,T_{-i}) \mid S_i(t,T_{-i})=s,G,X,\mathcal M \right] \]

其中 `M` 是固定机制，包括隐藏表征和已生成结构系数。目标是条件于暴露状态的结构平均响应，不包含观测噪声。

标准接口：

- `mu.shape == [N,2,2]`
- 轴 1：自身处理 `t`
- 轴 2：同伴暴露 `s`
- 节点顺序与 `TaskSpec.node_ids` 一致

规范状态顺序：

| 状态 | 数组索引    |
|------|-------------|
| 00   | `mu[:,0,0]` |
| 01   | `mu[:,0,1]` |
| 10   | `mu[:,1,0]` |
| 11   | `mu[:,1,1]` |

若为兼容旧模型输出保存 `[N,4]`，必须同时写入 `arm_order=["00","01","10","11"]`，并在适配器中显式 reshape/转换。

固定版本 `phase6_sparse_classical.py` 使用的旧状态整数编码为 `2*s+t`，其 flatten 顺序是 `00,10,01,11`；v2 canonical flatten 使用 `2*t+s`，顺序是 `00,01,10,11`。旧 baseline 对比索引 `((1,0),(3,1),(3,0))` 仅在旧编码下成立。任何 adapter 都必须通过显式 mapping 转换，禁止把两套索引直接混用。

### 3.5 真值计算

记：

\[ \bar e_{d,s}=\mathbb E_\pi\!\left[E_i(T)\mid S_i(T)=s,d_i=d\right] \]

\[ \bar h_i=(Ah)_i/d_i \]

则共同 DGP 的四状态真值为：

\[ \mu_i(t,s)= b_i+t\tau_i+ \bar e_{d_i,s} \left[ \gamma_i+t\eta_i+ \bar h_i^\top(\beta_J+t\beta_{TJ}) \right] \]

`arm_mean_exposure(degree, 0.5, s)` 是计算 `\bar e_{d,s}` 的固定函数。该公式与 Star 结构方程相容：条件于高/低暴露后，邻居处理向量在邻居集合内可交换，故被处理邻居隐藏表征之和的条件期望等于条件平均处理比例乘以邻居隐藏表征均值。

真值计算必须满足：

- shape `[200,2,2]`；
- 四状态顺序正确；
- 不含观测噪声；
- 与结构方程的独立枚举/数值核验一致；
- `Total = Direct + Spillover`。

### 3.6 效应与全图 ATE

\[ D_i=\mu_i(1,0)-\mu_i(0,0) \]

\[ S_i^\star=\mu_i(1,1)-\mu_i(1,0) \]

\[ Q_i=\mu_i(1,1)-\mu_i(0,0) \]

分别为 Direct、treated Spillover、Total。

恒等式：

\[ Q_i=D_i+S_i^\star \]

全图有限总体 ATE：

\[ \tau_{m,k}=\frac{1}{200}\sum_{i=0}^{199}\theta_{m,i,k} \]

其中 `k` 为 Direct、Spillover、Total。

**全图 ATE 必须对全部 200 个节点取平均，不得只对 50 个观测节点取平均。**

中心/叶子组效应另定义：

\[ \tau_{m,g,k}=\frac{1}{|g|}\sum_{i\in g}\theta_{m,i,k} \]

`g=center` 仅 node 0；`g=leaf` 为 node 1–199；`g=all` 为全部 200 节点。

------------------------------------------------------------------------

## 4. 配对实验设计与数据注册

### 4.1 Pair 结构

每个机制 m、重复 r 生成一个 pair，包含两个上下文：

- `center_t=0`
- `center_t=1`

两臂共享：

- 同一个机制、X、Star adjacency、h、beta、truth；
- 同一组 199 个叶子处理；
- 同一节点的观测噪声；
- 同一组 50 个观测节点索引。

两臂只改变中心节点处理值。

每个上下文包含 50 条观测结果记录；一个 pair 合计 100 条结果记录，但只有 50 个不同节点。两臂高度相关，不能视为 100 个独立节点或两个独立机制。

### 4.2 观测抽样

- 每个上下文固定 `M=50`；
- 从 200 个节点中均匀、不放回抽样；
- 同一个 pair 的两个上下文共享同一观测索引；
- 观测索引使用独立 RNG channel，不能与处理分配或噪声共用随机流。

观测节点包含概率：

\[ r_i=M/N=50/200=0.25 \]

### 4.3 新 cohort 规模

| Cohort            | 机制数 | 每机制 pair | Pair 数 | Context 数 | 用途                                             |
|-------------------|-------:|------------:|--------:|-----------:|--------------------------------------------------|
| Historical v1 Dev |     20 |          10 |     200 |        400 | 只读历史证据，不作 v2 Pilot/Test，不修改原始数据 |
| v2 Pilot          |      5 |           2 |      10 |         20 | 接口、数值、完整性、资源检查；须另行获得执行授权 |
| v2 Test           |    200 |           3 |     600 |      1,200 | 最终留出评估；须在协议和实现冻结后另行授权       |

本版不设置 v2 Calibration cohort，不新增 `j-calibrated`，不增加 Mechanism OOD cohort，也不增加人工 Support pressure cohort。冻结 J 的原始区间直接进入 Test。Pilot 只用于预先规定的工程与数值验收，不用于区间校准、模型选择或阈值调整。

### 4.4 新 RNG 身份

v2 不得复用 v1 的 `DEFAULT_SEED=20261024`、`REPEAT_NAMESPACE=2026102501`、旧 `dev` split 或旧机制 ID 作为 v2 随机身份。v2 的以下注册值已获用户明确批准，作为 Candidate v0.3.2 的协议参数：

| 字段                | 已批准值             |
|---------------------|----------------------|
| `protocol_id`       | `e1_5_star_v2`       |
| `root_seed`         | `20261030`           |
| `protocol_code`     | `2026103001`         |
| Pilot `cohort_code` | `1`                  |
| Test `cohort_code`  | `2`                  |
| Bit generator       | `PCG64`              |
| 派生器              | NumPy `SeedSequence` |

派生规则固定为：

``` python
seed_sequence = np.random.SeedSequence([
    root_seed,
    protocol_code,
    cohort_code,
    mechanism_id,
    channel_code,
    pair_repeat_id,
])
rng = np.random.Generator(np.random.PCG64(seed_sequence))
```

不得使用 Python 运行时随机化的 `hash()` 生成种子。每个随机流必须记录完整 identity、`SeedSequence` entropy/spawn metadata、bit generator、channel code 及对应任务 ID。

固定 channel code：

| Channel           | Code | 作用                   |
|-------------------|-----:|------------------------|
| population        |    1 | 20 维总体均值          |
| x                 |    2 | 节点 X 噪声            |
| coefficient       |    3 | 25 维 beta 抽样        |
| leaf_assignment   |    4 | pair 内 199 个叶子处理 |
| noise             |    5 | pair 内 200 个节点噪声 |
| observation_index |    6 | pair 内 50 个观测节点  |
| bootstrap         |    7 | 机制块 Bootstrap       |

机制生成通道 `population/x/coefficient` 使用 `pair_repeat_id=0`；pair 通道使用实际重复 ID。Pilot mechanism ID 为 0–4，Test mechanism ID 为 0–199；不同 cohort code 保证两组身份空间区分。共同 projection 继续使用固定源码中的 `REPRESENTATION_SEED=2026101604`，并记录 projection hash；它不是每个机制重新抽取的参数。

**执行边界：**上述随机身份注册已随本协议正式冻结。DGP 的工程实施及确定性夹具测试须遵守获准的阶段范围；正式 Pilot/Test 数据生成与运行仍分别需要独立执行授权。

### 4.5 StarV2Config 生成器要求

新模块应建立独立 `StarV2Config`，复用共同 DGP 纯函数，但不调用旧 `phase6_limited_outcomes.configuration()` 的 `split="dev"` 机制生成入口。

新机制生成应与固定 base setting 0 保持一致：

1.  读取 base setting 0 的 representation/noise 配置；
2.  按独立 RNG channel 生成 `population ~ N(0, 0.8^2 I_20)`；
3.  按固定公式生成 `X = population + 0.6 * Normal(0,1)`，shape `[200,20]`，保留现有 float32 X 约定；
4.  使用固定 `common_projection()`（`REPRESENTATION_SEED=2026101604`）和 `hidden_features()` 生成 h；
5.  从 `coefficient_prior()` 抽取 25 维 beta；
6.  构造确定性的 Star adjacency；
7.  使用共同 `_components()` 和 `structural_outcome()`；
8.  保存机制配置、随机身份、X/h/beta/truth 哈希。

pair 生成：

1.  用独立 `leaf_assignment` channel 抽取 199 个 iid Bernoulli(0.5)；
2.  用独立 `noise` channel 抽取 200 个 `N(0, noise_sd²)`；
3.  用独立 `observation_index` channel 抽取 50 个节点；
4.  构造中心 T=0 与 T=1 两臂；
5.  两臂使用相同 noise、leaf assignment 和 observation index；
6.  在 float64 中计算两臂原始 factual outcomes 并加同一 noise；原始结果以 float64 保存，不能在保存前统一降为 float32。
7.  只将 `outcomes_observed[50]` 交给预测器；该规范输入保持 float64。
8.  iPFN-J 仅在自己的 adapter 内将收到的 `outcomes_observed` 转换为 float32 后送入冻结模型；转换不得覆盖或回写原始 float64 数据。其他方法按各自已冻结接口处理，不得擅自改变共同原始数据。

完整 Y、beta、mu、ITE、ATE 仅保留在 simulator/evaluator 隔离路径，不能放进普通 predictor 输入。J 的 dtype 转换只发生在模型适配边界。v1 历史数据、结果和资产均不修改。

------------------------------------------------------------------------

## 5. 配对设计概率与 HT 数学核验

### 5.1 每个节点状态在 pair 中的 inclusion probability

记 pair 的两个中心处理为 `c=0,1`，节点 i 的实际状态为 `Z_i^(c)=(T_i^(c),S_i^(c))`。

对任意固定状态 `z=(t,s)`，在一个 pair 中：

- 中心节点：中心自身处理在两臂间切换；其暴露由 199 个叶子处理决定。由于 199 为奇数且叶子处理 iid Bernoulli(0.5)，`P(S_0=s)=0.5`。给定 S=s，pair 中恰有一臂满足自身处理 t。
- 叶子节点：其暴露等于中心处理，因此两臂分别得到 S=0 和 S=1；叶子自身处理在 pair 内共享且服从 Bernoulli(0.5)。给定 T_i=t，pair 中恰有一臂满足状态 (t,s)。

因此，对每个节点、每个四状态：

\[ q_i(z)=P\left(\sum_{c=0}^1 \mathbf 1\{Z_i^{(c)}=z\}=1\right)=0.5 \]

每个节点在一个 pair 中对同一状态最多出现一次。

结合观测抽样：

\[ P(R_i=1,\ z\text{ 在 pair 中出现})=r_iq_i(z) =0.25\times0.5=0.125 \]

故正式 Star pair-level HT 权重：

\[ w_i(z)=\frac{1}{r_iq_i(z)}=8 \]

等价地，若先在 pair 的两臂中均匀选取一个 context，则单个 context 的状态边际概率为 `\pi_i^{ctx}(z)=0.25`，相应的两臂平均写法为 `1/(2 r_i \pi_i^{ctx}(z))=8`。两种写法等价，不能重复或遗漏两臂平均中的因子 2。

这里的 `q_i(z)=0.5` 是“该状态在 pair 两臂中恰好出现一次”的概率，不是单个固定中心上下文下四个状态的概率。正式 Star 的 `r=0.25`、`q=0.5`、联合包含概率 `0.125` 和 HT 权重 `8` 必须成套使用。

### 5.2 HT 状态均值估计量

定义：

- `R_i=1`：节点 i 被选入共同 50 节点观测集合；
- `I_i(z)=1`：节点 i 在 pair 两臂中恰有一臂实现状态 z；
- `Y_i(z)`：记作匹配状态时的事实结果；更严谨的实现以两臂 `Y_i^{(c)}` 与指示函数直接求和，不需要对没有匹配状态的节点定义虚构事实值。事实结果随叶子分配与噪声变化，不是固定的节点潜在状态常数。

则：

\[
\widehat{\mu}^{HT}_z
=\frac{1}{N}\sum_{i=0}^{N-1}\sum_{c=0}^{1}
\frac{R_i\mathbf{1}\{Z_i^{(c)}=z\}Y_i^{(c)}}{r_iq_i(z)}
=\frac{8}{200}\sum_{i=0}^{199}\sum_{c=0}^{1}
R_i\mathbf{1}\{Z_i^{(c)}=z\}Y_i^{(c)}.
\]

效应由四个状态均值之差得到。

在随机处理与随机观测设计下：

\[
\mathbb{E}_{\mathrm{assign,obs,noise}}[\widehat{\mu}^{HT}_z\mid\mathcal M]
=\frac{1}{N}\sum_{i=0}^{N-1}\mu_i(z).
\]

HT 在**对叶子随机分配、观测抽样与均值为零的观测噪声共同取期望**时，对固定机制的四状态目标均值设计无偏。由于共同 DGP 的事实结果还依赖连续邻居处理比例和干扰项，证明/实现时应使用每个 node-state 在给定状态条件下的**结构结果条件均值**；不能未经条件化便把某次事实结果视为固定的 `Y_i(z)`。这种无偏性不保证任何单次 pair 的估计误差为零。

如果某个状态在观测样本中没有记录，HT 的求和项可能为 0；这不意味着该状态真实响应为 0。必须同时记录 empirical support=false 和具体原因。

### 5.3 State Mean 与 Hájek

State Mean：

\[
\widehat{\mu}^{SM}_z=
\frac{\sum_{i=0}^{N-1}\sum_{c=0}^{1}R_i\mathbf{1}\{Z_i^{(c)}=z\}Y_i^{(c)}}
{\sum_{i=0}^{N-1}\sum_{c=0}^{1}R_i\mathbf{1}\{Z_i^{(c)}=z\}}.
\]

若分母为 0，**该状态均值**不可估计，返回状态级 `NOT_ESTIMABLE`，不得填 0。但 Pair ATE 按每个已请求效应独立判断：Direct 只需要 `10/00`，Spillover 只需要 `11/10`，Total 只需要 `11/00`。`01` 虽属于规范四状态和 Support 记录，却**不进入本版三个 ATE 对比**；`01` 空状态本身不得使这三个效应或整个 Pair ATE 自动失败。若某效应必需状态为空，只将该效应的 `output_status` 标为 `NOT_ESTIMABLE`，其他仍可计算的效应继续保留。

由于当前设计下所有节点和状态的 `r_iq_i(z)=0.125` 相同，Hájek 归一化权重估计与上述 State Mean 数值相同。实现必须用非空状态的测试验证二者等价。

Hájek 不作为第八种普通方法排名，避免与 State Mean 重复计数。

### 5.4 旧 HT 实现不得直接复用

固定版本 `DesignBased` 使用单上下文 `majority_arm_propensities()`。该概率对应所有节点在同一上下文 iid 随机处理的设计，不是中心固定 0/1 的 pair design。

v2 必须实现上面的 pair-level `q_i(z)=0.5` 和权重 8。HT 必须经过：

- 解析概率核验；
- pair 状态唯一性测试；
- 观测包含概率测试；
- 独立模拟无偏性测试；
- 状态编码映射测试。

模拟无偏性测试应使用单独的 unit-test RNG，不得用正式 Test 结果来验证估计器。

### 5.5 四节点 Fixture A：48 种穷举组合的确定性核验

除正式 N=200 Star 外，确定性验收矩阵必须包含一个独立的四节点 Fixture A：

- 拓扑：4-node Star，中心 node 0，叶子 node 1–3；
- 每个 context 观测 `M=2` 个节点；
- 两臂中心处理固定为 `T=0` 与 `T=1`；
- 两臂共享同一组 3 个叶子处理和同一组 2 个观测节点；
- 穷举全部 `2^3=8` 个叶子处理向量及 `C(4,2)=6` 个观测节点子集，共 `8×6=48` 种组合。

Fixture A 的概率和权重为：

\[ r=2/4=0.5,\qquad q_i(z)=0.5,\qquad r q_i(z)=0.25 \]

\[ w_i^{FixtureA}(z)=\frac{1}{r q_i(z)}=4 \]

等价的单 context 边际概率写法为 `\pi_i^{ctx}(z)=0.25`，因此 `1/(2r\pi_i^{ctx}(z))=4`。**Fixture A 的权重 4 仅用于该四节点、M=2 的测试夹具；正式 N=200、M=50 Star 的权重仍为 8。实现必须从设计参数/包含概率计算权重，不得把 8 或 4 硬编码为跨设计常数。**

48 种组合的穷举测试必须验证：

1.  对每个 node-state，pair-state inclusion 概率及观测联合包含概率与解析值一致；
2.  **测试层 A（固定四状态结果表）**：人为给定任意固定的 `Y_i(t,s)` 表，令事实结果只按已实现状态从表中读取，且不加入观测噪声；验证 48 组合的 HT 平均值严格等于四节点有限总体状态均值（在浮点容差内）。
3.  **测试层 B（真实连续暴露 DGP）**：固定一个机制、设 noise=0，逐项穷举处理分配，使用与模拟器一致的结构方程计算事实结果；独立按 conditional assignment 真值定义计算 `mu_i(t,s)`，再检查 48 组合的 HT 平均值与目标均值一致。两层不得混用固定状态结果表和连续暴露结构方程。
4.  空状态时保留 `count=0`、`empirical_support=false`，不得把零贡献解释为真实响应为零；
5.  由四状态对比得到的 `Total = Direct + Spillover` 恒等式成立；
6.  两臂是共享随机量的配对上下文，不得作为独立样本处理。

Fixture A 仅是确定性 unit-test，不属于 Pilot/Test cohort，不增加正式实验任务数，也不生成正式实验数据。

------------------------------------------------------------------------

## 6. 七种普通方法的输入输出契约

### 6.1 共同事实数据预算

每个上下文 predictor 的规范输入严格为：

``` text
x:                  float32[200,20]
adjacency:          float[200,200]
treatment:          float[200]
observed_indices:   int[50]
outcomes_observed:  float64[50]  # canonical raw factual outcomes
```

`outcomes_observed` 在数据文件、TaskSpec 和通用适配层中均保持 float64。iPFN-J adapter 在调用冻结 J 模型前显式转换为 float32；转换结果只用于该次模型调用，不覆盖原始数组或保存文件。一个 pair 有两组这样的上下文输入。不得额外传入：

- full Y；
- 未观测节点的结果；
- beta；
- h（Oracle 以外的方法）；
- truth mu / ITE / ATE；
- Test 评价结果。

同一个 pair 的两臂使用相同 `x`、`adjacency` 和 `observed_indices`；处理和观测结果按中心处理不同。

### 6.2 方法能力表

| 方法       | pair 估计方式                   | 本协议输出能力                                     | 区间                                     |
|------------|---------------------------------|----------------------------------------------------|------------------------------------------|
| State Mean | 两臂合并，按四状态均值          | Pair `ATE[3]`；`mu/ITE` unsupported                | 不支持                                   |
| HT         | pair-level HT，正式 Star 权重 8 | Pair `ATE[3]`；`mu/ITE` unsupported                | 不支持                                   |
| F-z2       | 两臂联合加权结果回归            | **仅 Pair `ATE[3]`**；不要求节点级 `mu/ITE` 主结果 | 不支持                                   |
| Lin-z2     | 两臂联合状态交互结果回归        | **仅 Pair `ATE[3]`**；不要求节点级 `mu/ITE` 主结果 | 不支持                                   |
| kernel-old | 两个上下文分别推断，ATE 取平均  | `mu[N,2,2]`、`ITE[N,3]`、Pair `ATE[3]`             | 不支持                                   |
| kernel-new | 两个上下文分别推断，ATE 取平均  | `mu[N,2,2]`、`ITE[N,3]`、Pair `ATE[3]`             | 不支持                                   |
| iPFN-J     | 两个上下文分别推断，ATE 取平均  | `mu[N,2,2]`、`ITE[N,3]`、Pair `ATE[3]`             | 单上下文三个总体 ATE 的边际 95% 后验区间 |

State Mean、HT、F-z2、Lin-z2 均不参加节点级 `mu/ITE` 主结果。尤其 F-z2/Lin-z2 的协议任务仅为 Pair ATE 点估计比较；不得因内部计算过节点查询行，就把它们扩展成强制节点级主结果。

### 6.3 State Mean

- 使用两臂所有已观测事实记录；
- 根据各记录实际 `(t,s)` 状态分组；
- 每个状态取观测结果的简单均值；
- 四状态均值形成 Direct、Spillover、Total ATE；
- 对空状态记录状态级 `NOT_ESTIMABLE`，不得伪造均值；**逐效应**检查所需状态是否存在，不能因未使用的 `01` 为空而错误拒绝 Direct/Spillover/Total；
- 对每个效应分别写入 `output_status`；未用状态缺失不会使其他效应失败；
- 输出每个状态样本数和 Support 原因。

### 6.4 HT

按第 5 节的 pair-level HT 公式实现。不得使用单上下文的旧 propensity 替代 `q_i(z)=0.5`。

### 6.5 F-z2 与 Lin-z2

两者使用固定资产 `models/feature_transform.npz`：

\[ z_i^{raw}= \left(\frac{x_i-x_{mean}}{x_{scale}}\right)P \]

其中 `P` 为冻结的 20×2 projection。

之后严格复用固定 `state_design()` 的 full-N covariate 处理原则：

1.  对全体 N 个节点的 z 去均值；
2.  按全体 N 个节点的 RMS scale 标准化；
3.  用预设 `rcond=1e-6` 的 SVD 去除数值共线方向；
4.  保留的左奇异向量乘 `sqrt(N)`；
5.  不使用观测子集均值代替全图均值。

pair 内两臂的观测行联合堆叠，拟合结果模型：

**F-z2：**

\[ Y_{ic}=\alpha_{Z_{ic}}+\beta^\top z_i+\varepsilon_{ic} \]

四个状态各有截距，二维协变量斜率在四状态间共享。

**Lin-z2：**

\[ Y_{ic}=\alpha_{Z_{ic}}+\beta_{Z_{ic}}^\top z_i+\varepsilon_{ic} \]

四个状态各有截距和各自的二维协变量斜率。

每个 node-state 记录的 pair inclusion probability 为 0.125，故设计权重为 8。该权重在当前设计下是常数，WLS 点估计与对应 OLS 点估计相同。

ATE 通过全体 N 节点上的查询设计行平均计算，不以观测子集平均替代。F-z2/Lin-z2 的交付结果限于 Pair `ATE[3]` 与必要的拟合/可估计性诊断；不要求提交节点级 `mu` 或 ITE，也不纳入节点级主结果表。

必须检查：

- 设计矩阵秩、奇异值和条件数；
- 每个效应对比是否位于设计矩阵行空间；
- 对比不可估计时返回 `NOT_ESTIMABLE`；
- 不得仅因 `lstsq` 返回最小范数解就声称目标可估计。

由于同一节点两臂噪声共享，若额外报告回归系数不确定性，应以节点为 cluster；本协议主比较只使用点估计，不把 100 条记录当作 100 个独立误差样本。

### 6.6 kernel-old 与 kernel-new

- 复用固定 commit 中的冻结模型和资产；
- 每个上下文单独调用；
- 输入严格使用五数组接口；
- 不 refit、不重新选择 prior、不调整 mixture weights；
- 两个上下文的全图 ATE 点估计取算术平均作为 pair-level ATE；
- 任一上下文失败，则该方法该 pair 的 ATE 记为失败，不得只保留成功臂。

### 6.7 iPFN-J

- 使用固定 `models/iPFN_J.pt`；
- `predict.py` 的 J checkpoint SHA256 必须与固定 `experiment.json` 一致；
- 每个上下文独立推断，不跨上下文更新参数；
- adapter 将规范 float64 `outcomes_observed` 显式转换为模型所需的 float32；转换不回写原始事实数据；
- 两个上下文的全图 ATE 点估计取算术平均作为 pair-level ATE；
- 两个上下文分别计算原始 J 的三个边际 95% ATE 后验可信区间；
- 不把两个区间端点平均后称为 pair-level 95% 区间；
- 任一上下文点预测失败，该 pair 的 J ATE 失败；任一区间请求失败，仍保留该区间请求分母。

### 6.8 资产哈希

固定 `experiment.json` 中至少核验：

| 资产                               | SHA256                                                             |
|------------------------------------|--------------------------------------------------------------------|
| `models/iPFN_J.pt`                 | `a0dde8e928d7d48c0ef8581b81cf38fecd9525408bff7179b4df7ee761fed7a6` |
| `models/feature_transform.npz`     | `b261372fea1cb241e4acbd9734eab5b9506f6478e90d90c0f5f3e633b7a3a737` |
| `models/kernel_subspace_prior.npz` | `02734745ee96c712da1494ced0960538a5ac4b8eeb31b2dfa70c90ccf5ebe130` |
| `models/kernel_linear_prior.npz`   | `90e040ecdf4eba9d92e90f98c457dbffc24821498cc0f189c8f77cc52239806c` |
| `models/kernel_components.json`    | `594aca0221f35e364d012e2067546f0ea4c61bb6cf89db8a87cbdd9f6b19fb3d` |
| `models/kernel_mixture.json`       | `3e8a76ab443649831824c61e3157be497345c66d9c8548852dcf8d5320da3981` |
| `models/kernel_calibration.pt`     | `6273c1abb1255a7d8159fbbd6558c56490915e60ebd312c7c36b33c26b064a2e` |

此外，按固定 `verify_assets()` 规则核验 `regional_selected.json` 中登记的 `regional_inference.py` 哈希。N=200 不触发小网络 regional gate，但源文件完整性仍应记录。

固定 `experiment.json` 还记录 `families=7`、`mechanisms_per_family=8`、`assignment_noise_repeats=12` 及 `deep_fit_seed` 公式；这些信息可帮助追溯训练设置，但当前可见元数据没有完整列出 J/kernel 的训练、验证、校准和模型选择机制身份。

以上哈希只证明当前资产字节身份，不证明其训练/验证/选择数据来源。

------------------------------------------------------------------------

## 7. Oracle 设计

Oracle 作为已知 DGP 的 privileged reference，独立报告，不参与七种普通方法排名，也不被称为必然性能上界。

### 7.1 Oracle-context

使用 PR #3 的 `e14_oracle.py` 作为参考。

允许输入：

- 真实隐藏表征 `true_h`；
- 共同 DGP 的已知 25 维系数先验 `coefficient_prior()`；
- 已知噪声标准差 `noise_sd`；
- 当前上下文的事实 X、A、T、observed_indices、outcomes_observed。

禁止输入：

- 实际抽取的 beta；
- truth mu / ITE / ATE；
- 未观测 Y；
- Test 评价结果。

使用共同 25 维设计：

\[
F(h,\bar h,t,e,j)=
[\,1,\ h^\top,\ \bar h^\top,\ t(1,h^\top),\ e(1,h^\top),\ te(1,h^\top),\ j^\top,\ tj^\top\,].
\]

单上下文后验：

\[
\Sigma_{\mathrm{post}}=
\Sigma-\Sigma F^\top(F\Sigma F^\top+\sigma^2I)^{-1}F\Sigma.
\]

\[
m_{\mathrm{post}}=
m+\Sigma F^\top(F\Sigma F^\top+\sigma^2I)^{-1}(Y-Fm).
\]

对全图四状态 query 计算 posterior mean；对全图 ATE 对比行计算边际 Gaussian posterior interval。

Oracle-context 提供：

- `mu[N,2,2]` posterior mean；
- `ITE[N,3]` posterior mean；
- `ATE[3]` posterior mean；
- `ATE[3]` 的 95% Gaussian credible interval。

PR #3 参考实现不提供节点级可信区间，v2 不得虚构该能力。

Oracle-context 的 pair 点估计按两臂分别推断后取平均，作为独立参考行；其区间仍是两个单上下文区间，不能平均端点形成 pair 区间。

### 7.2 Oracle-pair：Star-specific 联合约束

Star pair 的两臂共享噪声。对同一批 M=50 个观测节点：

\[ Y^{(0)}=F^{(0)}\beta+\epsilon \]

\[ Y^{(1)}=F^{(1)}\beta+\epsilon \]

定义：

\[ \bar Y=(Y^{(0)}+Y^{(1)})/2,\quad \Delta Y=Y^{(1)}-Y^{(0)} \]

\[ \bar F=(F^{(0)}+F^{(1)})/2,\quad \Delta F=F^{(1)}-F^{(0)} \]

则：

\[ \bar Y=\bar F\beta+\epsilon \]

\[ \Delta Y=\Delta F\beta \]

注意：

\[ Var(\epsilon_{\bar Y})=\sigma^2 \]

不是 `sigma²/2`，因为两臂噪声完全共享。

`DeltaY` 给出无噪声线性约束。浮点运算不保证 bitwise exact，因此：

- 两臂事实 Y 原始值使用 float64；
- 预先固定差分约束残差容差；
- 预先固定 SVD numerical-rank 规则；
- 不得通过任意 ridge 把精确约束变成有噪声约束；
- 若约束不一致或数值分解失败，Oracle-pair 返回失败并保留证据。

### 7.3 Oracle-pair 的先验白化与 SVD 参数化

设：

\[ \beta=m+Lz,\quad z\sim N(0,I) \]

其中 `LLᵀ=Σ`。令：

\[ A=\Delta F L,\quad b=\Delta Y-\Delta Fm \]

对 A 做 SVD：

\[ A=U_rS_rV_r^\top \]

在预先规定 numerical-rank 阈值下检查：

\[ |b-U_rU_r^\top b| \]

若超过容差，标记 `FAILED`，不继续伪造后验。

若约束一致，取：

\[ z_p=A^+b \]

并以 `V_0` 表示 A 的零空间正交基，则：

\[ z=z_p+V_0u,\quad u\sim N(0,I) \]

再使用 noisy average likelihood：

\[
\bar Y-\bar F(m+Lz_p)=\bar F L V_0u+\epsilon.
\]

在低维零空间中做普通 Gaussian 更新。最终得到 beta 的约束后验，并对全图 ATE 线性对比计算 posterior mean 与 interval。

此前数学审查提出：Star 叶子差分设计理论秩上限约 8；若中心观测进入差分设计，上限约 9。实现必须用实际矩阵 SVD 验证该预期，不能把理论上限当作无需检查的常数。

明确的数值规则（**尚未实测，不得声称已通过**；必须先完成独立确定性 reference test，Pilot 前冻结实现）：

- rank threshold：`tol_rank = max(A.shape) * eps_float64 * largest_singular_value`；
- constraint residual：`||b - U_r U_r^T b||_2 <= 1e-10 * max(1, ||b||_2)`；
- posterior covariance 的对称性和半正定检查使用预先固定的 float64 相对容差；
- 不满足约束残差门槛时返回 `FAILED`，不得自动 ridge 或放宽 tolerance。

在原始 float64 事实数据形成时，必须对同一 25 维系数顺序的 `structural_outcome` 与 Oracle factual design 做逐行对齐测试，明确哪一步发生舍入、是否与双臂共享噪声精确一致；比较必须使用**相同来源的两臂 float64 数组和有序 `observed_indices`**。先用不含随机噪声的合成线性设计验证 SVD、约束与后验，再用共同 DGP 的确定性参考机制验证差分残差。在检查失败时先判断是设计矩阵错序、混合 dtype 还是约束不满足；保留失败证据，不允许通过变更正式 Test seed 或事后放宽数值阈值使其通过。如原定阈值确实与共同 DGP 的浮点路径不兼容，必须在 Pilot 前提出**单独记录的协议修订**并重新审查，而不能静默调参。

### 7.4 Oracle-pair 报告边界

- Oracle-pair 使用两臂联合事实信息和共享噪声结构；
- 它比 Oracle-context 的两次独立推断利用了更强的配对信息；
- 它不与七种普通方法进行同等信息预算排名；
- 它的 pair interval 单独报告，不用于替代 J 的单上下文区间覆盖率；
- Oracle-context 和 Oracle-pair 都不能使用真实 beta 或真值拟合。

------------------------------------------------------------------------

## 8. Support、识别与外推

### 8.1 四类概念

必须分开保存：

1.  `population_support`：原始目标分配分布 iid Bernoulli(0.5) 下状态是否有正概率；
2.  `design_support`：给定当前中心固定状态后，该上下文是否可能出现该节点状态；
3.  `empirical_support`：本次事实观测是否真正覆盖该节点/状态；
4.  `method_capability`：方法是否能输出该目标。

不能把它们合并为一个 `supported` 布尔值。

### 8.2 Population support

在当前 Star 中，中心度数 199、叶子度数 1；所有节点在目标 iid Bernoulli(0.5) 分配下，四状态均具有正概率。因此当前目标的 `population_support[N,2,2]` 全部为 true。

### 8.3 Context-level design support

在 `center_t=c` 的单上下文：

- 中心节点 0：自身处理必须等于 c；暴露可为 0 或 1；
- 叶子节点：暴露必须等于 c；自身处理可为 0 或 1。

因此：

\[ \operatorname{design\_support}_{center}(i=0,t,s;c)=\mathbf1\{t=c\} \]

\[ \operatorname{design\_support}_{leaf}(i>0,t,s;c)=\mathbf1\{s=c\} \]

这属于给定中心固定处理的条件设计零概率，不是原 iid 目标分配分布下的 population structural zero。

### 8.4 Pair-level design support

跨 pair 两臂后，每个节点的四状态在 pair 随机设计下均具有正概率，且每个状态在 pair 中最多出现一次。但一个 pair 中同一节点通常只实现四状态中的两个状态，不能把 pair 汇总后的群体覆盖解释成同一节点四个反事实均被观察。

### 8.5 Empirical support 与原因

保留四类原因：

- `conditional_structural_zero`：当前单上下文固定中心设计下不可能；
- `assignment_zero`：设计允许，但本次完整图处理分配没有出现；
- `observation_sampling_zero`：完整图中出现，但节点未被抽入 50 个观测节点；
- `covered`：事实观测实际覆盖。

对每个上下文保存：

- node-level `population_support[N,2,2]`；
- node-level `design_support[N,2,2]`；
- node-level `empirical_support_mask[N,2,2]`；
- node-level `support_reason[N,2,2]`；
- center/leaf × state 的 `full_graph_count` 和 `observed_count`；
- `center_observed` 单独记录。

Support 原因按下列互斥顺序赋值：先检查 `design_support=false` → `conditional_structural_zero`；否则检查给定完整处理向量中节点是否实现指定状态，未实现 → `assignment_zero`；若实现但节点不在 50 个观测索引中 → `observation_sampling_zero`；若实现且已观测 → `covered`。`population_support` 始终独立保存，不混入原因枚举。`center_observed` 是该 context 的观测集合是否包含 node 0，不表示其四状态都得到观测。

如果 population support=true 但 empirical support=false，模型可以输出预测，但必须标记为外推。若当前 context design support=false，也应单独标记为 context-specific design extrapolation。**下游决策闸门：** 当某查询 `empirical_support_mask=false` 时，数值预测可以保留用于本实验误差研究，但不得将该查询直接当作普通已支持的 plug-in 决策输入；若未来决策模块尝试消费该结果，必须先按上游 Identification/Support 契约显式拦截或使用另行授权的外推处理策略。本版不实施任何决策实验。

### 8.6 识别状态

`EstimationRequest` 状态由识别层提供。若为 `NOT_IDENTIFIED`：

- 不调用正常预测路径；
- 不生成正常有效的 mu/ITE/ATE；
- 原样保留识别失败原因；
- `EstimateBundle.status=NOT_IDENTIFIED`，同时 `execution_status=NOT_IDENTIFIED`；正常 `mu/ite/ate` 不得有效输出。

不能因为模型可以输出数值就覆盖上游识别状态。

------------------------------------------------------------------------

## 9. 评价指标与聚合

### 9.1 主分析 A：配对级 ATE 点估计

对机制 m、pair repeat r、效应 k：

\[ e_{m,r,k}=\widehat\tau_{m,r,k}^{pair}-\tau_{m,k} \]

每个机制内：

\[ \bar e_{m,k}=\frac1R\sum_{r=1}^R e_{m,r,k} \]

主 RMSE：

\[ RMSE_k= \sqrt{ \frac1K\sum_{m=1}^K \frac1R\sum_{r=1}^R e_{m,r,k}^2 } \]

有符号 Bias：

\[ Bias_k=\frac1K\sum_m\bar e_{m,k} \]

绝对条件 Bias：

\[ ACB_k=\frac1K\sum_m|\bar e_{m,k}| \]

总体绝对 Bias（对应学长转述的 *absolute Bias*；不能与 ACB 混淆）：

\[
\mathrm{AbsoluteBias}_k=|Bias_k|.
\]

机制内（重复 pair 的）误差方差：

\[
V_{\mathrm{within},k}=\frac{1}{K}\sum_{m=1}^{K}
\frac{1}{R-1}\sum_{r=1}^{R}(e_{m,r,k}-\bar e_{m,k})^2.
\]

机制间波动另单列，不得冒称为同一方法的条件抽样方差：

\[
V_{\mathrm{between},k}=\frac{1}{K-1}\sum_{m=1}^{K}
(\bar e_{m,k}-Bias_k)^2.
\]

其中 `V_within` 反映已模拟的每机制 `R=3` 次 pair 设计重复下的误差波动（仅是 3 次重复的经验估计）；`V_between` 反映机制间的平均误差异质性。二者都不同于单独一个模型后验预测分布的 variance。结果表必须同时报告 `signed_bias`、`absolute_bias`、`absolute_conditional_bias`、`within_mechanism_variance`、`between_mechanism_variability`，并标明定义与任务成功数。

其中 `K=200`、`R=3`，共 600 个 pair-level 请求/方法。**mechanism 是独立统计单位**；每个机制内的 3 个 pair repeat 是嵌套重复，不是 3 个独立机制。600 个 pair、1,200 个 context、节点或状态均不得替代 200 个独立机制作为统计推断单位。同时报告每个机制指标、机制间标准差、方法请求数、有效数和失败原因。若任何 pair 点估计不完整，完整 cohort 的 RMSE/Bias/Variance 不可静默按成功子集重算并标为主结果；允许生成显式标注的 `success_subset_*` 描述性指标，正式排名依第 9.3 节记为 `NOT_RANKABLE`。

RMSE 使用先在机制内平均平方误差、再机制等权、最后开根号的定义。若另报告任务 RMSE 的算术平均，必须标记 `mean_task_rmse`，不能与主 `pooled_rmse` 混用。

#### 9.1.1 运行时间和资源指标（对齐共同交付要求）

Test 每个方法必须记录 `runtime_seconds`、`peak_rss_bytes`；使用 GPU 时还需记录 GPU 峰值内存与设备型号。计时边界为**完成一个 pair 的该方法 ATE 点估计所需的适配器、特征变换、模型调用与汇总**，不包含共同的数据生成、模型首次导入/加载、evaluator 指标计算或图表生成；这部分开销另记为 `setup_seconds`、`data_generation_seconds`、`evaluation_seconds`。计时采用单调时钟，GPU 异步调用必须同步后记录。J 的 context-level posterior interval 另记 `interval_runtime_seconds`，不与点预测时间混在一起。

每方法按机制先合并其 3 个 pair 的 runtime，然后报告 200 个机制块的平均值、标准差、中位数、P95；同时保留每 pair 原始计时、失败状态和硬件/软件环境。不同方法使用的合并策略不同（传统方法联合拟合、J/kernel 两次调用），需展示方法内端到端成本，而不是强行把模型调用次数归一化后称为公平 runtime。`peak_rss_bytes`/GPU peak 使用各自明确的测量窗口，若测量不可得写 `UNAVAILABLE` 及原因，不伪造为零。

### 9.2 方法比较与不确定性

- 以 200 个 mechanism 作为独立统计单位；
- Bootstrap 必须按完整机制块重抽样，固定 10,000 次；
- 每次抽中一个机制时，整体携带该机制全部 3 个 pair、每个 pair 的两种 `center_t` 上下文、所有方法结果、区间记录和失败状态；
- 方法间比较使用同一机制/同一 pair 的配对差异；
- Bootstrap 固定使用 Test cohort `cohort_code=2`、`mechanism_id=0`、`channel_code=7`、`pair_repeat_id=0`；即 `SeedSequence([20261030, 2026103001, 2, 0, 7, 0])` → `PCG64` 的唯一全局 Bootstrap 随机流。它仅用于评价器的**机制索引重抽样**，不会生成任何机制/处理/噪声，也不因为 mechanism_id=0 就把某个 Test 机制特殊对待；
- 将机制按 `mechanism_id=0,...,199` 排序，每次从 `0,...,199` 有放回抽取 200 个索引，重复 10,000 次；保留 Bootstrap replicate ID、抽样索引或可重建索引的 RNG identity、NumPy 版本；同一抽样索引用于七方法配对差异与全部三种效应，不为每个方法独立重抽；
- 不把 600 个 pair、1,200 个 context、节点、状态、中心臂或重复上下文当成独立机制；
- 若某方法未完成全部预请求任务，该方法仍标记 `NOT_RANKABLE`；不得通过只抽取成功 pair 的 Bootstrap 让其重新获得正式排名；
- 不根据单个 RMSE 数字宣称显著性。

### 9.3 失败对排名的影响

对每个普通方法、每个效应，预请求点估计总数为 600 个 pair（七种普通方法合计 4,200 个 pair-method 请求）。

若任一预请求 pair 的**对应效应**点估计缺失、失败或不可估计（尤其传统状态均值/回归应按效应的必需状态与对比可估计性判定，而不能因不参与该效应的 `01` 状态缺失而整体失败）：

- 保留原请求记录及失败状态，不删除、不覆盖；
- 报告成功率和失败原因；
- 标记该方法该效应 `NOT_RANKABLE`；
- 成功子集 RMSE 只能作为描述性结果；
- 不设置任意巨额误差惩罚；
- 不得静默删除失败 pair。

对 J/kernel/Oracle-context，任一中心上下文失败，则整个 pair 的 pair ATE 失败。

### 9.4 主分析 B：节点级响应与 ITE

只纳入具有节点级输出能力的 J、kernel-old、kernel-new；Oracle-context 独立展示。State Mean、HT、F-z2、Lin-z2 不伪造节点 ITE。F-z2/Lin-z2 仅参加主分析 A 的 Pair ATE 点估计比较，不要求节点级 `mu/ITE`，也不进入本节节点级主结果。

节点四状态响应误差：

\[
\mathrm{MSE}_{\mu,mrc,g}=
\frac{1}{4|g|}\sum_{i\in g}\sum_{t=0}^{1}\sum_{s=0}^{1}
(\widehat{\mu}_{mrc,i}(t,s)-\mu_{m,i}(t,s))^2.
\]

节点效应误差：

\[
\mathrm{MSE}_{\theta,mrc,g,k}=
\frac{1}{|g|}\sum_{i\in g}
(\widehat{\theta}_{mrc,i,k}-\theta_{m,i,k})^2.
\]

分组：

- center：node 0；
- leaf：node 1–199；
- all：全部 200 节点。

聚合顺序：

1.  每个 context 在组内对节点和状态/效应平均；
2.  每个机制内对 3 个 pair × 2 个 center_t 等权平均；
3.  200 个机制等权平均；
4.  pooled RMSE 为平均 MSE 后开根号。

中心与叶子必须单独报告，不能让 199 个叶子掩盖中心表现。

### 9.5 组效应差异

对 center、leaf、all 分别计算：

- 真实组平均 Direct / Spillover / Total；
- 预测组平均 Direct / Spillover / Total；
- 组级 ATE RMSE、Bias、ACB。

center 只有一个节点，因此中心误差跨机制汇总；不能把每个机制中的同一个中心节点当作 200 个独立节点样本。

中心与叶子 RMSE 差异是描述性/条件性比较，不解释为 Support 对误差的因果效应。

#### 9.5.1 中心—叶子效应差异及误差（E1.5 核心交付）

对每个机制 m、效应 k 定义中心相对于叶子的真实差异：

\[
\Delta_{m,k}=\theta_{m,0,k}-\frac{1}{199}\sum_{i=1}^{199}\theta_{m,i,k}.
\]

对 J/kernel 的每个 context `(m,r,c)` 使用同一全图四状态预测计算 `\widehat{\Delta}_{mrc,k}`；定义误差 `e^\Delta_{mrc,k}=\widehat{\Delta}_{mrc,k}-\Delta_{m,k}`。保留 1,200 个 context 的原始误差记录，按 context → 机制内六次重复 → 200 个机制等权计算 RMSE、signed Bias、absolute Bias 与 ACB，另记录中心/叶子组均值、机制间标准差。Oracle-context 只作独立参考。

ATE-only 的四种传统方法**不需要**提供节点级/中心—叶子差异估计，标为 `UNSUPPORTED`，不能把全图 ATE 广播给中心和叶子后宣称预测了差异。此分析只复用原有 J/kernel 节点预测和 truth，不增加任何额外 context 或新 cohort。

#### 9.5.2 自然 Support 分层误差（描述性）

仅基于现有 Test 的自然分配和观测抽样，对 J/kernel 的每个 `context × node × state`，按第 8.5 节的**互斥优先级**分成 `conditional_structural_zero`、`assignment_zero`、`observation_sampling_zero`、`covered`，另外对 context 标注 `center_observed`。每层记录请求总数、有效预测数、错误/缺失数、真实 Support 计数；在有效预测上计算节点状态误差的 MSE/RMSE 和 signed Bias，按机制聚合而不是把节点作为独立机制。

并分别报告 center / leaf 以及 `center_t=0/1` 的诊断性分层。层内缺失预测不得静默丢弃；只有至少一个有效请求的层才能报告条件误差，空层返回 N/A。分层误差是**描述性关联**，不用于说明 Support 对误差具有因果作用，也不是新的 Support 压力试验；不得修改叶子处理概率、节点抽样规则或目标。

### 9.6 J 区间评价

J 的 `predict_intervals()` 对每个单上下文返回三个效应的边际名义 95% ATE 后验可信区间。区间是 context-level 输出，不是 pair-level 区间。

Test 共：

- 1,200 个 context-level J interval 请求；
- 每个请求 3 个效应区间；
- 共 3,600 个 effect-specific interval records；
- Direct、Spillover、Total 每个效应各有 1,200 个总体请求；
- `center_t=0` 和 `center_t=1` 各有 600 个请求/效应。

对每个效应 `k` 分别报告：

\[ coverage_{valid,k}=\frac{covered_{valid,k}}{valid_{intervals,k}} \]

\[ coverage_{including_failures,k}=\frac{covered_{k}}{all_requested_{k}} \]

其中 `all_requested_k=1,200`。任何请求后失败、依赖的点预测失败、缺失、非有限、上下界反向、输出损坏或无法判定覆盖的区间，均保留在 `all_requested_k` 中，并在 `coverage_including_failures` 中按未覆盖计；即使因依赖失败未真正进入推断函数，也必须保留原预请求身份并标记 `FAILED` 及明确原因。若 `valid_intervals=0`，则 `coverage_valid=N/A` 而非 0 或除零，主要分母仍是全部 1,200 个请求。不得先删除失败区间再计算主要覆盖率。

`coverage_valid` 仅作为“成功返回区间中的条件覆盖率”描述性诊断；它不能替代主要验收指标，也不能用于判定是否达到 90%。同时报告 `interval_success_rate=valid_intervals/all_requested`、平均区间宽度、失败数及失败原因。

其他不支持区间的方法记 N/A，不进入 J 区间分母。中心处理分层结果分别以每层 600 个请求为分母完整报告。

**严禁将两个单上下文区间端点平均后称为 pair-level 95% 区间。** Oracle-pair 区间若另行输出，独立报告，不替代 J 的单上下文区间评价。

### 9.7 90% 覆盖率目标

J 的三个总体 ATE 名义 95% 区间覆盖率是三个彼此独立的主要验收单元：

- Direct 总体 `coverage_including_failures >= 0.90`；
- Spillover 总体 `coverage_including_failures >= 0.90`；
- Total 总体 `coverage_including_failures >= 0.90`。

三个效应必须分别达到 90%；不得把三者平均后掩盖任一效应未达标。失败、缺失或无效区间在各自的 1,200 请求分母中按未覆盖计。`coverage_valid` 仅供诊断，不作为通过门槛。

`center_t=0/1` 的六个分层结果（两个中心处理层 × 三个效应）必须完整报告，但仅作为诊断分析，不设独立 90% 硬性门槛。机制聚类不确定性按完整机制块 Bootstrap 报告；不要求 Bootstrap 覆盖率区间下界也超过 90%。90% 是预先指定的经验覆盖率点目标，不是置信区间下界检验。

任何一项总体效应低于 90%，如实报告科学未达标；不得删除机制、换种子、放宽阈值或利用 Test 结果调整 J 区间。

## 10. 校准、训练与信息泄漏边界

### 10.1 不新增 J 校准方法

注意：固定资产 `models/kernel_calibration.pt` 是现有 kernel-new baseline 的冻结 checkpoint。它不是本协议拟新增的 J 区间校准，也不意味着 v2 会重新拟合或调整 J 的 posterior interval。

Candidate v0.3.2 主方案：

- 不建立 Calibration cohort；
- 不新增 `j-calibrated`；
- 不增加 Mechanism OOD cohort；
- 不增加人工 Support pressure cohort；
- 不在 Test 结果上调整区间；
- 冻结 J 的原始 95% posterior interval 直接进入 Test。

PR #2 的独立区间校准是 E1.2 Configuration 场景的既有工作，不自动证明该校准规则可迁移到 E1.5 Star。若未来要校准，必须单独定义校准分数、机制级依赖结构、校准 cohort、冻结时点和适用范围。

### 10.2 不重新训练或选择模型

原始任务书列出 PFN checkpoint、训练评测程序等拓扑子项的共同交付表述。本版获批方案采用**冻结已有 J checkpoint 的评估**，不是每拓扑重新训练一个新 checkpoint；交付时必须提供冻结 checkpoint 的身份/哈希和 provenance 限制，不得声称新训练已完成。若 E1 负责人后来明确要求 E1.5 专项重新训练，属于**本协议范围外的项目要求冲突**，须另立变更决议，不得在当前 Test 内私自执行。

不允许：

- 重新训练 J；
- 调整 J checkpoint；
- 重新选择 kernel prior；
- 改动 kernel-new checkpoint；
- 使用 Pilot/Test 真值选择方法；
- 使用 Test 覆盖率调整区间；
- 在 Test 上进行 hyperparameter tuning。

F-z2/Lin-z2 是每个 pair 上按预先规定模型形式拟合的经典结果回归基线，不属于预训练 PFN/kernel 权重的重新训练。

### 10.3 预测输入与评估真值隔离

普通方法：

- 只接收 X、A、T、observed_indices、outcomes_observed；
- 不接收 true_h、beta、truth mu、ITE、ATE 或未观测 Y。

Oracle：

- 可额外接收 true_h、已知 prior、已知 noise_sd；
- 不接收真实抽取 beta 或任何真值响应；
- 不接收未观测 Y。

Evaluator：

- 在预测完成后读取 truth；
- 独立计算误差、覆盖率和表格；
- 不得把 evaluator 数据传回预测器。

------------------------------------------------------------------------

## 11. TaskSpec / EstimationRequest / EstimateBundle

### 11.0 调用与状态权属（兼容原始 Estimation PDF）

规范调用链为：`TaskSpec + EstimationRequest(Identification) + model -> EstimateBundle`。原始任务书要求 `target_id/status` **继承识别请求**，所以接口中 `status` 专用于上游 Identification 的 `POINT` / `NOT_IDENTIFIED`，不得把执行成功的 `OK` 覆写到 `status`。实现成功、不可估计、失败或输出不支持等信息使用独立字段 `execution_status` 与 `output_status`。

此规范是对 v0.3 中单字段 `status` 多义性的修复；若某个已有实现只返回 `status=OK`，只能在显式 legacy adapter 中映射到 `execution_status=OK`，同时从原请求原样继承 `status=POINT`，不能让 adapter 自行推断识别状态。

### 11.1 TaskSpec

每个 context 至少包含：

- `protocol_version`（已冻结的唯一协议标识 `Candidate v0.3.2`；实际执行时必须记录**本冻结版文件**的 SHA256，不得用冻结前原稿 SHA256 或移动的 `latest` 代替）
- `protocol_id`（`e1_5_star_v2`）
- `protocol_code`（`2026103001`）
- `root_seed`（`20261030`）
- `cohort` 与 `cohort_code`（Pilot `1`；Test `2`）
- `mechanism_id`
- `pair_repeat_id`
- `center_t`
- `node_ids`（长度 200，顺序固定）
- `x`（`[200,20]`）
- `adjacency`（`[200,200]`）
- `treatment`（`[200]`）
- `observed_indices`（`[50]`）
- `outcomes_observed`（规范原始 dtype `float64[50]`）
- `rng_identity`（含 channel、SeedSequence identity/entropy、PCG64）
- `input_hash`

不包含完整 Y 或 truth。J 的 float32 转换只发生在 J adapter 内部，不改变 TaskSpec 中的 float64 原始值。

#### 11.1.1 PairTaskSpec（配对任务汇总契约）

这是两个 context `TaskSpec` 的**只读引用和一致性校验封装**，不替代原 `TaskSpec`，不向 J/kernel 暴露额外信息：

- `pair_id`：由 `(protocol_id, cohort_code, mechanism_id, pair_repeat_id)` 的规范序列化确定，禁用不稳定的 Python `hash()`；
- `mechanism_id`、`pair_repeat_id`、`cohort`/`cohort_code`、`target_id`（来自请求）；
- `context_ids`：恰有 `center_t=0/1` 两个且不可重复；`context_refs` 对应这两个任务；
- `shared_input_hashes`：X、adjacency、叶子处理、原始噪声、`observed_indices` 的一致性证据（privileged 原始噪声哈希仅用于模拟/验收，**不可**作为预测输入）；
- `n_unique_observed_nodes=50`、`n_factual_records=100`，两臂 `outcomes_observed` 都是 `float64[50]`；
- `pair_design_id`、`reference_assignment`、`arm_order`、`node_ids`、`rng_identity` 和 `protocol_hash`。

如果共享约束被破坏，先保存两个原 context 的错误证据，再使 pair 执行状态为 `FAILED`；不得通过重采样、只保留一臂或自动重建缺失臂来“修复”该任务。

### 11.2 EstimationRequest

固定 target：

`target_id = "star_v2_mu_iid_bernoulli_0p5_majority_gt0p5_v1"`

请求至少包含：

- `status`：原样继承上游 `POINT` 或 `NOT_IDENTIFIED`，属于 identification 状态；
- `target_id`；
- 目标泛函；
- `reference_assignment`：iid Bernoulli(0.5)；
- `exposure_definition`：邻居处理比例严格大于 0.5；
- population support；
- 所需输出（mu、ITE、ATE、interval）；
- 节点顺序和状态轴顺序。

估计器不得自行改变目标量、reference assignment 或 exposure definition。

### 11.3 EstimateBundle

候选字段：

- `protocol_version`
- `target_id`
- `request_id`
- `method_id`
- `status`（上游 identification 的 `POINT`/`NOT_IDENTIFIED`，不可被 `OK` 覆写）
- `execution_status`（该方法该请求的执行状态）
- `output_status`（按 `mu`、`ite`、`ate`、`interval` 分别记录 `OK`/`UNSUPPORTED`/`NOT_REQUESTED`/失败等）
- `node_ids`
- `mu[N,2,2]`（仅节点响应方法；轴依次为自身处理 `t`、同伴暴露 `s`；状态顺序 `00,01,10,11`）
- `ite[N,3]`（仅节点效应方法；效应顺序 `Direct, Spillover, Total`）
- `ate[3]`（效应顺序 `Direct, Spillover, Total`；Pair ATE 对比使用该字段）
- `intervals`（每个 context×effect 的区间请求/状态/上下界及覆盖所需元数据；J 不输出合成的 pair-level credible interval）
- `sampling_uncertainty`（原任务 PDF 的要求；按方法记录后验样本/区间、机制级重复或 Bootstrap 汇总的**来源、统计单位、有效样本数与状态**；没有合法不确定性输出时标为 `UNSUPPORTED`/`NOT_REQUESTED`，不能捏造标准误；机制级 Bootstrap 是 evaluator 汇总产物，不得伪装为某单 context 的模型 posterior）
- `population_support`
- `design_support`
- `empirical_support_mask`
- `method_capability`
- `estimability`
- `extrapolation_flags`
- `diagnostics`
- `failure_type`
- `failure_message`
- `input_hash`
- `asset_hashes`

State Mean、HT、F-z2、Lin-z2 等 ATE-only 方法的 `mu/ITE` 明确为 `UNSUPPORTED`，不得广播一个总体均值伪装节点异质预测。F-z2/Lin-z2 的预注册交付只包含 Pair `ATE[3]` 和必要诊断。

#### 11.3.1 PairEstimateBundle（配对估计结果）

`pair_id / request_id / method_id / target_id / status / execution_status / output_status / ate[3] / failure_type / failure_message / diagnostics / input_hashes / asset_hashes / protocol_hash` 为最小规范字段。`status` 同样原样继承上游识别状态；`ate[3]` 仅在对应 effect 的 `output_status=OK` 时有效。还须保存 `context_ids=[id_0,id_1]` 与两臂 execution/output statuses，以免丢失失败原因。

- **State Mean/HT/F-z2/Lin-z2**：基于同一 pair 联合事实记录形成唯一的 `ate[3]`；
- **J/kernel**：两臂各产生全图 `ate_context[3]`，各效应取两臂算术平均生成 pair ATE；若必要的任一臂失败，pair 对应效应失败；
- **Oracle-context**：两臂分别推断后按上述规则形成独立参考 pair ATE；
- **Oracle-pair**：使用共享噪声约束得到独立的联合 posterior pair ATE；
- **区间字段**：J 仅有**两个 context-level** `intervals`，不能通过平均上下界构造 pair 95% 区间；Oracle-pair 区间单独保存，不参与 J 的 1,200 请求分母。

ATE-only 方法的 `mu/ite` 以逐字段 `UNSUPPORTED` 表示，而不是在一个已成功的 ATE 请求上把整个 `execution_status` 写成失败。**逐效应结果契约：** `ate[3]` 的某项仅当相应 `output_status.ate[k]=OK` 时为有限数；否则以无效占位和原因代码保存，但不得以零代替。`execution_status=OK` 仅当本请求全部**已请求且适用**的效应输出均有效；一个效应失败时，保留其他有效效应及其独立请求/验收分母。`01` 空状态不自动触发 ATE 失败。

### 11.4 状态码

| 字段 | 状态 | 语义 |
|---|---|---|
| `status` | `POINT` | 上游已经点识别，继承原请求 |
| `status` | `NOT_IDENTIFIED` | 上游不允许形成正常响应/效应 |
| `execution_status` | `OK` | 本次请求的所有必需输出成功 |
| `execution_status` | `NOT_ESTIMABLE` | 已识别，但当前样本/设计不支持有效目标对比 |
| `execution_status` | `FAILED` | 工程或数值计算失败 |
| `execution_status` | `NOT_IDENTIFIED` | 因上游不识别而未调用正常预测 |
| `output_status` | `UNSUPPORTED` | 方法不提供该字段 |
| `output_status` | `NOT_REQUESTED` | 该输出不在本次预注册请求集合 |
| `output_status` | `OK` / `FAILED` / `NOT_ESTIMABLE` | 逐字段/逐效应状态，必须与输出值有效性相符 |

不得把这些状态统一转换成 0 或直接删除。

------------------------------------------------------------------------

## 12. B6 Step 1–5 确定性验收矩阵

以下是 Codex 必须实现的测试规格，不表示这些测试已经运行。

### Step 1：统一接口契约

- TaskSpec / PairTaskSpec / EstimationRequest / EstimateBundle / PairEstimateBundle 字段与 shape；
- `status` 原样继承 `POINT/NOT_IDENTIFIED`，`execution_status` 与逐字段 `output_status` 独立；
- `sampling_uncertainty` 保存来源/单位/能力状态，不混淆 posterior 与 Bootstrap；
- `mu[N,2,2]` 轴顺序；
- node_ids 不重排；
- `target_id` 原样继承；
- `NOT_IDENTIFIED` 不生成正常 mu/ITE/ATE，并原样继承识别失败原因；
- ATE-only 方法不生成伪造 ITE；
- Support 与 method capability 分开；
- `empirical_support_mask=false` 的外推查询被下游普通 plug-in 决策闸门拦截，但允许保留预测值供误差评价；
- State Mean、F-z2、Lin-z2 的 Pair ATE 可估计性按**效应对比**独立判定，`01` 缺失不引发三效应整体失败。

### Step 2：随机身份和配对

- 同一 identity 重复生成逐数组一致；
- 不同 cohort、机制、repeat、channel 不发生意外 RNG 复用；
- pair 两臂 X/A/leaf_t/noise/observed_indices 完全一致；
- 两臂 center treatment 分别为 0、1；
- 机制之间 beta/X/population 独立；
- 新 v2 cohort 与 v1 dev 旧 RNG identity 不重叠；
- raw factual outcomes 保存为 float64；J adapter 内部转换为 float32，且原始数组 hash 不变；
- `SeedSequence` identity/entropy 和 PCG64 信息写入 manifest；
- v1 原始数据、结果和资产不被修改。

### Step 3：失败记录与任务全集

- 每个预请求 task 有唯一 task_id；
- 失败不自动重试、不换种子、不删除；
- pair 方法一臂失败则 pair 失败；
- 区间请求失败仍进入请求分母；
- `NOT_ESTIMABLE`、`NOT_IDENTIFIED`、`UNSUPPORTED`、`FAILED`、`NOT_REQUESTED` 分开；
- 不完整方法标记 `NOT_RANKABLE`。

### Step 4：方法级确定性验收

**DGP/真值**

- Star 拓扑、度数、节点顺序；
- 四状态真值与结构方程；
- Direct + Spillover = Total；
- 事实状态编码与 mu 数组顺序映射。

**HT/State Mean**

- 解析验证 `q_i(z)=0.5`；
- 正式 Star `r=0.25`、`q=0.5`、pair HT 权重为 8；
- Fixture A（4-node Star、M=2）穷举 `2^3 × C(4,2)=48` 组合，验证 `r=0.5`、`q=0.5`、HT 权重为 4 和有限总体设计无偏性；
- 不把 Fixture A 的 4 与正式 Star 的 8 混用，也不硬编码单一权重常数；
- State Mean 与 Hájek 在非空状态等价；
- 构造 `01` 空、`00/10/11` 非空的确定性夹具：三个 Pair ATE 效应仍应可估计；分别删去 `00`、`10`、`11` 后，按 Direct=`10-00`、Spillover=`11-10`、Total=`11-00` 的依赖关系逐效应失败，不得过度传播；
- HT 在独立随机模拟中设计无偏；
- 空状态时 State Mean 正确标记；HT 不把零贡献解释成真实零响应。

**F-z2/Lin-z2**

- 仅产生 Pair ATE 主比较输出，不要求节点级 `mu/ITE` 主结果；
- 冻结 transform hash；
- full-N centering；
- SVD rank/condition；
- query contrast row-space estimability；
- F-z2/Lin-z2 的 pair 结果只提供 `ate[3]`，`mu/ite` 返回 `UNSUPPORTED`；
- 不可估计时返回明确状态；
- 同一 pair 两臂共享节点噪声时，不把 100 行当作独立误差单位。

**J/kernel**

- 五输入键严格；
- 输入数组调用前后 hash 不变；
- 资产 hash 校验；
- 无全量 Y/truth/beta 泄漏；
- 模型参数和 checkpoint 不发生更新；
- pair 任一臂失败则 pair 失败。

**Oracle-context**

- coefficient prior 与 DGP prior 一致；
- 25 维设计列顺序一致；
- posterior covariance 对称半正定；
- ATE posterior variance 合法；
- 不接收真实 beta/真值。

**Oracle-pair**

- 共享噪声差分残差在预设 float64 tolerance 内；
- `Var(Ybar noise)=sigma²`；
- SVD rank、零空间和约束一致性；
- 不对奇异矩阵直接 Cholesky；
- 不通过随意 ridge 规避失败；
- posterior mean/covariance 与独立低维参考计算一致。

### Step 5：统一协议交叉验收

- 任务清单与 manifest 一致，包含 `01` 状态为空但效应仍可估计的逐效应状态用例；
- Bootstrap 的固定 Test/global 随机身份、200 个机制 ID 排序、10,000 次重抽与配对差异共用抽样索引可复现；
- factual/truth/prediction/support/interval 文件一一对应；
- Pilot 为 5 mechanisms / 10 pairs / 20 contexts；Test 为 200 mechanisms / 600 pairs / 1,200 contexts；
- 七种普通方法各有 600 个 Pair ATE 预请求；J 三种效应各有 1,200 个 interval 请求；
- CSV 与 NPZ 指标可复算；
- signed/absolute Bias、ACB、within/between variance、runtime/peak memory、中心—叶子差异、自然 Support 分层及其有效分母均有可复算原始表；
- `make_figures.py` 和 `make_table.py` 从规范 `evaluation/` 与 manifest 生成预定义图表与 CSV/LaTeX 表，禁止手工修改数字；
- Bootstrap 机制块单位正确，整块携带每个机制的 3 个 pair 和两臂上下文；
- 失败点估计与失败区间均保留在任务记录/对应统计分母中；
- 所有哈希可核验；
- 原 v1 目录、结果和模型资产未改变；
- 工程验收通过不等于科学表现达标。

------------------------------------------------------------------------

## 13. Pilot、Test 与冻结入口

### 13.1 当前实施边界

Candidate v0.3.2 已获用户明确批准，**当前科学协议状态为 `FROZEN`**；Stage 1 已完成只读工程审查。截至该阶段，尚未执行 v2 代码实现、确定性单元测试、数据生成、模型加载、预测、训练、校准、Smoke、Pilot 或 Formal Test。协议冻结本身不授权后续执行。

后续工程实施和 Pilot 准入必须按阶段完成以下项目（协议审阅与用户冻结批准已完成）：

- 记录并核验本冻结版文件的 SHA256、固定源码基点、独立工程环境及实施授权范围；
- 按本协议实现独立 `StarV2Config` 与 RNG 注册；
- 七种普通方法及两种 Oracle 的输入输出契约；
- 正式 Star HT 权重 8 与 Fixture A 权重 4 的解析/穷举核验；
- float64 原始 outcomes 与 J adapter float32 转换边界；
- 全部 B6 Step 1–5 确定性验收测试实现。

协议参数获批不等于代码实现或实验运行获批。任何执行阶段均需另行授权。

### 13.2 Pilot

Pilot 规模：`5 mechanisms × 2 pairs = 10 pairs = 20 contexts`。

预请求数量：

- 七种普通方法：每种 10 个 Pair ATE 请求，共 70 个 pair-method 请求；
- J、kernel-old、kernel-new：每种 20 个 context-level 点预测请求，共 60 个 context-method 请求；
- J 区间：20 个 context-level 请求，每个请求 3 个效应区间，共 60 个 effect-specific interval records；
- Oracle-context：20 个 context-level 请求；每个请求 3 个效应区间，共 60 个 effect-specific interval records；
- Oracle-pair：10 个 pair-level 请求；每个请求 3 个效应区间，共 30 个 effect-specific interval records（独立报告，不进入 J 覆盖率分母）。

Pilot 只检查：

- DGP 与真值；
- pair 共享量与 RNG 可复现性；
- 七种方法接口及 Pair ATE 输出；
- Oracle 数值稳定性；
- Support 和失败记录；
- float64 原始 outcomes / J adapter float32 边界；
- 文件读写、manifest、hash；
- 时间、峰值内存和磁盘占用（记录真实硬件、计时边界、是否 GPU 同步及首次加载成本；Test 也必须保留相同类型指标）；
- 确定性复现。

Pilot 不用于选择 baseline、调整 J/kernel、选择 Test 机制、判断 90% 覆盖率是否达标、校准 J 区间或调整验收阈值。若 Pilot 发现科学设计错误，先修订协议；若发现工程 bug，修复后重新运行必要测试。原失败任务和每次尝试必须保留，重跑记录为新 attempt，不得覆盖或删除旧失败记录。

### 13.3 Test 前冻结

Test 前必须满足：

1.  本冻结版 Candidate v0.3.2 的 SHA256 已记录并校验，冻结批准可追溯；
2.  v2 代码、模型资产、transform 和评估脚本冻结；
3.  Test cohort 的机制 ID、repeat ID、RNG identity 预注册；
4.  Pilot 工程验收通过，且 Pilot 失败/重跑历史完整保留；
5.  模型 provenance 状态如实记录；
6.  运行资源预算以实测 Pilot 为依据；
7.  Test 结果在所有方法和任务执行完成前不得查看或用于调参；
8.  Runtime/absolute Bias/Variance/中心—叶子效应差异/Support 分层与图表输出的计算口径及验收脚本冻结。

### 13.4 Test 验收与任务数量

**请求数量与单位固定。** Runtime、Support、差异指标、图表表格均是从既定请求生成的派生分析，不新增 context、pair、mechanism 或模型预测调用；如果派生统计因上游失败而缺失，同样保留请求计数及 N/A 状态。

Test 总体规模：

- 200 个独立 mechanisms；
- 每个机制 3 个 pair repeats；
- 600 pairs；
- 每个 pair 2 个 center contexts；
- 1,200 contexts。

预请求数量：

以下“七种普通方法 Pair ATE”是其后七个方法行的合计，后续方法行用于展开该合计，不应再次相加；context-level 点预测行是 J/kernel 形成 Pair ATE 的底层调用。

| 请求类型                           |              数量 | 说明                                     |
|------------------------------------|------------------:|------------------------------------------|
| 七种普通方法 Pair ATE              | `7 × 600 = 4,200` | 每种普通方法各 600 个 pair-level 请求    |
| State Mean                         |               600 | 两臂合并估计                             |
| HT                                 |               600 | pair-level HT，权重 8                    |
| F-z2 Pair ATE                      |               600 | 仅 Pair ATE                              |
| Lin-z2 Pair ATE                    |               600 | 仅 Pair ATE                              |
| kernel-old Pair ATE                |               600 | 由两臂 context 预测取平均                |
| kernel-new Pair ATE                |               600 | 由两臂 context 预测取平均                |
| iPFN-J Pair ATE                    |               600 | 由两臂 context 预测取平均                |
| kernel-old context 点预测          |             1,200 | 600 pairs × 2 contexts                   |
| kernel-new context 点预测          |             1,200 | 600 pairs × 2 contexts                   |
| iPFN-J context 点预测              |             1,200 | 600 pairs × 2 contexts                   |
| J context-level interval 请求      |             1,200 | 每次返回 Direct/Spillover/Total 三个区间 |
| J effect-specific interval records |             3,600 | 每个效应的总体分母均为 1,200             |
| Oracle-context                     |             1,200 | 每个 context 一个请求                    |
| Oracle-context interval records    |             3,600 | 独立报告，不进入 J 覆盖率分母            |
| Oracle-pair                        |               600 | 每个 pair 一个请求                       |
| Oracle-pair interval records       |             1,800 | 独立报告，不进入 J 覆盖率分母            |

每个普通方法的 600 个 Pair ATE 请求全部保留；任一预请求点估计失败/缺失/不可估计时，该方法对应效应标记 `NOT_RANKABLE`，成功子集指标仅作描述。J 的每个总体效应区间覆盖率以 1,200 个请求为分母，失败区间按未覆盖计，并分别要求至少 90%。

表中 Pair ATE 数量与 context-level 点预测数量属于不同层级，不应简单相加：J/kernel 的两臂 context 预测是形成其 600 个 Pair ATE 输出的底层调用；J interval 与 Oracle 请求则是各自独立的预请求。以上均为协议预注册的请求数量，不是已执行结果。工程完整性、方法科学性、实验表现分开验收；工程验收通过但覆盖率低于 90%，仍应报告科学未达标。

## 14. 交付目录与复现验收建议

以下结构是**实施建议**，不意味着仓库中已经存在这些文件；在获准的实施阶段由 Codex 对照实际现有仓库结构核对路径，不改动旧 v1 文件/目录。`make_figures.py`/`make_table.py` 如已存在则优先在独立 v2 入口复用；不得未经核验声称现成脚本已支持 v2 schema。

```text
iPFN_demo/
  code/
    sparse_j/
      star_v2/
        protocol/
          E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md
          star_v2_config.json
          rng_registry.json
          target_manifest.json
        config.py
        dgp.py
        contracts.py
        support.py
        baselines.py
        frozen_models.py
        oracle_context.py
        oracle_pair.py
        metrics.py
        runner.py
        acceptance.py
        make_figures.py
        make_table.py
        tests/
          test_dgp_truth.py
          test_rng_pairing.py
          test_pair_ht.py
          test_state_mean_hajek.py
          test_f_z2_lin_z2.py
          test_oracle_context.py
          test_oracle_pair.py
          test_support.py
          test_failure_contract.py
          test_artifact_acceptance.py
      # v1 源码与已有 models/ 保持原样（不移动、不覆盖）
  results/
    e1_5_star_v2/
      pilot/<run_id>/
      test/<run_id>/
```

每个 run 的最小原始证据集：

- `run_manifest.json`：协议文档 SHA256、source SHA、模型资产哈希、依赖锁定信息、硬件/软件信息、RNG/channel 映射和真实执行命令；
- `task_manifest.jsonl`：完整 task/request identity、方法、效应、context/pair 关联及最终状态、所有失败/重跑 attempt；
- `source_snapshot.zip`：实际执行的源码快照与文件哈希清单；Git HEAD 单独记录但**不替代**有未提交修改时的有效源码快照；
- `factual/`、`truth/`、`predictions/`、`pair_estimates/`、`support/`、`intervals/`；
- `evaluation/`：机制级、pair 级、context 级原始表（含失败和分母）、绝对 Bias、方差、runtime 和 Support 分层；
- `report_views/figures/`、`report_views/tables/`：图、CSV、LaTeX 等可再生成展示产物；
- `acceptance_report.json`、`README.md`：状态、失败明细、复现命令、目录说明、证据边界和科学结论。

**两个指定脚本的输出契约：** `make_figures.py` 至少生成七方法 Pair ATE RMSE 对比图、J 覆盖率/区间宽度图、center/leaf 节点误差与真实—预测效应差异图、自然 Support 分层图及 runtime 图（不支持项和失败保留可见）；`make_table.py` 至少生成方法/效应级 `rmse,bias,absolute_bias,ACB,within_variance,between_variability,runtime,success,denominator` 汇总、J 区间覆盖率（全体与 center_t 分层）、center/leaf 效应和 Support 原始计数的 CSV 与可选 LaTeX 表。输出项 N/A 必须显式标识，不得以零替代。

**复验规则：** 在独立目录使用固定源码快照、相同模型资产和依赖重建数据身份与 manifest；单独运行“不重算模型预测”的 `--validate` 路径只做证据、哈希、字段与指标复算，且不得误将其称为重新独立预测的复现实验。原始结果与展示副本分开，原始数据、失败记录与实验指令均须随未来 GitHub PR 的提交说明和可审计证据路径交付；GitHub PR 属于后续实施/提交阶段，需另行获准方可创建或更新。

------------------------------------------------------------------------

## 15. Codex Implementation Handoff

本节是供 Codex 在**另行获准的各工程阶段**使用的实施规格，并非无条件执行授权。Candidate v0.3.2 **已经冻结**；代码修改、单元测试、模型调用、数据生成及 Git/PR 操作必须受相应阶段的明确授权约束，Pilot 和 Test 还需分别单独批准。Codex 须按获准范围及以下依赖顺序推进，不得自行跨阶段。

### Phase A：独立配置和 DGP

- 建立 `star_v2/` 独立模块；
- 不修改旧 `e1_5_star.py`、`phase5_large_network.py`、模型权重或旧结果；
- 实现 StarV2Config 与独立 SeedSequence；
- 实现机制、pair、真值生成；
- raw factual outcomes float64；J adapter 仅在模型调用边界转换为 float32；v1 数据不修改；
- 实现 Fixture A 48 组合穷举测试，验证 HT 权重 4；正式 Star 权重 8；
- 完整确定性单元测试。

### Phase B：契约与 Support

- 实现 TaskSpec / EstimationRequest / EstimateBundle；
- 实现标准 `[N,2,2]` mu；
- 实现 node-level population/design/empirical support；
- 实现 center/leaf group counts；
- 实现状态码和失败保留。

### Phase C：普通 baseline

- State Mean；
- pair-level HT；
- F-z2；
- Lin-z2；
- State Mean/Hájek 等价性测试；
- HT 解析概率、Fixture A 48 组合穷举及正式 Star 模拟无偏性测试；
- F-z2/Lin-z2 仅实现 Pair ATE 主比较，不扩展为节点级主结果；
- 回归 estimability 检查。

### Phase D：冻结 J/kernel

- J；
- kernel-old；
- kernel-new；
- asset hash 校验；
- 五数组输入；
- pair-level context mean；
- 任一臂失败则 pair 失败；
- 不更新模型参数。

### Phase E：Oracle

- Oracle-context；
- Oracle-pair；
- 25 维 design；
- float64 difference constraint；
- prior whitening + SVD nullspace；
- 解析后验核验。

### Phase F：统计与验收

- pair ATE 指标（pooled RMSE、signed Bias、absolute Bias、ACB、within/between variance，严格机制等权）；
- node-level mu/ITE 指标、中心—叶子效应差异、自然 Support 分层误差（不新增人工干预）；
- J interval coverage（失败区间计入每个效应的 1,200 分母，三种总体效应各自 ≥90%）；
- mechanism-block Bootstrap（10,000 次，每次整体重抽 200 个机制块）；
- strict failure-aware ranking；
- 按统一计时边界保存 pair-runtime、posterior interval runtime、CPU/GPU 峰值内存及环境说明；
- `make_figures.py` 和 `make_table.py` 确定性汇总、CSV/LaTeX/图表验收；
- manifest/hash/source snapshot；
- 完整验收器。

### Phase G：Pilot

仅在用户单独批准且实施权限明确后运行 5×2 Pilot。保留全部失败和重跑 attempt。Pilot 验收通过后，再冻结实现并申请正式 Test 授权。

------------------------------------------------------------------------

## 16. 版本演进与修订记录

### 16.1 Candidate v0.1 → v0.2 变更表

| 主题             | v0.1                                   | v0.2                                                                        |
|------------------|----------------------------------------|-----------------------------------------------------------------------------|
| 普通方法         | 存在方法集合与 R6 决议不完全一致       | 固定七种：State Mean、HT、F-z2、Lin-z2、kernel-old/new、J                   |
| Hájek            | 列为候选方法                           | 仅作 State Mean 等价性诊断                                                  |
| HT               | 使用平均臂概率表达，容易与单臂概率混淆 | 明确 pair inclusion `q=0.5`、observation `r=0.25`、权重 8                   |
| F-z2             | 候选定义不够明确                       | 四状态截距 + 共享二维斜率                                                   |
| Lin-z2           | 定义不够明确                           | 四状态截距 + 各状态独立二维斜率                                             |
| Baseline adapter | 容易误用旧单上下文代码                 | 明确 pair-level 联合输入和新适配器                                          |
| RNG              | 具体 split/identity 未注册             | 独立 StarV2Config + SeedSequence cohort/channel                             |
| 新 Test          | 仍有旧 dev 继承风险                    | 不调用旧 dev 机制入口                                                       |
| Oracle           | Oracle-context 与 pair 关系不够明确    | 分离 Oracle-context 与 Oracle-pair                                          |
| Oracle-pair      | 只有概念推导                           | 给出共享噪声、先验白化、SVD 零空间后验方案                                  |
| 原始 Y dtype     | 旧流程 float32                         | v2 原始事实 Y 保存 float64                                                  |
| Support          | 四类概念已区分                         | 明确 context 与 pair 的 node-level mask 规则                                |
| Calibration      | 预留 100 mechanisms                    | 主方案不新增 Calibration / j-calibrated                                     |
| Coverage         | 90%分层硬门槛未明确                    | 三个总体效应分别为主要目标，center_t 分层为诊断                             |
| Provenance       | 未核实                                 | 明确 `PROVENANCE_PARTIALLY_DOCUMENTED` / `INDEPENDENCE_UNVERIFIED` 声明边界 |

### 16.2 Candidate v0.2 → v0.3 最终整合修订

| 主题                | Candidate v0.2                                 | Candidate v0.3                                                                                |
|---------------------|------------------------------------------------|-----------------------------------------------------------------------------------------------|
| RNG 注册            | root seed、protocol code、cohort code 为候选值 | 用户已批准 `20261030`、`2026103001`、Pilot `1`、Test `2`；固定 NumPy `SeedSequence` + `PCG64` |
| 正式 Star HT        | pair HT 权重 8                                 | 保留 8，并增加明确的四节点 Fixture A：48 组合、权重 4；权重按设计参数计算，不硬编码           |
| 原始 outcomes dtype | float64 方案已提出，但 adapter 边界不够完整    | 原始 factual outcomes 保存 float64；J adapter 内部转换 float32；v1 不修改                     |
| F-z2/Lin-z2         | Pair ATE-only 能力表已有定义                   | 明确仅参与 Pair ATE 主比较，不强制节点级主结果                                                |
| 区间覆盖率          | 90% 候选规则及失败分母表述尚需统一             | 三种总体 ATE 覆盖率各自 ≥90%；失败区间按未覆盖计入每效应 1,200 分母                           |
| 失败规则            | 失败感知，不能静默删除                         | 明确点估计任务、interval request 和重跑 attempt 均保留；失败不能通过成功子集获得正式排名      |
| 统计单位            | mechanism-block Bootstrap 已提出               | 明确 mechanism 为独立单位；每次 Bootstrap 整体重抽机制的 3 pairs、两臂及所有方法记录          |
| 实验范围            | 对 OOD/Support 压力扩展保留候选讨论            | 明确不新增 Mechanism OOD、人工 Support pressure 或 Calibration cohort                         |
| 任务计数            | 分项列出部分请求                               | 补齐 Pilot/Test 的 Pair ATE、context prediction、interval、Oracle 请求总数并交叉核算          |
| 状态与权限          | Candidate v0.2 待审阅                          | Candidate v0.3 当时尚待冻结；这仅是历史版本状态，不代表当前 v0.3.2 的 `FROZEN` 状态              |

### 16.3 Candidate v0.3 → v0.3.1 定向复核修订

| 主题 | 新增/修订 | 科学地位 |
|---|---|---|
| 数学公式 | 修复 v0.3 中被转成 Markdown 标题的九处 display math 和转义符，统一节点索引为 `0..N-1` | 排版及接口一致性，不改变原设计 |
| 完整指标 | 增加 `absolute_bias`，区分 `ACB`、机制内方差和机制间波动；补全运行时间/峰值资源 | 对齐任务和学长转述要求 |
| Identification 接口 | `status` 继承上游 `POINT/NOT_IDENTIFIED`，新增执行/输出状态与 `sampling_uncertainty` | 对齐原始 Estimation PDF |
| Pair 契约 | `PairTaskSpec`/`PairEstimateBundle` 定义 task identity、两臂引用、失败传播、Pair ATE 和区间语义 | 收紧实施条件 |
| E1.5 核心分析 | 增加 center–leaf 效应差异与自然 Support 分层误差 | 仅后处理既定 Test，不增加 cohort |
| Oracle/Fixture | 明确真实连续暴露下 HT 条件期望与固定潜在状态表测试的区别；收紧数值 reference test | 防止伪无偏性及 dtype 漏洞 |
| 交付 | 修正目录缩进、明确图表/表格脚本、复现和 PR 验收 | 不包含本轮代码/PR 授权 |

------------------------------------------------------------------------

### 16.4 Candidate v0.3.1 → v0.3.2 终审勘误

- 修复状态定义、条件均值与 Support 指示函数的数学转义/显示问题，不改变目标量或 DGP。
- 明确 State Mean 的**逐效应**可估计性：规范四状态 `01` 的缺失不自动否决三个不使用 `01` 的 ATE 对比；逐效应失败传播与测试矩阵同步更新。
- 唯一确定机制级 Bootstrap 的 Test cohort 全局 SeedSequence 身份、抽样索引顺序及配对重抽规则；无需改变此前批准的 root seed、cohort、channel 规则。
- 补全 `empirical_support_mask=false` 的下游普通 plug-in 决策闸门；仍不新增决策实验、人工 Support 压力测试或任何数据。
- v0.3.2 终审勘误初稿仅完成静态审阅；随后于 2026-10-30 经用户明确批准正式冻结，未以协议冻结替代源码测试或 Pilot/Test。

------------------------------------------------------------------------

## 17. 决议台账与当前未决事项

### 17.1 已继承的 R3–R7 决议

- **R3 — Oracle 方法：** PR #3 Oracle 是参考，不是唯一实现要求；保留 Oracle-context 与 Star-specific Oracle-pair 两种接口。
- **R4 — 共同事实数据预算：** 所有普通方法共享同一 pair 的事实数据；每个 context 50 个观测节点，两臂共 100 条记录、50 个共同节点、共享噪声。不同方法可以联合拟合或分别推断，但必须披露统计利用方式差异。
- **R5 — 失败感知评价：** 所有预请求任务保留完整记录；失败不得静默删除；需要两臂的配对方法任一臂失败则该 pair 失败；失败区间留在区间请求分母。
- **R6 — 普通方法集合：** 七种普通方法为 State Mean、HT、F-z2、Lin-z2、kernel-old、kernel-new、J；Hájek 仅作 State Mean 等价性诊断。
- **R7 — 独立 v2 模块与版本证据：** v2 使用独立模块、数据身份和 RNG 命名空间；v1 历史结果与模型资产不修改；实施时记录源码 commit、资产哈希、配置、随机身份、任务清单和产物哈希。

### 17.2 Candidate v0.3.2 已批准的最终参数与方法决议

| 决议项              | 已批准内容                                                                                           |
|---------------------|------------------------------------------------------------------------------------------------------|
| 拓扑与规模          | 无向无自环 Star；N=200；中心 node 0；叶子 node 1–199；每 context M=50                                |
| Pair 结构           | 每 pair 两个 context，中心处理分别为 0 与 1；共享 X、图、机制、叶子处理、噪声、观测节点索引          |
| Pilot/Test          | Pilot 5 mechanisms × 2 pairs；Test 200 mechanisms × 3 pairs                                          |
| 普通方法            | State Mean、HT、F-z2、Lin-z2、kernel-old、kernel-new、iPFN-J                                         |
| Hájek               | 仅 State Mean 等价性诊断，不参加排名                                                                 |
| Oracle              | Oracle-context 与 Star-specific Oracle-pair 分开报告，不参加七方法排名                               |
| 响应目标            | `mu[N,2,2]`，自身处理 t 轴在前、同伴暴露 s 轴在后，状态顺序 00、01、10、11                           |
| 效应顺序            | Direct、Spillover、Total；`ITE[N,3]` 与 `ATE[3]` 均按此顺序                                          |
| 原始 outcomes dtype | float64 保存；J adapter 内部转 float32；v1 数据不修改                                                |
| RNG                 | root_seed 20261030；protocol_code 2026103001；Pilot/Test cohort code 1/2；NumPy SeedSequence + PCG64 |
| HT 权重             | 正式 N=200 Star pair 权重 8；四节点 Fixture A 48 种穷举组合权重 4                                    |
| F-z2/Lin-z2         | 仅参加 Pair ATE 点估计比较，不强制节点级主结果                                                       |
| 区间目标            | J 三种总体 ATE 的名义 95% 区间覆盖率分别至少 90%；失败区间计入分母                                   |
| 统计单位            | mechanism 独立；每机制 3 pairs 嵌套；Bootstrap 整体重抽机制块                                        |
| 范围排除            | 不增加 Mechanism OOD、人工 Support pressure 或 Calibration cohort                                    |

### 17.3 仍须如实披露的限制

- J、kernel-old、kernel-new 与 feature transform 的训练/验证/选择来源尚未完全核实；状态继续标记为 `PROVENANCE_PARTIALLY_DOCUMENTED` / `INDEPENDENCE_UNVERIFIED`。
- 资产哈希只证明资产字节身份，不证明训练/验证/选择数据与 v2 Test 的机制独立。
- 因 provenance 尚不完整，不得声称已严格证明 unseen-topology generalization。
- F-z2/Lin-z2 的节点级 `mu/ITE` 不属于本版主结果要求；其主要评价范围是 Pair ATE。

### 17.4 当前阻断（不是新增科学问题）

- Candidate v0.3.2 科学协议已获用户明确批准并冻结；冻结文档的实际 SHA256 须在后续实施 manifest 中重新登记（与冻结前原稿不同）。
- v2 代码尚未按本协议实现；B6 Step 1–5 验收矩阵尚未运行。
- Fixture A 48 组合穷举、正式 Star HT 测试、Oracle-pair 数值测试均尚未运行。
- Pilot 尚未授权；Formal Test 尚未授权。
- 科学协议冻结不授权仓库代码、Git/PR 或任何实验执行；后续工程与确定性测试须限于各阶段明确批准范围，模型预测、Smoke/Pilot/Formal Test 及训练/校准均不能擅自进行。

## 18. 最终状态

**状态：`FROZEN`（2026-10-30 用户明确批准）**

**Candidate v0.3.2 终审勘误完成且科学协议已冻结；Stage 1 只读工程审查已通过，Pilot/Test 均未执行。**

本版完成：

- 以 Candidate v0.2 为主体继承科学定义、数学公式、方法规格、实验规模、评价指标、失败规则、接口契约、测试矩阵、Pilot/Test 计划和 Codex 实施要求；
- 对照 B6 Step 1–5 与 R3–R7 记录，保留其设计结论并区分“设计审查完成”与“代码测试已运行”；
- 对照原始 `estimation任务分配.pdf` 全部 3 页，保留 E1.5 固定 Star 子任务责任边界；
- 整合用户批准的 RNG、dtype、HT 权重、Fixture A、F-z2/Lin-z2、失败分母、J 区间覆盖率和机制级 Bootstrap 决议；
- 修复 display math/转义与接口状态冲突，补齐 absolute Bias/Variance/runtime、center–leaf 差异、自然 Support、Pair contract、图表和复现规格，并在终审中补充逐效应可估计性、Bootstrap RNG 唯一身份、外推决策闸门；
- 在静态范围核对公式、数组维度、状态/效应顺序、字段名、统计分母、Pilot/Test 任务数量和章节引用；尚未运行任何方法级确定性单元测试。

本版没有完成、也没有声称完成：

- 代码实现或仓库修改；
- Git、分支或 PR 修改；
- 新实验数据生成；
- 模型加载或预测；
- 训练或校准；
- Smoke、Pilot 或 Formal Test；
- 任何实验结果或覆盖率验收。

**冻结确认与后续边界：** 用户已于 2026-10-30 在 E1.5 Star v2 项目对话中明确批准冻结 Candidate v0.3.2；本文件是该科学版本的 `FROZEN` 状态整理版，**未修改科学设计与实验注册值**。状态说明和目录示例的更新改变了文件原始字节，故本文件 SHA256 与冻结前原稿不同；后续 Codex、manifest、校验脚本均须使用**本文件的实际 SHA256**，原稿 SHA256 仅供溯源。协议冻结不证明代码/数值测试已经通过，不授权 Pilot 或 Formal Test，也不证明模型训练/Test 完全独立。
