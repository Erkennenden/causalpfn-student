# 网络干扰下的 PFN 估计示例

这个例子模拟 1000 名销售员使用 AI 助手：一个人的连续业绩既受自己是否使用 AI 影响，也受合作同事的使用情况影响。程序生成网络数据，用 PFN 估计四种处理／暴露状态的平均响应，再计算直接、溢出和总效应，与模拟真值及基线比较。

## 从这里开始

- **看讲解：**下载 [HTML 阅读版](notebooks/er1000_student_walkthrough.html)，用浏览器打开。文字、公式和图表已经保存，无需运行 Python。
- **看代码或运行：**打开 [Notebook](notebooks/er1000_student_walkthrough.ipynb)。按“网络 → 数据生成 → 暴露 → 真值 → 训练与推断 → ITE / ATE 评测”的顺序阅读。

## 按流程读源码

所有源码集中在 `pfn_interference/`。先按下表阅读，运行入口统一为 `python -m pfn_interference`。

| 顺序 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 1. 网络 | [graph.py](pfn_interference/graph.py) | 固定图文件 → 邻接矩阵、节点度数和图摘要 |
| 2. 数据生成 | [dgp.py](pfn_interference/dgp.py) | 网络与随机种子 → 特征、处理、暴露、连续结果和模拟真值 |
| 3. 目标量 | [estimands.py](pfn_interference/estimands.py)、[cepo.py](pfn_interference/cepo.py) | 处理／暴露状态 → 四状态响应、直接／溢出／总效应及训练损失 |
| 4. 任务划分 | [data.py](pfn_interference/data.py) | 图、DGP 与任务编号 → 可重复的训练／验证／测试任务及缓存 |
| 5. 模型 | [model.py](pfn_interference/model.py) | 观测数据、网络与查询 → 四个状态的 GMM 预测分布 |
| 6. 训练 | [training.py](pfn_interference/training.py) | 任务与训练配置 → 权重、训练记录和验证曲线 |
| 7. 评测 | [evaluation.py](pfn_interference/evaluation.py) | 权重与测试任务 → PFN／基线的 ITE、ATE 和误差表 |

`training.py` 中的 `RunConfig` 是训练默认配置。换图时先看 `graph.py` 和 `data.py`；修改结果机制看 `dgp.py`；增加基线看 `baselines/` 及 `evaluation.py`。

## 示例配置

| 项目 | 设置 |
| --- | --- |
| 网络 | 固定 ER 图，1000 个节点，连边概率 0.02，图 seed=12345 |
| 处理分配 | 各节点独立 Bernoulli(0.5) |
| 结果 | 连续型；每份数据的处理系数 a ~ Uniform(4,8)，噪声标准差 4 |
| 同伴状态 | 接受处理的邻居超过一半为高暴露，否则为低暴露 |
| 模型 | 10 层、128 维、4 heads、FFN=512；四状态各 5 个 GMM 分量 |
| 训练 | 40960 个任务、5120 个验证任务、125 个 epoch，每进程 batch size=16 |
| 随机种子 | 数据 seed=12345，训练 seed=0；测试 seeds=2025–2029，每个生成 2 份数据 |
| 输出 | 四状态响应，以及 Direct、Spillover、Total 的 ITE / ATE 与误差指标 |

Notebook 与主训练入口统一使用 10 层、128 维模型，保留网络与 DGP 图解；模型输出需加载匹配权重后运行生成。第 13 节另附固定系数 a=6 的结果展示示例，不是当前 Notebook 运行产生的结果。

## 需要运行时

使用 Python 3.11，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt -r requirements.txt
python -m jupyterlab
```

Windows 将激活命令换成 `.venv\Scripts\Activate.ps1`。也可以用 VS Code 打开 Notebook，选择上述 `.venv` 环境，从上到下运行。Notebook 默认使用 CPU，跳过训练，从 `CHECKPOINT_PATH` 加载主模型；默认路径为 `runs/training/model_best.pt`。仓库未附带 10 层主模型权重，需要提供匹配权重，或设置设备后开启 `RUN_TRAINING`。仅阅读已保存的图表无需运行。MUSA 训练使用已配置 `torch_musa` 的服务器环境。

也可在项目根目录启动训练与评测（以下使用 CUDA；按机器改成 `cpu` 或 `musa`）：

```bash
# 检查任务先验，输出到 runs/prior
python -m pfn_interference diagnose --output runs/prior

# 训练，输出权重和训练记录
python -m pfn_interference train --device cuda --output runs/training

# 在相同测试任务上比较 PFN 和全部基线
python -m pfn_interference evaluate --device cuda \
  --checkpoint runs/training/model_best.pt --output runs/evaluation
```

Notebook 与命令行使用相同的模型、任务数和种子默认配置；不同进程数会改变有效 batch size。

Linux / MUSA 服务器的一次性完整流程入口为 `bash scripts/train.sh`；已有权重只评测时用 `bash scripts/evaluate.sh CHECKPOINT OUTPUT_DIR`。Docker 后台启动脚本为 `scripts/launch_musa.sh`。

评测后先看 `runs/evaluation/all_results.csv`，需要追溯单个节点或数据集时，再看其下 `linear_b_random_a/` 中的原始 ITE / ATE 表。

目录整理会改变源码指纹；旧运行目录和旧权重需与其原始源码版本配套，本版训练请使用新的输出目录。

## 主要文件

```text
er_estimation_demo/
├── README.md
├── notebooks/          # 逐步讲解、HTML、图表
├── pfn_interference/   # 源码，按上面的 1—7 阅读
│   ├── baselines/      # 文献基线
│   ├── simulation/     # DGP 底层组件
│   ├── metrics/        # 通用评测组件
│   ├── experiments/    # 其他实验工具；主流程无需从这里开始
│   └── third_party/    # 第三方代码与许可证
├── scripts/            # 服务器启动脚本
├── data/               # 固定图
├── checkpoints/        # CausalPFN 比较基线的权重
├── tests/              # 程序校验
└── runs/               # 运行后生成的缓存、主模型权重和结果，不提交 Git
```

在项目根目录执行 `python -m pytest -q` 检查程序。运行 Notebook 时保留完整项目目录，第三方来源与许可证保留在对应代码目录。
