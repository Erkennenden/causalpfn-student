# 网络干扰下的 PFN 估计示例

这个例子模拟 1000 名销售员使用 AI 助手：一个人的连续业绩既受自己是否使用 AI 影响，也受合作同事的使用情况影响。程序生成网络数据，用 PFN 估计四种处理／暴露状态的平均响应，再计算直接、溢出和总效应，与模拟真值及基线比较。

## 从这里开始

- **看讲解：**下载 [HTML 阅读版](notebooks/er1000_student_walkthrough.html)，用浏览器打开。文字、公式和图表已经保存，无需运行 Python。
- **看代码或运行：**打开 [Notebook](notebooks/er1000_student_walkthrough.ipynb)。按“网络 → 数据生成 → 暴露 → 真值 → 训练与推断 → ITE / ATE 评测”的顺序阅读。

## 示例配置

| 项目 | 设置 |
| --- | --- |
| 网络 | 固定 ER 图，1000 个节点，连边概率 0.02，图 seed=12345 |
| 处理分配 | 各节点独立 Bernoulli(0.5) |
| 结果 | 连续型；每份数据的处理系数 a ~ Uniform(4,8)，噪声标准差 4 |
| 同伴状态 | 接受处理的邻居超过一半为高暴露，否则为低暴露 |
| 随机种子 | 数据 seed=12345，训练 seed=0；测试 seeds=2025–2029，每个生成 2 份数据 |
| 输出 | 四状态响应，以及 Direct、Spillover、Total 的 ITE / ATE 与误差指标 |

Notebook 默认载入已训练的教学小模型：2 层、32 维，32 个训练任务、8 个验证任务、6 个 epoch。完整训练默认采用 10 层、128 维模型；两者配置在 Notebook 中分别说明。第 14 节另附固定系数 a=6 的结果展示示例。

## 需要运行时

使用 Python 3.11，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Windows 将激活命令换成 `.venv\Scripts\Activate.ps1`。也可以用 VS Code 打开 Notebook，选择上述 `.venv` 环境，从上到下运行。默认使用 CPU 并加载配套权重；重新训练由 Notebook 中的开关控制。运行完整文献基线时，另外安装 `requirements.txt`。

## 主要文件

| 文件／目录 | 作用 |
| --- | --- |
| `notebooks/` | 完整讲解、图表、结果和教学权重 |
| `data/`、`fixed_er_graph.py` | 固定网络与图读取 |
| `linear_b_dgp.py` | 连续结果 DGP 与模拟真值 |
| `estimands.py`、`cepo.py` | 暴露状态、四状态响应与效应计算 |
| `train_local_network_interference.py`、`causalfm_experiment/` | 模型、训练和评测 |
| `baselines/`、`evaluation/` | 基线方法与统一评测 |
| `tests/` | 程序校验 |

运行 Notebook 时保留完整项目目录。第三方来源与许可证保留在对应代码目录。
