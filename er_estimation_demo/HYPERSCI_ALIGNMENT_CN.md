# HyperSCI 原文与当前实现核对

核对日期：2026-09-23。当前实现应描述为“基于作者公开实现、适配二元超边及多数状态效应的 HyperSCI”，不能称为原论文全部公式和实验流程的逐项复现。本次 PFN 命名和默认 DGP 标签修复均未改动 HyperSCI 算法。

## 保留的作者实现

作者代码固定版本为 `ba33ef07b16e2110faabd127bb510aed63e29ea1`。
重新下载 `src/Model.py` 后，对比 `HyperSCI` 类的语法树，与包内 `baselines/hypersci_vendor/model.py` 一致。外层 import、设备设置等按本项目环境适配。

保留协变量表示、处理掩码、超图消息传播、注意力选项及两个潜在结果输出头。当前使用一层超图注意力、两个 attention heads、hidden=32、phi_layer=2、skip=123、n_out=0；这些属于作者代码可配置的选项组合，并非其默认配置。当前标准化输入 X、节点度和观测 Y，用全部节点的 factual 数据拟合，500 epochs；这些均属于本实验设置。

## 需要明确披露的差别

1. 普通图每条边转换成大小为 2 的超边。该表示保留原图连接，但没有新增多人群体，也不提供高阶群体干扰的实验依据。
2. 原文 Eq. (1) 的个体效应在给定其他节点处理配置下比较自身处理；当前在低/高多数状态内按处理设计抽样并平均预测，输出 Direct、Spillover、Total。每个节点、每个状态默认 4096 次抽样。作者评估代码对一次 forward 返回的两个结果头作差；当前计算不同根节点处理时，会重算相应的处理相关超图表示。这些是本项目新增的反事实与效应适配。
3. 原文 §3.2 和 Eq. (15) 的文字描述包含协变量表示与干扰表示的两项平衡。固定版本作者代码 `compute_loss` 仅对模型返回的 `rep=phi_x` 计算 Wasserstein 距离。当前沿用该公开实现：标准化 factual MSE + 0.01 × Wasserstein(phi_x)，Adam weight_decay=0.01；没有另加干扰表示的平衡项。因此“跟随作者代码”与“完全逐项实现原文损失”须区分。
4. 原始数据划分、超参数和运行环境也与本项目不同。不能将当前效果宣称为复现原论文数值结果。

建议论文表格中继续使用 HyperSCI，并在方法说明中明确：采用作者公开实现，将普通边表示为二元超边，并将潜在结果预测按本研究的多数状态干预分布平均。

## 来源

- [原论文 §2–3](https://arxiv.org/html/2207.04049)
- [作者模型代码](https://github.com/jma712/HyperSCI/blob/ba33ef07b16e2110faabd127bb510aed63e29ea1/src/Model.py)
- [作者训练与评估代码](https://github.com/jma712/HyperSCI/blob/ba33ef07b16e2110faabd127bb510aed63e29ea1/src/HyperCI.py)
