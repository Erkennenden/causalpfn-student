# causalPFN 本地权重与推断

当前显示名称为 `causalPFN`，保留原协变量输入仅为 X 的 S-learner 实现。
入口包装位于 `pfn_interference/baselines/causalpfn_direct.py`，核心推断位于原样保留的
`pfn_interference/baselines/causalpfn_xonly.py`，权重为 `checkpoints/causalpfn_v0.pt`。

每张图使用 factual X、处理 T、观测 Y 拟合原梯度提升检索辅助模型，
再用冻结预训练模型推断，不微调其权重。输出 Direct 节点效应与全节点平均 ATE。
原 X+E 暴露适配版已删除。模型与检索实现沿用包内许可，见
`pfn_interference/third_party/causalpfn/LICENSE`。依赖 scikit-learn，不需要安装 FAISS、
Hugging Face 或另行下载权重。完整方法说明见 [基线 README](../README.md)。
