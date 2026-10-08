causalCFM—pipeline
请阅读 README_CN.md。正式入口 run_causalfm.sh；宿主机入口 launch_causalfm_docker.sh。
默认 固定 ER(20,0.5,seed=12345) 图、B DGP、4 个 GMM mu、4 项平均 NLL、10 份测试数据。NUM_LAYERS 调整主模型深度，baseline 不变。

图种子 12345 只决定保存的邻接矩阵；训练/测试数据种子分别控制 X、T、噪声，数据任务不会重抽图。旧 K10 checkpoint 请重新训练。
