> **V2 scaling 新入口：请先阅读 [V2_先看这里.md](V2_先看这里.md)。包含数据量、模型容量、训练步数及多种子统计；下文保留原有实验说明。**

# 训练规模图（Scaling）

本模块已经接到当前随机函数 DGP、三个无噪声 CEPO 标签及其 NLL；与主实验使用同一固定 ER 图。旧 scaling checkpoint / results 的版本号不同，当前报告器会拒绝混合。

它是独立实验，不随 40,960 任务的主训练自动启动。五组配置依次比较：

| 组 | 模型 | 任务库大小 |
|---|---|---|
| A | medium：4 层、宽度 64 | 256 |
| B | medium | 1,024 |
| C | medium | 4,096 |
| D | small：2 层、宽度 32 | 4,096 |
| E | large：8 层、宽度 96 | 4,096 |

同 data seed 下训练任务 ID 嵌套，验证/测试任务固定且与训练独立。比较时固定优化步数、batch 和测试集；各任务库完整遍历的次数会不同，所以它回答的是固定更新预算下的比较。默认 2,000 步、batch=4；这是一个可调的起点，并不是经过证明充分的训练预算。

在容器的项目目录运行以下命令，会保存完整日志并在后台串行跑五组；每组单卡，不是主实验的 8 卡 DDP：

```bash
mkdir -p runs
nohup bash -c 'set -e; python -m scaling_v1 sweep --device musa --steps 2000 --output runs/scaling_cepo; python -m scaling_v1 report runs/scaling_cepo --plot' > runs/scaling_cepo.log 2>&1 < /dev/null &
```

输出：
- `summary.csv`：实际训练步数、真正看过的不同任务数、任务呈现总数、参数量和三个效应的 ITE/ATE 误差。
- `scaling_tasks_group1.png`：横轴训练任务库大小，纵轴各效应 ITE RMSE。
- `scaling_parameters_group1.png`：横轴模型参数量，纵轴各效应 ITE RMSE。
- `training_curves_group1.png`：验证 CEPO NLL 随更新步数/训练计算时间变化。

若实际看到的不同任务数不足任务库大小，图标题会显示 PARTIAL TRAINING / preview only；不能把只跑 2 步的试跑图解读为训练规模定律。至少先看 summary.csv 的 unique_train_tasks_seen 和完整验证曲线，再判断训练预算是否足够。

默认只有 seed=2026，不会画假的重复实验误差条。需要比较稳定性时可用 `--seeds 2026 2027 2028`，报告器按兼容配置分组保存，不把不同预算的结果拼成一条线。
