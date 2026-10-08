# er_estimation_demo

固定 ER 图：**1000 个节点、p=0.02、图种子 12345**。主模型：**10 层、128 维、4 个注意力头、FFN=512**。线性 B 的处理项中，把固定的 6 改为每份数据抽取一次的 **a ~ Uniform(4,8)**。保留四个 μ 的 GMM、四项等权 NLL、完整 baseline，以及默认 10 份测试数据。

本版默认 units 从 300 改为 1000，主模型需要在 1000 节点图上从头训练。旧的固定系数、随机函数、20 节点或三 μ checkpoint 不用于本版续训或评估；baseline 自带的 causalPFN checkpoint 保持原样。

## 图与数据生成

图保存在 `data/fixed_er_N1000_p0.02_seed12345.json`，训练、验证、测试始终读取同一个邻接矩阵。沿用原项目的 PyTorch 上三角 Bernoulli 采样与无孤立节点拒绝重采样规则：第一次抽样即无孤立节点，直接通过并固定保存。因此准确描述是 **ER(1000,0.02) 在无孤立节点条件下的一次固定抽样**。

实际有 9914 条边，平均度 19.828，节点度数 9–37，图连通。邻接矩阵 SHA256：

```text
df22ac22826655d22fe20edc830d4f981815a4339f28b7d92fbc5f5c329fbe13
```

不同随机数库即使使用相同 seed，也不保证得到同一张图，以包内文件为准。

```text
每份数据抽一次 a ~ Uniform(4,8)
X_i ~ Normal(0,1)
T_i ~ Bernoulli(0.5)，节点间独立
neighbor_X_i = sum_j A_ij X_j / d_i
E_i = sum_j A_ij T_j / d_i
Y_i = 1-X_i-3*neighbor_X_i+(a+0.2*exp(X_i))*T_i-0.9*E_i+epsilon_i
epsilon_i ~ Normal(0,4^2)，节点间独立
```

同一份数据的所有节点、事实结果和四个反事实 μ 共用同一个 a。a 由 `(data_seed, split, task_id)` 的专用随机流决定；不受调用顺序、分布式 rank、观测噪声大小影响。训练、验证、测试使用独立 split 随机流。同一训练任务在不同 epoch 重放时，其 a 保持一致；不会每个节点、每次查询或每个 epoch 重新抽。

仅将 a 写入缓存元数据、先验诊断和测试结果记录，**不作为 PFN 或 baseline 的输入特征**。模型 token 仍为 X、T、观测 Y、E、归一化度数五列。训练监督仍使用无观测噪声的四个 μ 标签。

## 四个 μ 与效应

查询的第二个坐标是多数状态 `S=1{K>floor(d_i/2)}`，不是直接令连续暴露 E 等于 0 或 1。μ 对状态内精确的 Binomial(d_i,0.5) 条件权重积分。

```text
b_i = 1-X_i-3*neighbor_X_i
tau_i = a+0.2*exp(X_i)
e_low_i  = E[K/d_i | K <= floor(d_i/2)]
e_high_i = E[K/d_i | K >  floor(d_i/2)]
mu00 = b_i       -0.9*e_low_i
mu01 = b_i       -0.9*e_high_i
mu10 = b_i+tau_i -0.9*e_low_i
mu11 = b_i+tau_i -0.9*e_high_i

Direct    = mu10-mu00
Spillover = mu11-mu10
Total     = mu11-mu00
```

每个节点的每个 μ 单独输出 5 分量 GMM：权重 pi、分量均值 m、标准差 sigma。形状为 `[batch,1000,4,5]`，arm 顺序为 `00,01,10,11`。点预测为各分量加权均值。训练、验证、选模和学习率调度仍使用：

```text
loss = (NLL_mu00 + NLL_mu01 + NLL_mu10 + NLL_mu11)/4
```

四个分布是边际 GMM；共享网络并不等于已建模四个 μ 的联合后验。本次只随机化直接效应中的截距；spillover 机制仍为 -0.9*E，其真值在固定图上仍由节点度数决定。结果应分别报告三种效应，不能假定本次修改必然提高 RMSE 或改变方法排名。

## 训练、测试与 baseline

| 项目 | 默认值 |
|---|---|
| 主模型 | 10 层，d_model=128，4 heads，FFN=512，dropout=0 |
| GMM | 每个 μ 5 个分量 |
| 训练任务 | 40960 |
| 验证任务 | 5120 |
| Epoch | 125 |
| 并行 | 8 张 MUSA，每卡 batch=16 |
| 选模 | 四项验证 NLL 均值最小 |
| 测试 | seeds=2025、2026、2027、2028、2029，各 task_id=0、1，共 10 份 |

十份测试数据独立生成 a、X、T、噪声，使用同一张固定图；ATE、ITE、主模型和全部 baseline 使用完全相同的十份数据。测试数据不参与选模。全部支持时，每种方法/效应的 ITE support=10000/10000，ATE support=10/10。汇总 RMSE 由全部有效原始误差计算，不是每份数据 RMSE 的简单平均。

baseline 实现、结构、checkpoint 和默认拟合预算沿用原包：HyperSCI 500 epoch、Tnet 160 epoch、HyperSCI arm integration 4096 samples。ITE 保留 Localized DR-Lasso、NSI、HyperSCI、Tnet、causalPFN；ATE 保留 HT、Haj、F、L、reg-net、Tnet、causalPFN。causalPFN 仍只估计 Direct。不支持的输出保留 NaN 和实际支持率。

## 上传和启动

在 Mac 上传：

```bash
scp "$HOME/Desktop/er_estimation_demo_units1000.zip" mccxadmin@10.121.39.7:/home/mccxadmin/
```

登录服务器后，在宿主机执行以下整段，新建目录解压并复制进 musacode；容器内使用 nohup 从头训练，训练结束自动运行十份测试和全部 baseline：

```bash
bash <<'BASH'
set -euo pipefail
DEST=$(mktemp -d "$HOME/er_estimation_demo_XXXXXXXX")
unzip -q "$HOME/er_estimation_demo_units1000.zip" -d "$DEST"
cd "$DEST/er_estimation_demo"
unset TEST_SEED
PFN_CONTAINER=musacode NUM_LAYERS=10 D_MODEL=128 NUM_HEADS=4 \
  DEVICES=8 EPOCHS=125 TEST_SEEDS="2025 2026 2027 2028 2029" TEST_TASKS=2 \
  bash launch_causalfm_docker.sh
BASH
```

沿用服务器已有 Python/PyTorch/MUSA 环境，不自动安装或替换依赖。退出 SSH 后任务继续。启动器打印容器项目路径；`.latest_causalfm_job` 保存本次运行结果路径。

查看最新训练日志：

```bash
docker exec musacode bash -lc '
cd "$(cat /workspace/latest_causalfm_project.txt)"
tail -n 80 -f host.log
'
```

Ctrl+C 只退出日志跟踪。查看完成状态和结果：

```bash
docker exec musacode bash -lc '
set -e
cd "$(cat /workspace/latest_causalfm_project.txt)"
JOB=$(cat .latest_causalfm_job)
echo "结果目录：$JOB"
for f in stage.txt exit_code.txt; do
  if [ -f "$JOB/$f" ]; then printf "%s: " "$f"; cat "$JOB/$f"; fi
done
if [ -f "$JOB/evaluation/all_results.txt" ]; then
  cat "$JOB/evaluation/all_results.txt"
else
  tail -n 60 host.log
fi
'
```

完成标志为 stage=complete 且 exit_code=0。检查实际层数和数据设置可查看 `RUN/training/manifest.json`。

## 结果文件

```text
RUN/prior/prior_coefficients.csv       训练先验诊断任务的 a 与随机种子
RUN/training/
  manifest.json                       实际模型、训练、图和 DGP 配置
  model_best.pt / model_last.pt
  history.jsonl / validation_metrics.csv
  four_mu_loss_curves.png / .pdf
RUN/evaluation/
  all_results.txt / .csv               ATE/ITE 全方法汇总；txt 附四 μ 汇总
  results.json                        模型配置、DGP、测试安排等
  test_datasets.csv                   dataset_id、test_seed、task_id、a、a_seed、图哈希
  linear_b_random_a/
    pfn_mu00_predictions.csv          10000 行，每个节点的 mu00 GMM
    pfn_mu01_predictions.csv          10000 行，每个节点的 mu01 GMM
    pfn_mu10_predictions.csv          10000 行，每个节点的 mu10 GMM
    pfn_mu11_predictions.csv          10000 行，每个节点的 mu11 GMM
    pfn_mu_predictions.csv            合计 40000 行
    pfn_mu_gmm_components.csv         合计 200000 个分量记录
    pfn_mu_graph_means.csv            十份数据各自的四 μ 均值
    pfn_mu_summary.csv
    ite_unit_results.csv / ate_graph_results.csv
    per_dataset_summary.csv / summary.csv
    task_0000/ ... task_0009/          单份数据结果
```

μ 预测 CSV 中 `gmm_pi,gmm_mu,gmm_sigma` 均为长度 5 的 JSON 数组，sigma 为标准差；均值为 `sum(pi_k*mu_k)`。a 为模拟机制的真值记录，只用于审计与评价。

本版模型训练完成后可单独重新评估：

```bash
TEST_SEEDS="2025 2026 2027 2028 2029" TEST_TASKS=2 DEVICE=musa \
  bash evaluate_causalfm.sh runs/YOUR_RUN/training/model_best.pt runs/eval10_new
```

该例要求未设置 TEST_SEED。TEST_TASKS 表示每个种子的数量；设成 4 时共 20 份。继续训练同一个 run 时，必须保持原数据、模型与分布式设置；本次改了 DGP，应启动新训练。

## 验证范围与历史工具

本次 1000 节点版本的验证见 `VERIFICATION_SUMMARY.json`：使用 CPU 检查固定图、数据标签及缩短预算的主模型训练和十份测试导出。完整 baseline、正式 125 epoch 和 MUSA 训练未在本次修改中重跑。原 300 节点版本的验证记录单独保留在报告的 `previous_release_verification` 字段中。

原包的 scaling、旧 CSV、随机函数生成器及其测试作为复用/历史工具保留，其独立文档不代表本版默认流程。本版以此 README、PACKAGE_MANIFEST.json 和 causalfm_experiment 入口为准。旧运行结果、smoke checkpoint 和缓存不放入发布包；causalPFN baseline 所需的预训练权重保留。
