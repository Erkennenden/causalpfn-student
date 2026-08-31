# 8 张 MUSA 训练与 PFN-only 测试

当前版本只保留 Interference-CausalPFN：ER network/topology embedding + 普通 Transformer self-attention + Direct/Spillover/Total GMM heads。

旧模型结构的 checkpoint 不能用于本版本，需要重新训练。

## 启动

```bash
bash run_8musa_train.sh
```

默认设置：

- 8 张 MUSA；
- 每卡 batch size = 16，总 batch size = 128；
- N = 300；
- 只使用 ER 图，边概率为 0.02；
- 训练 episode = 25600，80%/20% train/validation；
- 125 epochs；
- test seed = 2026，64 个 episode；
- 测试阶段只输出 PFN 的 Direct / Spillover / Total ITE/ATE 结果，不运行任何比较方法。

## 查看训练

启动后终端会打印 log 路径，也可以：

```bash
cat .latest_train_log
cat .latest_train_pid
cat .latest_result_dir
```

然后：

```bash
tail -f "$(cat .latest_train_log)"
```

## 查看最终 PFN 结果

```bash
RESULT_DIR="$(cat .latest_result_dir)"
python -m json.tool "$RESULT_DIR/algorithm2_summary.json"
head -n 20 "$RESULT_DIR/algorithm2_dataset_metrics.csv"
```
