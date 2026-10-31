# E1.3 稀疏干扰因果估计实验
## 实验设置
本实验在稀疏干扰因果仿真设定下，对比 j、kernel-new、kernel-old、non-interference 四种ATE估计方法。
评估指标：95%置信区间覆盖率、ATE估计偏差（ate_bias）、ATE均方根误差（ate_rmse）。

## 结果汇总表
| method | total_context | coverage_rate | ate_bias | ate_rmse |
|--------|---------------|---------------|----------|----------|
| j | 216 | 0.5000 | -0.0085 | 0.1781 |
| kernel-new | 216 | 0.5417 | -0.0143 | 0.1411 |
| kernel-old | 216 | 0.3981 | -0.1050 | 0.1805 |
| non-interference | 144 | 0.0000 | -0.2371 | 0.2874 |

## 结果说明
原始一共2160条记录，其中792条为有效估计样本，其余样本在估计时出现数值异常，返回NaN，不纳入统计。
按照要求，理想目标是95%置信区间覆盖率达到90%；RMSE越小越好，bias越接近0越好。
1. kernel-new 在本组中综合最优，RMSE最小（0.1411），同时拥有最高覆盖率54.17%；
2. j 方法的估计偏差绝对值最小（-0.0085），均值估计最贴近真实ATE；
3. kernel-old 覆盖率偏低，存在明显负偏差；
4. non-interference 基线方法在该稀疏干扰场景下失效严重，覆盖率为0，系统性低估ATE；
5. 所有方法的置信区间覆盖率均没有达到90%的预期水平，说明当前构造的置信区间偏窄，区间覆盖能力不足。

## 配套文件
- final_all_result.csv：全部有效汇总数据
- summary_table.csv：指标汇总表
- ate_scatter.png：True vs Predicted ATE散点图
- coverage_bar.png：覆盖率柱状图
- 汇总、绘图Python脚本
