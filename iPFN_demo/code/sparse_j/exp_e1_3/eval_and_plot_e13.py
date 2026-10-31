# E1.3: evaluate ATE metrics and plot figures
#汇总结果（含NAN检查）
import numpy as np
from pathlib import Path
import pandas as pd

# ========== 【每次修改文件夹编号】 ==========
dir_list = [
    "/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/full_0_4_v1",
    "/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/full_10_14_v1",
]
out_csv_name = "/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/batch_01.csv"
# =====================================

rows = []

for root_dir in dir_list:
    p = Path(root_dir)
    print(f"\n正在检查目录：{root_dir}")
    if not p.exists():
        print(f" 不存在：{root_dir}")
        continue
    truth_dir = p / "truth"
    pred_dir = p / "predictions"
    if not truth_dir.exists() or not pred_dir.exists():
        print(f" truth或者predictions文件夹缺失")
        continue

    truth_npzs = sorted(list(truth_dir.glob("*.npz")))
    print(f" truth找到文件数量：{len(truth_npzs)}")
    for t_npz in truth_npzs:
        ctx_id = t_npz.stem
        t_data = np.load(t_npz)
        truth_arm = t_data["arm_means"]
        ate_true = truth_arm[:,1] - truth_arm[:,0]
        ate_true_val = float(ate_true.mean())

        pred_files = list(pred_dir.glob(f"{ctx_id}__*.npz"))
        for pf in pred_files:
            method_name = pf.stem.split("__")[-1]
            p_data = np.load(pf)
            pred_arm = p_data["arm_means"]
            ate_pred = pred_arm[:,1] - pred_arm[:,0]

            # 新增NaN校验
            if np.any(np.isnan(ate_pred)):
                pred_mean = np.nan
                pred_std = np.nan
                ci_low = np.nan
                ci_high = np.nan
                cover = np.nan
            else:
                pred_mean = np.mean(ate_pred)
                pred_std = np.std(ate_pred)
                ci_low = pred_mean - 1.96 * pred_std
                ci_high = pred_mean + 1.96 * pred_std
                cover = 1 if (ci_low <= ate_true_val <= ci_high) else 0

            rows.append({
                "folder": str(root_dir),
                "context": ctx_id,
                "method": method_name,
                "ate_true": ate_true_val,
                "ate_pred_mean": float(pred_mean),
                "ate_pred_std": float(pred_std),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "coverage": cover
            })

df_batch = pd.DataFrame(rows)
df_batch.to_csv(out_csv_name, index=False)
print(f"\n 本批次保存完成：{out_csv_name}，一共 {len(df_batch)} 条")
print(df_batch.head(10))

eval_and_plot_e13.py
#复制结果汇总文件夹
from shutil import copyfile

src = "/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/batch_01.csv"
dst = "/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/final_all_result.csv"
copyfile(src, dst)
print(" 已经复制生成 final_all_result.csv")

#生成指标汇总表格（过滤NAN）
import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/final_all_result.csv")
df_valid = df.dropna(subset=["ate_pred_mean"])
print(f"总记录：{len(df)}，有效记录：{len(df_valid)}")

res_table = df_valid.groupby("method").agg(
    total_context=("context", "count"),
    coverage_rate=("coverage", "mean"),
    ate_bias=("ate_pred_mean", lambda x: np.mean(x - df_valid.loc[x.index, "ate_true"])),
    ate_rmse=("ate_pred_mean", lambda x: np.sqrt(np.mean((x - df_valid.loc[x.index, "ate_true"]) ** 2)))
).reset_index()

print("===== 指标汇总表（仅有效样本） =====")
print(res_table.round(4))
res_table.to_csv("/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/summary_table.csv", index=False)

#画图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/final_all_result.csv")
df_valid = df.dropna(subset=["ate_pred_mean"])

sns.set_style("whitegrid")
# 1. True vs Pred ATE
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_valid, x="ate_true", y="ate_pred_mean", hue="method", alpha=0.6)
plt.plot([df_valid.ate_true.min(), df_valid.ate_true.max()],
         [df_valid.ate_true.min(), df_valid.ate_true.max()], "k--")
plt.xlabel("True ATE")
plt.ylabel("Predicted ATE")
plt.title("True vs Predicted ATE")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/ate_scatter.png", dpi=300)
plt.show()

# 2. Coverage bar plot
plt.figure(figsize=(8, 4))
summary = df_valid.groupby("method")["coverage"].agg(["mean","count"]).reset_index()
summary.columns = ["method","coverage_rate","valid_count"]
sns.barplot(data=summary, x="method", y="coverage_rate")
plt.ylim(0, 1)
plt.axhline(y=0.95, color="red", linestyle="--", label="Nominal 95%")
plt.ylabel("Coverage Rate")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/causalpfn-student-main/iPFN_demo/code/sparse_j/coverage_bar.png", dpi=300)
plt.show()

