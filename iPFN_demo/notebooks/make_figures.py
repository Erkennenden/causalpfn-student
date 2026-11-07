import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse

def build_figure(rows):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 直接读取你的 864 点大表
    csv_path = r"D:\causalpfn-student\iPFN_demo\ultimate_results.csv"
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # 画出所有 864 个点，横轴为密度，纵轴为 RMSE，用半透明散点显示
    sns.stripplot(data=df, x='Degree', y='RMSE', color='#2166AC', alpha=0.5, jitter=0.2, size=4, ax=ax, label='Individual Runs (864)')
    
    # 用折线把每个密度的平均值连起来，突出趋势
    means = df.groupby('Degree')['RMSE'].mean().reset_index()
    ax.plot(means['Degree'], means['RMSE'], color='#d62728', marker='o', linewidth=2, label='Mean RMSE')
    
    ax.set_title("E1.1 Expanded Results: 864 Individual Runs Across Densities", fontweight='bold', fontsize=14)
    ax.set_xlabel("Average Degree (Density)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    
    return fig, []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help='Ignore this')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
    
    filename = 'ate_accuracy'
    fig, records = build_figure([])
    
    fig.savefig(args.output.parent / f'{filename}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✅ 包含 864 个散点的图表已成功生成至: {args.output.parent}")

if __name__ == '__main__':
    main()