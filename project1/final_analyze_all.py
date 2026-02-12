import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= 配置区 =================
FILE_MAPPING = {
    "0-shot Direct": "result_gsm8k_exp_01_no_cot_no_shot.csv",
    "0-shot CoT":    "result_gsm8k_exp_02_with_cot_no_shot.csv",
    "4-shot Direct": "result_gsm8k_exp_03_no_cot_few_shot.csv", 
    "4-shot CoT":    "result_gsm8k_exp_04_with_cot_few_shot.csv" 
}

ORDER_LIST = ["0-shot Direct", "0-shot CoT", "4-shot Direct", "4-shot CoT"]

# ================= 工具函数 =================

def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent = os.path.join("..", filename)
    if os.path.exists(parent):
        return parent
    return None

def load_and_clean(label, filename):
    path = find_file(filename)
    if not path:
        print(f" [Skip] File not found: {filename}")
        return None
    
    df = pd.read_csv(path)
    df_success = df[df['error'].isna() | (df['error'] == "")].copy()
    df_clean = df_success.drop_duplicates(subset=['question'], keep='last').copy()
    df_clean['Group'] = label
    df_clean['Result'] = df_clean['is_correct'].apply(
        lambda x: 'Correct' if 'Correct' in str(x) else 'Wrong'
    )
    return df_clean

# ================= 绘图函数 1 =================
def plot_1_distribution(df, order):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.stripplot(
        data=df, x='Group', y='tokens', hue='Result',
        order=order, hue_order=['Wrong', 'Correct'],
        palette={'Correct': '#2ecc71', 'Wrong': '#95a5a6'},
        jitter=0.25, size=5, alpha=0.6, edgecolor='white', linewidth=0.3
    )
    
    means = df.groupby('Group')['tokens'].mean()
    for i, label in enumerate(order):
        if label in means:
            val = means[label]
            plt.hlines(val, i-0.35, i+0.35, color='#e74c3c', linestyles='--', linewidth=2, zorder=5)
            plt.text(i, val + (df['tokens'].max()*0.02), f"Avg: {val:.1f}", 
                     ha='center', color='#c0392b', fontweight='bold')

    plt.title("Plot 1: Token Consumption & Result Density", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Tokens Consumed", fontsize=12)
    plt.xlabel("Experiment Groups", fontsize=12)
    plt.legend(title='Result', loc='upper left', frameon=True)
    
    plt.savefig("analysis_plot_1_distribution.png", dpi=300)
    print("Plot 1 saved: analysis_plot_1_distribution.png")

# ================= 绘图函数 2 =================
def plot_2_slope(df, order):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    sns.stripplot(
        data=df, x='Group', y='tokens', hue='Result',
        order=order, hue_order=['Wrong', 'Correct'],
        palette={'Correct': '#2ecc71', 'Wrong': '#95a5a6'},
        jitter=0.25, size=4, alpha=0.15, legend=False
    )

    stats = []
    for label in order:
        group_data = df[df['Group'] == label]
        if not group_data.empty:
            stats.append({
                'Group': label,
                'avg_tokens': group_data['tokens'].mean(),
                'accuracy': (group_data['Result'] == 'Correct').mean()
            })
    stats = pd.DataFrame(stats)

    x_indices = np.arange(len(stats))
    y_values = stats['avg_tokens'].values
    acc_values = stats['accuracy'].values
    
    plt.plot(x_indices, y_values, color='#c0392b', marker='o', markersize=10, linewidth=2.5, label='Cost Path', zorder=10)

    for i in range(len(stats)):
        plt.text(x_indices[i], y_values[i] + 15, f"Acc: {acc_values[i]:.1%}", 
                 ha='center', color='white', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#c0392b", ec="none"))
        if i > 0:
            d_acc = (acc_values[i] - acc_values[i-1]) * 100 
            d_tok = y_values[i] - y_values[i-1]
            if abs(d_tok) > 0.5:
                slope = d_acc / d_tok
                mid_x = (x_indices[i] + x_indices[i-1]) / 2
                mid_y = (y_values[i] + y_values[i-1]) / 2
                # -----------------------
                plt.annotate(f"Gain: {slope:.2f}% / tok", xy=(mid_x, mid_y), 
                             xytext=(0, -25), textcoords='offset points', ha='center', 
                             color='#2980b9', fontweight='bold', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", fc="#ecf0f1", ec="#2980b9", alpha=0.9),
                             arrowprops=dict(arrowstyle="-", color='#2980b9', alpha=0.5))

    plt.title("Plot 2: Accuracy vs. Budget Path (Slope Analysis)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Average Tokens (Cost)", fontsize=12)
    plt.xlabel("Experiment Groups", fontsize=12)
    plt.savefig("analysis_plot_2_slope.png", dpi=300)
    print("Plot 2 saved: analysis_plot_2_slope.png")

# ================= 绘图函数 3 (Performance-Budget Scatter) =================
def plot_3_efficiency_scatter(df, order):
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    # 计算每组的重心 (Accuracy 和 Tokens)
    stats = []
    for label in order:
        group_data = df[df['Group'] == label]
        if not group_data.empty:
            acc = (group_data['Result'] == 'Correct').mean()
            tok = group_data['tokens'].mean()
            stats.append({'Group': label, 'Accuracy': acc, 'Avg_Tokens': tok})
    
    stats_df = pd.DataFrame(stats)

    # 直接绘制散点图 (X=Token, Y=Accuracy)
    sns.scatterplot(
        data=stats_df, 
        x='Avg_Tokens', 
        y='Accuracy', 
        hue='Group', 
        hue_order=order,
        s=300,             
        palette='bright', 
        style='Group',     # 不同 Setting 用不同形状
        edgecolor='black', 
        linewidth=1.5
    )

    # 标注百分比
    for i in range(len(stats_df)):
        plt.text(
            stats_df.iloc[i]['Avg_Tokens'], 
            stats_df.iloc[i]['Accuracy'] + 0.02,
            f"{stats_df.iloc[i]['Accuracy']:.1%}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    plt.title("Plot 3: Accuracy vs. Token Budget Scatter", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Average Tokens Used", fontsize=12)
    plt.ylim(0, 1.1) 
    plt.legend(title='Settings', bbox_to_anchor=(1.05, 1), loc='upper left') 
    
    plt.tight_layout()
    plt.savefig("analysis_plot_3_efficiency.png", dpi=300)
    print("Plot 3 saved: analysis_plot_3_efficiency.png")
# ================= 主程序 =================
def main():
    print(">>> Analysis started...")
    all_data = []
    valid_order = [k for k in ORDER_LIST if k in FILE_MAPPING]
    
    for label in valid_order:
        df = load_and_clean(label, FILE_MAPPING[label])
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print(" No valid data found. Check file paths.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    # 生成报表
    print("\n" + "="*50)
    print(f"{'Group':<15} | {'Acc':<8} | {'Avg Tokens':<10}")
    print("-" * 50)
    for label in valid_order:
        g_df = final_df[final_df['Group'] == label]
        if not g_df.empty:
            acc = (g_df['Result'] == 'Correct').mean()
            tok = g_df['tokens'].mean()
            print(f"{label:<15} | {acc:.1%}   | {tok:.1f}")
    print("="*50 + "\n")

    # 调用绘图函数
    plot_1_distribution(final_df, valid_order)
    plot_2_slope(final_df, valid_order)
    plot_3_efficiency_scatter(final_df, valid_order) 

    print("\nAnalysis Complete! Three plots are generated.")
    plt.show()

if __name__ == "__main__":
    main()