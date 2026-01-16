import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 解决可能存在的字体问题
plt.rcParams['font.family'] = 'sans-serif'

def load_data(filename):
    if not os.path.exists(filename):
        print(f"找不到文件: {filename}")
        return None
    df = pd.read_csv(filename)
    # 1. 过滤掉报错的行 (error 列不为空的剔除)
    df = df[df['error'].isna()].copy()
    # 2. 确保索引是连续的数字，用作 X 轴
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    # 文件路径
    file_no = "data_api_without_cot.csv"
    file_cot = "data_api_with_cot.csv"

    df_no = load_data(file_no)
    df_cot = load_data(file_cot)

    if df_no is None or df_cot is None:
        print("缺少数据文件，无法绘图。")
        return

    print("数据加载成功，开始绘图...")

    # ==========================================
    # 图表 1: Token 消耗分布 (上下子图，统一坐标)
    # ==========================================
    
    # 计算全局最大 Token 数，用于统一 Y 轴刻度
    max_token_val = max(df_no['tokens'].max(), df_cot['tokens'].max())
    # 留出 10% 的顶部空间
    y_limit = max_token_val * 1.1

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- 子图 1: Without CoT ---
    avg_no = df_no['tokens'].mean()
    sns.barplot(x=df_no.index, y=df_no['tokens'], ax=ax1, color='#4c72b0', alpha=0.8) # 蓝色
    ax1.axhline(avg_no, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_no:.1f}')
    ax1.set_title("Without CoT - Token Usage per Question", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Tokens")
    ax1.set_ylim(0, y_limit) # 统一刻度
    ax1.legend(loc='upper right')
    # 隐藏过密的 X 轴标签，每隔 20 个显示一个
    for label in ax1.get_xticklabels():
        label.set_visible(False)

    # --- 子图 2: With CoT ---
    avg_cot = df_cot['tokens'].mean()
    sns.barplot(x=df_cot.index, y=df_cot['tokens'], ax=ax2, color='#dd8452', alpha=0.8) # 橙色
    ax2.axhline(avg_cot, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_cot:.1f}')
    ax2.set_title("With CoT - Token Usage per Question", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Tokens")
    ax2.set_xlabel("Question Index (0-250)")
    ax2.set_ylim(0, y_limit) # 统一刻度
    ax2.legend(loc='upper right')
    
    # 设置 X 轴刻度显示间隔
    import matplotlib.ticker as ticker
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))

    plt.tight_layout()
    plt.savefig("token_analysis.png", dpi=300)
    print(">> [成功] Token 消耗图已保存为: token_analysis.png")


    # ==========================================
    # 图表 2: 准确率对比 (简单柱状图)
    # ==========================================
    
    plt.figure(figsize=(8, 6))
    
    # 计算准确率
    def get_acc(df):
        # 兼容 "Correct" 字符串和 1.0 数字
        return (df['is_correct'].astype(str).str.contains('Correct', case=False) | 
                (df['is_correct'] == 1)).mean()

    acc_no = get_acc(df_no)
    acc_cot = get_acc(df_cot)
    
    data_acc = pd.DataFrame({
        'Mode': ['Without CoT', 'With CoT'],
        'Accuracy': [acc_no, acc_cot]
    })

    # 绘图
    bars = sns.barplot(x='Mode', y='Accuracy', data=data_acc, palette=['#4c72b0', '#dd8452'])
    plt.title("Model Accuracy Comparison (GSM8K)", fontsize=15, fontweight='bold')
    plt.ylabel("Accuracy (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.1) # Y轴固定为 0 到 110%

    # 在柱子顶端标数值
    for i, v in enumerate(data_acc['Accuracy']):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("accuracy_analysis.png", dpi=300)
    print(">> [成功] 准确率对比图已保存为: accuracy_analysis.png")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    main()