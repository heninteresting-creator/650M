import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_cot = pd.read_csv("results_with_cot.csv")
df_no_cot = pd.read_csv("results_without_cot.csv")

# Filter out rows with errors (Using .isna() to handle CSV null values)
df_cot_clean = df_cot[df_cot['error'].isna()].copy()
df_no_cot_clean = df_no_cot[df_no_cot['error'].isna()].copy()

print(f"CoT Valid Samples: {len(df_cot_clean)}/{len(df_cot)}")
print(f"Non-CoT Valid Samples: {len(df_no_cot_clean)}/{len(df_no_cot)}")

# ========== Fig 1: Token Usage Comparison ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Calculate global maximum for Y-axis scaling
max_tokens = max(df_cot_clean['tokens'].max(), df_no_cot_clean['tokens'].max())
y_limit = max_tokens * 1.1

# Subplot 1: CoT Token Usage
avg_cot = df_cot_clean['tokens'].mean()
x_cot = range(len(df_cot_clean))
ax1.scatter(x_cot, df_cot_clean['tokens'], alpha=0.6, s=20, label='Per Question')
ax1.axhline(y=avg_cot, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_cot:.1f}')
ax1.set_title("CoT Mode - Token Usage per Question", fontsize=14, fontweight='bold')
ax1.set_ylabel("Token Count")
ax1.set_ylim(0, y_limit)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Non-CoT Token Usage
avg_no_cot = df_no_cot_clean['tokens'].mean()
x_no_cot = range(len(df_no_cot_clean))
ax2.scatter(x_no_cot, df_no_cot_clean['tokens'], alpha=0.6, s=20, color='orange', label='Per Question')
ax2.axhline(y=avg_no_cot, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_no_cot:.1f}')
ax2.set_title("Non-CoT Mode - Token Usage per Question", fontsize=14, fontweight='bold')
ax2.set_xlabel("Question Index")
ax2.set_ylabel("Token Count")
ax2.set_ylim(0, y_limit)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("token_usage.png", dpi=300)
print("Saved: token_usage.png")

# ========== Fig 2: Accuracy Comparison ==========
plt.figure(figsize=(8, 6))

# Calculate Accuracy
def calc_accuracy(df):
    return (df['is_correct'] == 'Correct').mean()

acc_cot = calc_accuracy(df_cot_clean)
acc_no_cot = calc_accuracy(df_no_cot_clean)

# Plot Bar Chart
modes = ['Non-CoT', 'CoT']
accuracies = [acc_no_cot, acc_cot]
colors = ['#4c72b0', '#dd8452']

bars = plt.bar(modes, accuracies, color=colors, alpha=0.8, width=0.6)
plt.title("Accuracy Comparison: CoT vs Non-CoT", fontsize=15, fontweight='bold')
plt.ylabel("Accuracy Score", fontsize=12)
plt.ylim(0, 1.1)

# Add value labels on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
print("Saved: accuracy_comparison.png")

# Display Statistics in Terminal
print("\n" + "="*50)
print("STATISTICS SUMMARY:")
print("="*50)
print(f"CoT Mode:")
print(f"  - Accuracy: {acc_cot:.2%}")
print(f"  - Avg Tokens: {avg_cot:.1f}")
print(f"  - Total Tokens: {df_cot_clean['tokens'].sum():.0f}")
print(f"\nNon-CoT Mode:")
print(f"  - Accuracy: {acc_no_cot:.2%}")
print(f"  - Avg Tokens: {avg_no_cot:.1f}")
print(f"  - Total Tokens: {df_no_cot_clean['tokens'].sum():.0f}")
print("="*50)

plt.show()