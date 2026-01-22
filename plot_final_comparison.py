import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import platform



def configure_plotting():
    # 使用 Seaborn 学术风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Mac 系统字体修正
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False


configure_plotting()


def generate_comparison_plot():
    # 1. 读取数据 (兼容不同路径)
    try:
        rf_df = pd.read_csv('results/fast_rf_summary.csv')
        lstm_df = pd.read_csv('results/lstm_summary.csv')
    except FileNotFoundError:
        try:
            rf_df = pd.read_csv('fast_rf_summary.csv')
            lstm_df = pd.read_csv('lstm_summary.csv')
        except:
            print(" 错误：未找到 CSV 文件，请确保它们在 results/ 文件夹或当前目录下。")
            return

    # 2. 数据预处理
    rf_df['Model_Short'] = 'Fast RF'
    lstm_df['Model_Short'] = 'LSTM'

    df = pd.concat([rf_df, lstm_df], ignore_index=True)

    # 确保 House 顺序 (4 -> 1 -> 5)
    df['House'] = df['House'].astype(str)
    house_order = ['1', '4', '5']
    df = df[df['House'].isin(house_order)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    sns.barplot(
        data=df,
        x='House',
        y='RMSE',
        hue='Model_Short',
        ax=axes[0],
        # 绿色 (RF), 蓝色 (LSTM)
        palette=['#2ecc71', '#3498db'],
        order=house_order,
        edgecolor='black', linewidth=1
    )

    axes[0].set_title('Prediction Accuracy (RMSE)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('RMSE (Watts) - Lower is Better', fontsize=12)
    axes[0].set_xlabel('House ID', fontsize=12)
    axes[0].legend(title='Model', loc='upper left')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 标注数值
    for p in axes[0].patches:
        if p.get_height() > 0:
            axes[0].annotate(f'{p.get_height():.1f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')


    # 定义特定的颜色：Purple (Fast RF), Red (LSTM)
    custom_palette = ['#9b59b6', '#e74c3c']

    sns.barplot(
        data=df,
        x='House',
        y='Train_Time_s',
        hue='Model_Short',
        ax=axes[1],
        palette=custom_palette,
        order=house_order,
        edgecolor='black', linewidth=1
    )

    axes[1].set_title('Training Efficiency (Time vs. Scale)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Training Time (Seconds) - Log Scale', fontsize=12)
    axes[1].set_xlabel('House ID', fontsize=12)
    axes[1].set_yscale('log')  # 保持对数坐标


    axes[1].legend(title='Model', loc='upper left')

    axes[1].grid(axis='y', linestyle='--', alpha=0.5, which='major')


    for p in axes[1].patches:
        if p.get_height() > 0:
            height = p.get_height()
            label = f'{height:.2f}s'
            axes[1].annotate(label,
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333',
                             xytext=(0, 3), textcoords='offset points')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_path = 'results/final_model_comparison_v2.png'
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"已生成紫红配色对比图: {save_path}")
    # plt.show()


if __name__ == "__main__":
    generate_comparison_plot()