import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置绘图风格 (学术风)
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 1. 读取数据
file_path = 'output/house_5_processed.csv' # 确保路径对应
# 如果你的文件在根目录，改为 'house_1_processed.csv'
if not os.path.exists(file_path):
    file_path = 'house_5_processed.csv'

df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# 创建图片保存目录
if not os.path.exists('paper_images'):
    os.makedirs('paper_images')

print("正在生成论文图表...")


plt.figure(figsize=(10, 6))

# 按小时分组计算均值和置信区间
sns.lineplot(x='hour', y='power_clean', data=df,
             color='#2b5797', linewidth=2.5, errorbar='sd')

plt.title('Average Daily Power Consumption Profile (House 1)', fontsize=16, pad=20)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Power (Watts)', fontsize=12)
plt.xticks(range(0, 25, 2)) # 每2小时显示刻度
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('paper_images/House5_Fig1_Daily_Profile.png', dpi=300)
print(" 图 1 已生成: 典型日负荷曲线")

# 图 2: 特征相关性热力图 (Correlation Matrix)

plt.figure(figsize=(10, 8))

# 选取关键特征进行相关性分析
cols = ['target_1h', 'power_clean', 'power_lag_1', 'power_lag_24',
        'hour', 'power_std_6h', 'temp_mean_rolling']
# 如果数据里没有 temp_mean_rolling，代码会自动忽略
available_cols = [c for c in cols if c in df.columns]

corr = df[available_cols].corr()

# 绘制热力图
mask = np.triu(np.ones_like(corr, dtype=bool)) # 只显示下三角
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r',
            vmin=-1, vmax=1, square=True, linewidths=.5)

plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()

plt.savefig('paper_images/House5_Fig2_Correlation.png', dpi=300)
print(" 已生成: 特征相关性热力图")


# 图 3: 原始数据 vs 预处理数据 (Preprocessing Effect)
plt.figure(figsize=(12, 5))

# 随机截取一周的数据 (168小时)
subset = df.iloc[1000:1168]


plt.plot(subset.index, subset['power_clean'], label='Processed Data', color='#c0392b', linewidth=2)



plt.title('Power Consumption Time Series (7-Day Sample)', fontsize=16, pad=20)
plt.ylabel('Power (Watts)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('paper_images/House5_Fig4_TimeSeries.png', dpi=300)
print(" 已生成: 时间序列片段")