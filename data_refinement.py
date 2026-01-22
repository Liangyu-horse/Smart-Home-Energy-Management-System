import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


class DataRefiner:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.input_dir = os.path.join(self.project_root, 'output')
        self.output_dir = os.path.join(self.project_root, 'dataset_ready')

        # 如果不存在 dataset_ready 文件夹，自动创建
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_processed_data(self, house_num):
        """读取之前生成的 processed.csv"""
        file_path = os.path.join(self.input_dir, f'house_{house_num}_processed.csv')
        if not os.path.exists(file_path):
            print(f"跳过 House {house_num}: 未找到 {file_path}")
            return None

        print(f"\n 正在读取 House {house_num}...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def remove_interpolation_artifacts(self, df, house_num, threshold_hours=12):

        original_len = len(df)

        # 计算滚动方差（反映数据的波动性）
        # 窗口大小 = 小时数 * 60 (假设数据是1分钟粒度)
        window_size = threshold_hours * 60
        rolling_std = df['power_clean'].rolling(window=window_size).std()

        # 标记“死值”区域 (标准差接近0)
        # 也就是连续12小时数值几乎没变，这在真实家庭用电中是不可能的
        dead_zones = rolling_std < 0.01

        # 获取要删除的索引
        indices_to_drop = df[dead_zones].index

        if len(indices_to_drop) > 0:
            print(f"检测到插值伪影（直线）: 删除了 {len(indices_to_drop)} 行数据")
            df_clean = df.drop(indices_to_drop)

            # 删除后可能会产生新的断点，这里我们简单处理，直接拼接
            # 如果断点很多，这种拼接可能会导致时间跳跃，但比保留直线好
        else:
            print("数据质量良好，未发现长插值直线。")
            df_clean = df

        print(f"数据量变化: {original_len} -> {len(df_clean)}")
        return df_clean

    def normalize_and_split(self, df, house_num):
        """
        [关键步骤] 数据集切分与标准化
        注意：必须只在【训练集】上fit scaler，然后应用到验证集和测试集，防止数据泄露！
        """
        # 1. 确定特征列 (排除非数值列)
        # 我们主要缩放功率相关的数值，时间特征(hour等)通常不需要MinMax或者单独处理
        # 这里为了LSTM简单起见，对所有数值列进行MinMax

        # 目标列
        target_cols = [c for c in df.columns if c.startswith('target_')]
        # 特征列 (排除目标列)
        feature_cols = [c for c in df.columns if c not in target_cols]

        # 2. 划分训练/验证/测试 (7:2:1)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"划分完成: 训练集({len(train_df)}) | 验证集({len(val_df)}) | 测试集({len(test_df)})")

        # 3. 标准化 (MinMaxScaler)
        print("执行标准化 (Fit on Train, Transform on All)...")

        # 定义特征缩放器和标签缩放器 (分开定义，方便反归一化)
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        # 拟合 (只用训练集!)
        feature_scaler.fit(train_df[feature_cols])
        target_scaler.fit(train_df[target_cols])

        # 转换并重组 DataFrame
        def transform_data(subset_df):
            X_scaled = feature_scaler.transform(subset_df[feature_cols])
            y_scaled = target_scaler.transform(subset_df[target_cols])

            # 合并回一个DataFrame (保持列名)
            X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=subset_df.index)
            y_df = pd.DataFrame(y_scaled, columns=target_cols, index=subset_df.index)
            return pd.concat([X_df, y_df], axis=1)

        train_scaled = transform_data(train_df)
        val_scaled = transform_data(val_df)
        test_scaled = transform_data(test_df)

        # 4. 保存文件
        self._save_datasets(house_num, train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler)

    def _save_datasets(self, house_num, train, val, test, f_scaler, t_scaler):
        """保存处理好的 numpy/csv 文件和 scaler"""
        save_path = os.path.join(self.output_dir, f'house_{house_num}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存 CSV
        train.to_csv(os.path.join(save_path, 'train.csv'))
        val.to_csv(os.path.join(save_path, 'val.csv'))
        test.to_csv(os.path.join(save_path, 'test.csv'))

        # 保存 Scaler (后续反归一化预测结果时必须用!)
        joblib.dump(f_scaler, os.path.join(save_path, 'feature_scaler.pkl'))
        joblib.dump(t_scaler, os.path.join(save_path, 'target_scaler.pkl'))

        print(f"已保存至: {save_path}/ (train.csv, scalers...)")

    def run_stage1(self):
        print("[Step 1] 开始执行数据精细化清洗与标准化...")

        # 遍历所有可能的家庭 (1-5)
        for i in range(1, 6):
            df = self.load_processed_data(i)
            if df is not None:
                # 步骤 A: 去除长直线 (针对 H2, H3, H4 特别有效)
                # House 1 和 5 质量较好，通常不会触发删除，但也检查一下无妨
                df_clean = self.remove_interpolation_artifacts(df, i)

                # 步骤 B: 归一化并切分
                self.normalize_and_split(df_clean, i)

        print("\n第一阶段处理全部完成！准备好进入模型训练。")


if __name__ == "__main__":
    refiner = DataRefiner()
    refiner.run_stage1()