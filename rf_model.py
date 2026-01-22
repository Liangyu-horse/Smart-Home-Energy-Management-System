import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac 中文支持
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class RandomForestTrainer:
    def __init__(self, house_num):
        self.house_num = house_num
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        # 路径配置
        self.data_dir = os.path.join(self.project_root, 'dataset_ready', f'house_{house_num}')
        self.model_dir = os.path.join(self.project_root, 'models')
        self.results_dir = os.path.join(self.project_root, 'results')

        # 自动创建文件夹
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 占位符
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.target_scaler = None
        self.model = None

    def load_data(self):
        """加载 Stage 1 准备好的标准化数据"""
        print(f"\n [House {self.house_num}] 正在加载数据...")

        try:
            # 读取 CSV
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'), index_col=0, parse_dates=True)
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), index_col=0, parse_dates=True)

            # 加载目标缩放器 (用于反归一化)
            scaler_path = os.path.join(self.data_dir, 'target_scaler.pkl')
            self.target_scaler = joblib.load(scaler_path)

            # 分离特征和目标
            # 目标列以 'target_' 开头
            target_cols = [c for c in train_df.columns if c.startswith('target_')]
            feature_cols = [c for c in train_df.columns if c not in target_cols]

            self.X_train = train_df[feature_cols]
            self.y_train = train_df[target_cols]
            self.X_test = test_df[feature_cols]
            self.y_test = test_df[target_cols]

            print(f"   - 训练集样本: {len(self.X_train)}, 特征数: {len(feature_cols)}")
            print(f"   - 测试集样本: {len(self.X_test)}")
            return True

        except FileNotFoundError:
            print(f" 错误: 找不到 House {self.house_num} 的数据。请先运行 stage1_data_refinement.py")
            return False

    def train_optimized(self, use_grid_search=False):

        print(f" [House {self.house_num}] 开始训练随机森林...")
        start_time = time.time()

        base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        if use_grid_search:
            print(" 正在进行超参数调优 (RandomizedSearch)")
            # 定义搜索空间
            param_dist = {
                'estimator__n_estimators': [100, 200],  # 树的数量
                'estimator__max_depth': [10, 20, None],  # 树深度
                'estimator__min_samples_split': [2, 5]  # 分裂最小样本
            }

            # 使用多输出包装器
            mor = MultiOutputRegressor(base_rf)

            # 时间序列交叉验证 (防止未来数据泄露)
            tscv = TimeSeriesSplit(n_splits=3)

            search = RandomizedSearchCV(
                estimator=mor,
                param_distributions=param_dist,
                n_iter=5,  # 尝试 5 种组合 (可调大)
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            search.fit(self.X_train, self.y_train)
            self.model = search.best_estimator_
            print(f"   最佳参数: {search.best_params_}")

        else:
            # 使用经验参数快速训练
            print("   使用默认参数快速训练...")
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, n_jobs=-1, random_state=42)
            )
            self.model.fit(self.X_train, self.y_train)

        train_time = time.time() - start_time
        print(f" 训练耗时: {train_time:.2f} 秒")

        # 保存模型
        model_path = os.path.join(self.model_dir, f'rf_house_{self.house_num}.pkl')
        joblib.dump(self.model, model_path)

        return train_time

    def evaluate(self, train_time):
        """评估模型并生成可视化"""
        print(f"[House {self.house_num}] 正在评估...")

        # 1. 预测 (输出是 0-1 之间的值)
        start_pred = time.time()
        y_pred_scaled = self.model.predict(self.X_test)
        inference_time = (time.time() - start_pred) / len(self.X_test)

        # 2. 反归一化 (还原为瓦特 W)
        # 必须用之前的 target_scaler 还原，才能算出真实的误差
        y_test_real = self.target_scaler.inverse_transform(self.y_test)
        y_pred_real = self.target_scaler.inverse_transform(y_pred_scaled)

        # 3. 计算指标
        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, y_pred_real)


        print(f"   RMSE (均方根误差): {rmse:.2f} W")
        print(f"   MAE  (平均绝对误差): {mae:.2f} W")
        print(f"   推理时间: {inference_time * 1000:.4f} ms/样本")


        # 4. 保存指标到 CSV
        metrics = {
            'House': self.house_num,
            'Model': 'RandomForest',
            'RMSE': rmse,
            'MAE': mae,
            'Train_Time_s': train_time,
            'Inference_Time_ms': inference_time * 1000
        }
        self._save_metrics(metrics)

        # 5. 生成对比图
        self._plot_results(y_test_real, y_pred_real)

    def _save_metrics(self, metrics):
        file_path = os.path.join(self.results_dir, 'model_performance_summary.csv')
        df = pd.DataFrame([metrics])

        # 如果文件存在，追加；否则新建
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, mode='w', header=True, index=False)

    def _plot_results(self, y_true, y_pred):

        # 随机选一个时刻
        idx = np.random.randint(0, len(y_true))

        plt.figure(figsize=(10, 6))
        hours = range(1, 25)

        plt.plot(hours, y_true[idx], 'b-o', label='真实值 (Ground Truth)', linewidth=2)
        plt.plot(hours, y_pred[idx], 'r--x', label='预测值 (Prediction)', linewidth=2)

        plt.title(f'Random Forest: 24h Forecast (House {self.house_num}) - Sample #{idx}', fontsize=14)
        plt.xlabel('Future Hours (+h)', fontsize=12)
        plt.ylabel('Power (Watts)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.results_dir, f'rf_pred_house_{self.house_num}.png')
        plt.savefig(save_path, dpi=300)
        print(f"   预测图已保存: {save_path}")
        plt.close()


if __name__ == "__main__":
    # 清空旧的汇总文件
    summary_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'model_performance_summary.csv')
    if os.path.exists(summary_file):
        os.remove(summary_file)

    # 批量运行实验
    # House 1: 基准实验 (数据最全)
    # House 4: 小样本实验 (数据最少，测试 RF 优势)
    # House 5: 泛化实验 (高能耗)

    target_houses = [1, 4, 5]


    print("随机森林模型开发与评估")


    for h_num in target_houses:
        trainer = RandomForestTrainer(house_num=h_num)
        if trainer.load_data():
            # 开启 use_grid_search=True 可以进行超参数搜索（会慢一些）
            # 建议 House 1 开启 True，其他 False 快速验证
            do_search = True if h_num == 1 else False

            t_time = trainer.train_optimized(use_grid_search=do_search)
            trainer.evaluate(t_time)

    print("\n 查看 results/ 文件夹下的 CSV 和图片。")