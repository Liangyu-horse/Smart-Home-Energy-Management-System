import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


def configure_plotting_style():
    # 使用 Seaborn 的默认学术风格
    sns.set_style("whitegrid")
    # 设置默认字体为通用的 Sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    # 确保负号显示正常
    plt.rcParams['axes.unicode_minus'] = False


# 执行配置
configure_plotting_style()



class RandomForestFastTrainer:
    def __init__(self, house_num):
        self.house_num = house_num
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        # 路径配置
        self.data_dir = os.path.join(self.project_root, 'dataset_ready', f'house_{house_num}')
        self.model_dir = os.path.join(self.project_root, 'models')
        self.results_dir = os.path.join(self.project_root, 'results')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.target_scaler = None
        self.model = None

    def load_data(self):
        """Load normalized data"""
        print(f"\n[House {self.house_num}] Loading Data...")
        try:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'), index_col=0, parse_dates=True)
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), index_col=0, parse_dates=True)
            self.target_scaler = joblib.load(os.path.join(self.data_dir, 'target_scaler.pkl'))

            target_cols = [c for c in train_df.columns if c.startswith('target_')]
            feature_cols = [c for c in train_df.columns if c not in target_cols]

            self.X_train = train_df[feature_cols]
            self.y_train = train_df[target_cols]
            self.X_test = test_df[feature_cols]
            self.y_test = test_df[target_cols]

            print(f"   - Training Samples: {len(self.X_train)}")
            return True
        except FileNotFoundError:
            print(f" Error: Data for House {self.house_num} not found. Please run stage1 first.")
            return False

    def train_fast(self):
        """Execute Fast Training Mode"""
        print(f"️ [House {self.house_num}] Starting Fast Training...")
        start_time = time.time()


        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_split=10,
            max_samples=0.3,  # Subsampling for speed
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        self.model.fit(self.X_train, self.y_train)

        train_time = time.time() - start_time
        print(f" Training Complete! Time: {train_time:.2f} s")

        joblib.dump(self.model, os.path.join(self.model_dir, f'rf_fast_house_{self.house_num}.pkl'))
        return train_time

    def evaluate(self, train_time):
        """Evaluate and Save Results"""
        print(f" [House {self.house_num}] Evaluating...")
        start_pred = time.time()

        y_pred_scaled = self.model.predict(self.X_test)
        inference_time = (time.time() - start_pred) / len(self.X_test)

        # Inverse Transform
        y_test_real = self.target_scaler.inverse_transform(self.y_test)
        y_pred_real = self.target_scaler.inverse_transform(y_pred_scaled)

        # Metrics
        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, y_pred_real)


        print(f"   RMSE: {rmse:.2f} W")
        print(f"   MAE : {mae:.2f} W")
        print(f"   Train Time: {train_time:.2f}s")


        self._save_results(rmse, mae, train_time, inference_time)
        self._plot_sample(y_test_real, y_pred_real)

    def _save_results(self, rmse, mae, t_time, i_time):
        """Save metrics to CSV"""
        file_path = os.path.join(self.results_dir, 'fast_rf_summary.csv')

        metrics = {
            'House': self.house_num,
            'Model': 'RandomForest_Fast',
            'RMSE': rmse,
            'MAE': mae,
            'Train_Time_s': round(t_time, 4),
            'Inference_Time_ms': round(i_time * 1000, 4)
        }

        df = pd.DataFrame([metrics])

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, mode='w', header=True, index=False)

        print(f"   💾 Saved to: {file_path}")

    def _plot_sample(self, y_true, y_pred):
        """Plotting function with English Labels"""
        idx = np.random.randint(0, len(y_true))

        plt.figure(figsize=(10, 6))

        # 绘图内容
        plt.plot(range(1, 25), y_true[idx], 'b-o', label='Ground Truth', linewidth=2)
        plt.plot(range(1, 25), y_pred[idx], 'r--x', label='Prediction', linewidth=2)


        plt.title(f'Fast Random Forest Forecast (House {self.house_num})', fontsize=14)
        plt.xlabel('Future Time Steps (Hours)', fontsize=12)
        plt.ylabel('Power Consumption (Watts)', fontsize=12)
        plt.legend(fontsize=12, loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, f'fast_rf_house_{self.house_num}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    # Remove old summary to avoid duplicates if running fresh
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fast_rf_summary.csv')
    if os.path.exists(summary_path):
        os.remove(summary_path)

    houses_to_run = [4, 1, 5]

    print("   Stage 2 Fast: Random Forest Training")


    for h in houses_to_run:
        trainer = RandomForestFastTrainer(house_num=h)
        if trainer.load_data():
            t_time = trainer.train_fast()
            trainer.evaluate(t_time)

    print("\nAll tasks completed! Check results/fast_rf_summary.csv")