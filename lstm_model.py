import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')



def configure_plotting_style():
    sns.set_style("whitegrid")
    sys_os = platform.system()
    if sys_os == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False


configure_plotting_style()



class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMModel, self).__init__()
        # 第一层 LSTM
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        # 第二层 LSTM
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        # 全连接层
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMTrainerTorch:
    def __init__(self, house_num):
        self.house_num = house_num
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_root, 'dataset_ready', f'house_{house_num}')
        self.model_dir = os.path.join(self.project_root, 'models')
        self.results_dir = os.path.join(self.project_root, 'results')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 自动检测设备 (Mac M1/M2 会用 mps，普通电脑用 cpu)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"[House {house_num}] 使用设备: Apple Metal (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[House {house_num}] 使用设备: NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print(f"[House {house_num}] 使用设备: CPU")

        self.target_scaler = None

    def load_data(self):
        print(f"正在加载数据")
        try:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'), index_col=0, parse_dates=True)
            val_df = pd.read_csv(os.path.join(self.data_dir, 'val.csv'), index_col=0, parse_dates=True)
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), index_col=0, parse_dates=True)
            self.target_scaler = joblib.load(os.path.join(self.data_dir, 'target_scaler.pkl'))

            def prepare_tensor(df):
                target_cols = [c for c in df.columns if c.startswith('target_')]
                feature_cols = [c for c in df.columns if c not in target_cols]
                X = df[feature_cols].values.astype(np.float32)
                y = df[target_cols].values.astype(np.float32)
                # Reshape X to [batch, 1, features]
                X = X.reshape((X.shape[0], 1, X.shape[1]))
                return torch.tensor(X), torch.tensor(y)

            self.X_train, self.y_train = prepare_tensor(train_df)
            self.X_val, self.y_val = prepare_tensor(val_df)
            self.X_test, self.y_test = prepare_tensor(test_df)

            # 特征维度
            self.input_dim = self.X_train.shape[2]
            self.output_dim = self.y_train.shape[1]
            return True

        except FileNotFoundError:
            print(f"未找到 House {self.house_num} 数据。")
            return False

    def train(self):
        if not self.load_data(): return

        # 数据加载器
        train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=32)

        # 初始化模型
        model = LSTMModel(self.input_dim, self.output_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("开始训练 (Max Epochs=50)...")
        start_time = time.time()

        best_val_loss = float('inf')
        patience = 5
        counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(50):
            # 训练步
            model.train()
            batch_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)
            train_losses.append(avg_train_loss)

            # 验证步
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_batch_losses.append(loss.item())

            avg_val_loss = np.mean(val_batch_losses)
            val_losses.append(avg_val_loss)

            print(f"   Epoch {epoch + 1}/50 | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), os.path.join(self.model_dir, f'lstm_house_{self.house_num}.pth'))
            else:
                counter += 1
                if counter >= patience:
                    print("早停触发 (Early Stopping)")
                    break

        train_time = time.time() - start_time
        print(f"训练完成! 耗时: {train_time:.2f} 秒")

        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(os.path.join(self.model_dir, f'lstm_house_{self.house_num}.pth')))
        self.evaluate(model, train_time, train_losses, val_losses)

    def evaluate(self, model, train_time, train_losses, val_losses):
        print(f"[House {self.house_num}] 正在评估")
        model.eval()
        start_pred = time.time()

        with torch.no_grad():
            X_test_dev = self.X_test.to(self.device)
            y_pred_scaled = model(X_test_dev).cpu().numpy()

        inference_time = (time.time() - start_pred) / len(self.X_test)

        # 反归一化
        y_test_real = self.target_scaler.inverse_transform(self.y_test.numpy())
        y_pred_real = self.target_scaler.inverse_transform(y_pred_scaled)

        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, y_pred_real)

        print(f"LSTM RMSE: {rmse:.2f} W")
        print(f"训练耗时: {train_time:.2f}s")

        self._save_results(rmse, mae, train_time, inference_time)
        self._plot_loss(train_losses, val_losses)
        self._plot_prediction(y_test_real, y_pred_real)

    def _save_results(self, rmse, mae, t_time, i_time):
        file_path = os.path.join(self.results_dir, 'lstm_summary.csv')
        metrics = {
            'House': self.house_num,
            'Model': 'LSTM_PyTorch',
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

    def _plot_loss(self, train_loss, val_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title(f'LSTM Training Loss (House {self.house_num})')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, f'lstm_loss_house_{self.house_num}.png'))
        plt.close()

    def _plot_prediction(self, y_true, y_pred):
        idx = np.random.randint(0, len(y_true))
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 25), y_true[idx], 'b-o', label='Ground Truth')
        plt.plot(range(1, 25), y_pred[idx], 'r--x', label='Prediction')
        plt.title(f'LSTM Forecast Sample (House {self.house_num})')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, f'lstm_pred_house_{self.house_num}.png'))
        plt.close()


if __name__ == "__main__":
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lstm_summary.csv')
    if os.path.exists(summary_path):
        os.remove(summary_path)

    print("LSTM Training")

    # 依次跑 House 4, 1, 5
    for h in [4, 1, 5]:
        LSTMTrainerTorch(h).train()