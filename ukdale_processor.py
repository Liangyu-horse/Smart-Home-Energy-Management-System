import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class UKDALEProcessor:
    def __init__(self, data_path=None):

        if data_path is None:

            self.data_path = self._get_project_data_path()
        else:
            self.data_path = data_path

        self.data = {}
        self.available_households = []
        self.processed_households = []

        self.output_dir = self._create_output_dir()

        self._detect_available_households()

    def _get_project_data_path(self):

        project_root = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(project_root, 'data', 'ukdale')

        return data_path

    def _create_output_dir(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _detect_available_households(self):
        #检测可用的家庭数据
        print("正在检测家庭数据")

        if not os.path.exists(self.data_path):
            print(f"数据路径不存在: {self.data_path}")
            return

        #获取数据目录中的所有内容
        all_items = os.listdir(self.data_path)
        print(f"数据目录内容: {all_items}")

        #家庭文件夹模式
        house_patterns = [
            'house_1', 'house_2', 'house_3', 'house_4', 'house_5',
            'house1', 'house2', 'house3', 'house4', 'house5'
        ]

        #检测可用的家庭
        for house_pattern in house_patterns:
            house_path = os.path.join(self.data_path, house_pattern)
            if os.path.exists(house_path):
                #检查是否有数据文件
                data_files = self._find_data_files(house_path)
                if data_files:
                    #提取家庭编号
                    house_num = self._extract_house_number(house_pattern)
                    if house_num and house_num not in self.available_households:
                        self.available_households.append(house_num)
                        print(f"发现家庭 {house_num} 数据: {len(data_files)} 个文件")

        #如果没有找到家庭文件夹，检查是否有直接的数据文件
        if not self.available_households:
            data_files = self._find_data_files(self.data_path)
            if data_files:
                # 尝试从文件名中提取家庭编号
                for file in data_files:
                    house_num = self._extract_house_number_from_file(file)
                    if house_num and house_num not in self.available_households:
                        self.available_households.append(house_num)

        #排序家庭编号
        self.available_households.sort()

        print(f"检测到 {len(self.available_households)} 个可用的家庭: {self.available_households}")

    def _find_data_files(self, directory):
        #在目录中查找数据文件
        data_files = []
        if os.path.isdir(directory):
            for item in os.listdir(directory):
                if item.endswith('.dat') or item.endswith('.csv'):
                    data_files.append(os.path.join(directory, item))
        return data_files

    def _extract_house_number(self, pattern):
        #从文件夹模式中提取家庭编号
        import re
        match = re.search(r'house[_\s]*(\d+)', pattern, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_house_number_from_file(self, filename):
        #从文件名中提取家庭编号
        import re
        match = re.search(r'house[_\s]*(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def process_all_households(self, sample_freq='1H'):
        print(f"\n处理所有 {len(self.available_households)} 个家庭的数据")

        if not self.available_households:
            print("没有找到可用的家庭数据")
            return
        results = {}

        for house_num in self.available_households:
            print(f"\n{'=' * 50}")
            print(f"处理家庭 {house_num}")
            print(f"{'=' * 50}")

            try:
                df_raw = self.load_and_explore(house_num=house_num, sample_freq=sample_freq)

                if df_raw is not None:

                    df_processed = self.preprocess_data(house_num=house_num, method='linear', window_size=6)

                    df_features = self.feature_engineering(house_num=house_num, lookback_hours=24)

                    saved_file = self.save_processed_data(house_num=house_num)

                    if saved_file:
                        self.processed_households.append(house_num)
                        results[house_num] = {
                            'raw_data': df_raw,
                            'processed_data': df_processed,
                            'features_data': df_features,
                            'saved_file': saved_file
                        }

                        print(f"家庭 {house_num} 处理完成")
                    else:
                        print(f"家庭 {house_num} 保存失败")
                else:
                    print(f"家庭 {house_num} 数据加载失败")

            except Exception as e:
                print(f"处理家庭 {house_num} 时出错: {e}")
                import traceback
                traceback.print_exc()
        self._generate_summary_report(results)

        return results

    def load_and_explore(self, house_num=3, sample_freq='1H'):
        #数据集获取和初步探索
        print(f"加载家庭 {house_num} 的数据")
        print(f"数据路径: {self.data_path}")

        possible_paths = [
            os.path.join(self.data_path, f'house_{house_num}', 'channel_1.dat'),
            os.path.join(self.data_path, f'house_{house_num}', 'mains.dat'),
             ]

        main_power_file = None
        for path in possible_paths:
            if os.path.exists(path):
                main_power_file = path
                print(f"找到数据文件: {path}")
                break

        if main_power_file is None:
            print(f"无法找到家庭 {house_num} 的数据文件")
            return None

        try:
            #读取数据
            if main_power_file.endswith('.dat'):

                print("读取.dat格式文件")
                df = pd.read_csv(main_power_file, delim_whitespace=True, header=None,
                                 names=['timestamp', 'power'])
            else:

                print("读取CSV格式文件...")
                df = pd.read_csv(main_power_file)

            print(f"原始数据形状: {df.shape}")

            #转换时间戳（UK-DALE使用Unix时间戳）
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)

            #移除原始时间戳列
            if 'timestamp' in df.columns:
                df.drop('timestamp', axis=1, inplace=True)

            #数据探索
            self._explore_data(df, house_num)

            #重采样到指定频率
            print(f"重采样到 {sample_freq} 频率...")
            df_resampled = df.resample(sample_freq).mean()

            self.data[house_num] = df_resampled
            return df_resampled

        except Exception as e:
            print(f"加载家庭 {house_num} 数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _explore_data(self, df, house_num):
        print(f"\n--- 家庭 {house_num} 数据探索结果 ---")
        print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"总数据点数: {len(df)}")
        print(f"采样频率: 约 {self._estimate_frequency(df)}")

        #统计信息
        power_column = df.columns[0]  # 假设第一列是功率数据
        print(f"\n功率统计信息:")
        print(df[power_column].describe())

        #数据质量检查
        self._check_data_quality(df, house_num, power_column)

        #可视化原始数据
        self._visualize_raw_data(df, house_num, power_column)

    def _estimate_frequency(self, df):
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                median_freq = time_diffs.median()
                return str(median_freq)
        return "未知"

    def _check_data_quality(self, df, house_num, power_column):
        print(f"\n--- 数据质量检查 ---")

        #缺失值检查
        missing_count = df[power_column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        print(f"缺失值数量: {missing_count} ({missing_percentage:.2f}%)")

        #零值和异常值检查
        zero_count = (df[power_column] == 0).sum()
        zero_percentage = (zero_count / len(df)) * 100
        print(f"零值数量: {zero_count} ({zero_percentage:.2f}%)")

        #异常值检测（使用IQR方法）
        Q1 = df[power_column].quantile(0.25)
        Q3 = df[power_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[power_column] < lower_bound) | (df[power_column] > upper_bound)]
        print(f"异常值数量: {len(outliers)} ({(len(outliers) / len(df)) * 100:.2f}%)")

        #噪声分析
        self._analyze_noise(df, house_num, power_column)

    def _analyze_noise(self, df, house_num, power_column):
        power_diff = df[power_column].diff().abs()
        if len(power_diff.dropna()) > 0:
            noise_threshold = power_diff.quantile(0.95)
            potential_noise = power_diff[power_diff > noise_threshold]
            print(f"检测到的潜在噪声点: {len(potential_noise)}")
        else:
            noise_threshold = 0
            potential_noise = pd.Series([])
            print("无法计算噪声阈值")

        #可视化噪声分析
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        df[power_column].hist(bins=50, alpha=0.7)
        plt.title(f'House{house_num} - Power Distribution')
        plt.xlabel('Power (W)')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 2)
        if len(power_diff.dropna()) > 0:
            power_diff.hist(bins=50, alpha=0.7, color='orange')
            plt.axvline(noise_threshold, color='red', linestyle='--', label='Noise Threshold')
        plt.title('Power Change Distribution')
        plt.xlabel('Power Change')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 3)
        #随机选取一天的数据进行详细查看
        if len(df) > 24:
            sample_size = min(24, len(df) - 100)
            sample_day = df.iloc[100:100 + sample_size]
            plt.plot(sample_day.index, sample_day[power_column], marker='o', markersize=3)
            plt.title('Daily Power Pattern Example')
            plt.xlabel('Time')
            plt.ylabel('Power (W)')
            plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        # 小时模式分析
        df['hour'] = df.index.hour
        hourly_pattern = df.groupby('hour')[power_column].mean()
        plt.plot(hourly_pattern.index, hourly_pattern.values)
        plt.title('Average Hourly Power Pattern')
        plt.xlabel('Hour')
        plt.ylabel('Average Power (W)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'house_{house_num}_data_exploration.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  #关闭图形以避免显示问题

    def _visualize_raw_data(self, df, house_num, power_column):

        plt.figure(figsize=(15, 10))

        #整体趋势
        plt.subplot(2, 1, 1)
        if len(df) > 1000:
            df_resampled = df.resample('1D').mean()
            plt.plot(df_resampled.index, df_resampled[power_column], linewidth=1)
        else:
            plt.plot(df.index, df[power_column], linewidth=1)
        plt.title(f'House {house_num} - Total Power Trend')
        plt.ylabel('Power (W)')
        plt.grid(True)

        #最近一周的详细数据
        plt.subplot(2, 1, 2)
        if len(df) > 168:  # 一周的小时数
            last_week = df.iloc[-168:]
            plt.plot(last_week.index, last_week[power_column], linewidth=1)
            plt.title('Last Week Power Details')
            plt.ylabel('Power (W)')
            plt.xlabel('Time')
            plt.grid(True)
        else:
            #如果数据不足一周，显示所有数据
            plt.plot(df.index, df[power_column], linewidth=1)
            plt.title('Power Details')
            plt.ylabel('Power (W)')
            plt.xlabel('Time')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'house_{house_num}_power_trend.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def preprocess_data(self, house_num=3, method='linear', window_size=5):
        print(f"\n=== 对家庭 {house_num} 进行数据预处理 ===")

        if house_num not in self.data:
            print(f"错误: 家庭 {house_num} 的数据尚未加载")
            return None

        df = self.data[house_num].copy()
        power_column = df.columns[0]

        #缺失值处理
        original_length = len(df)
        missing_before = df[power_column].isnull().sum()

        if method == 'linear':
            df[power_column] = df[power_column].interpolate(method='linear')
            print(f"使用线性插值填充缺失值")
        elif method == 'rolling_mean':
            df[power_column] = df[power_column].fillna(
                df[power_column].rolling(window=window_size, min_periods=1).mean()
            )
            print(f"使用滑动平均填充缺失值 (窗口大小: {window_size})")

        missing_after = df[power_column].isnull().sum()
        print(f"缺失值处理: {missing_before} -> {missing_after}")

        #如果还有缺失值，使用前后向填充
        if missing_after > 0:
            df[power_column] = df[power_column].fillna(method='bfill').fillna(method='ffill')
            print(f"使用前后向填充处理剩余缺失值")

        #噪声处理 - 使用滑动平均平滑
        df['power_smooth'] = df[power_column].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        #异常值处理
        Q1 = df['power_smooth'].quantile(0.25)
        Q3 = df['power_smooth'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #将异常值限制在边界内
        df['power_clean'] = df['power_smooth'].clip(lower=lower_bound, upper=upper_bound)

        print(f"异常值处理: 将数值限制在 [{lower_bound:.2f}, {upper_bound:.2f}] 范围内")

        #可视化预处理效果
        self._visualize_preprocessing(df, house_num, power_column)

        self.data[house_num] = df
        return df

    def _visualize_preprocessing(self, df, house_num, original_power_column):
        plt.figure(figsize=(15, 12))

        #选取一个有代表性的时间段进行展示
        display_samples = min(100, len(df))
        sample_data = df.iloc[:display_samples]

        #原始数据 vs 平滑数据
        plt.subplot(3, 1, 1)
        plt.plot(sample_data.index, sample_data[original_power_column],
                 alpha=0.7, label='Original Data', color='blue')
        plt.plot(sample_data.index, sample_data['power_smooth'],
                 label='Smoothed Data', color='red', linewidth=2)
        plt.title(f'House {house_num} - Data Smoothing Effect')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)

        #异常值处理效果
        plt.subplot(3, 1, 2)
        plt.plot(sample_data.index, sample_data['power_smooth'],
                 alpha=0.7, label='Smoothed Data', color='red')
        plt.plot(sample_data.index, sample_data['power_clean'],
                 label='After Outlier Treatment', color='green', linewidth=2)
        plt.title('Outlier Treatment Effect')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)

        #数据分布对比
        plt.subplot(3, 1, 3)
        plt.hist(df[original_power_column].dropna(), bins=50, alpha=0.7,
                 label='Original Data', color='blue')
        plt.hist(df['power_clean'].dropna(), bins=50, alpha=0.7,
                 label='Processed Data', color='green')
        plt.title('Data Distribution Comparison')
        plt.xlabel('Power (W)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'house_{house_num}_preprocessing.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以避免显示问题

    def feature_engineering(self, house_num=3, lookback_hours=24):
        print(f"\n为家庭 {house_num} 构建特征工程")

        if house_num not in self.data:
            print(f"错误: 家庭 {house_num} 的数据尚未加载")
            return None

        df = self.data[house_num].copy()

        #使用清洗后的数据
        power_series = df['power_clean']

        #时间特征
        print("构建时间特征")
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_holiday'] = self._identify_holidays(df.index)

        #用电趋势特征
        print("构建用电趋势特征")
        windows = [3, 6, 12, 24]  # 3小时, 6小时, 12小时, 24小时

        for window in windows:
            df[f'power_mean_{window}h'] = power_series.rolling(window=window, min_periods=1).mean()
            df[f'power_std_{window}h'] = power_series.rolling(window=window, min_periods=1).std()
            df[f'power_max_{window}h'] = power_series.rolling(window=window, min_periods=1).max()
            df[f'power_min_{window}h'] = power_series.rolling(window=window, min_periods=1).min()
            df[f'power_trend_{window}h'] = power_series.diff(window)

        #滞后特征
        print("构建滞后特征")
        for lag in range(1, min(lookback_hours + 1, len(power_series))):
            df[f'power_lag_{lag}'] = power_series.shift(lag)

        #周期特征
        print("构建周期特征...")
        df['power_lag_24h'] = power_series.shift(24)
        df['power_lag_168h'] = power_series.shift(168)

        #统计特征
        print("构建统计特征")
        df['power_rolling_24h_mean'] = power_series.rolling(window=24, min_periods=1).mean()
        df['power_rolling_24h_std'] = power_series.rolling(window=24, min_periods=1).std()

        #变化率特征
        df['power_change_rate_1h'] = power_series.pct_change(periods=1)
        df['power_change_rate_24h'] = power_series.pct_change(periods=24)

        #时间周期特征（正弦余弦编码）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        #功率等级特征
        df['power_category'] = pd.cut(df['power_clean'],
                                      bins=[-1, 100, 500, 1000, float('inf')],
                                      labels=['low', 'medium', 'high', 'very_high'])

        #创建虚拟变量
        power_dummies = pd.get_dummies(df['power_category'], prefix='power_cat')
        df = pd.concat([df, power_dummies], axis=1)
        df.drop('power_category', axis=1, inplace=True)

        #目标变量 - 未来24小时用电量
        print("构建目标变量")
        for horizon in range(1, 25):
            df[f'target_{horizon}h'] = power_series.shift(-horizon)

        #移除因滞后和未来目标变量产生的NaN值
        initial_length = len(df)
        df = df.dropna()
        final_length = len(df)

        print(f"特征工程完成: 从 {initial_length} 个样本到 {final_length} 个样本")
        print(f"特征数量: {len(df.columns)}")

        #特征相关性分析
        self._analyze_feature_correlation(df, house_num)

        self.data[house_num] = df
        return df

    def _identify_holidays(self, dates):
        holidays = np.zeros(len(dates))
        for i, date in enumerate(dates):
            if date.month == 12 and date.day == 25:
                holidays[i] = 1
        return holidays

    def _analyze_feature_correlation(self, df, house_num):
        numeric_features = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_features].corr()

        #只显示与目标变量相关性较高的特征
        target_columns = [col for col in df.columns if col.startswith('target_')]
        if target_columns:
            target_corr = correlation_matrix[target_columns].abs().mean(axis=1).sort_values(ascending=False)

            plt.figure(figsize=(12, 8))
            top_features = target_corr.head(20)
            top_features.plot(kind='barh')
            plt.title(f'House {house_num} - Feature-Target Correlation')
            plt.xlabel('Average Absolute Correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'house_{house_num}_feature_correlation.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print("\nTop 10 features correlated with targets:")
            for feature, corr in target_corr.head(10).items():
                print(f"  {feature}: {corr:.4f}")

    def save_processed_data(self, house_num=3):

        if house_num in self.data:
            filename = f'house_{house_num}_processed.csv'
            full_path = os.path.join(self.output_dir, filename)
            self.data[house_num].to_csv(full_path)
            print(f"处理后的数据已保存到: {full_path}")
            return full_path
        else:
            print(f"错误: 家庭 {house_num} 的数据不存在")
            return None

    def _generate_summary_report(self, results):
        print(f"\n{'=' * 60}")
        print(f"UK-DALE 数据分析汇总报告")
        print(f"{'=' * 60}")

        print(f"数据路径: {self.data_path}")
        print(f"输出路径: {self.output_dir}")
        print(f"检测到的家庭: {self.available_households}")
        print(f"成功处理的家庭: {self.processed_households}")
        print(f"处理失败的家庭: {set(self.available_households) - set(self.processed_households)}")

        if results:
            print(f"\n详细结果:")
            for house_num, result in results.items():
                df_features = result['features_data']
                print(f"家庭 {house_num}:")
                print(f"数据形状: {df_features.shape}")
                print(f"特征数量: {len(df_features.columns)}")
                print(f"时间范围: {df_features.index.min()} 到 {df_features.index.max()}")
                print(f"保存文件: {result['saved_file']}")

        #生成汇总图表
        self._create_summary_visualization(results)

    def _create_summary_visualization(self, results):
        if not results:
            return

        plt.figure(figsize=(15, 10))

        #比较不同家庭的用电模式
        for i, (house_num, result) in enumerate(results.items()):
            df = result['features_data']

            #计算小时平均用电量
            hourly_avg = df.groupby('hour')['power_clean'].mean()

            plt.subplot(2, 2, 1)
            plt.plot(hourly_avg.index, hourly_avg.values, label=f'House {house_num}', linewidth=2)

            plt.subplot(2, 2, 2)
            df['power_clean'].hist(alpha=0.7, label=f'House {house_num}', bins=30)

        plt.subplot(2, 2, 1)
        plt.title('Average Hourly Power Consumption by Household')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power (W)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.title('Power Distribution by Household')
        plt.xlabel('Power (W)')
        plt.ylabel('Frequency')
        plt.legend()

        #处理状态饼图
        plt.subplot(2, 2, 3)
        success_count = len(self.processed_households)
        fail_count = len(self.available_households) - success_count
        sizes = [success_count, fail_count]
        labels = [f'成功 ({success_count})', f'失败 ({fail_count})']
        colors = ['#2ecc71', '#e74c3c']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('处理状态分布')

        #数据量比较
        plt.subplot(2, 2, 4)
        house_nums = []
        data_points = []
        for house_num, result in results.items():
            house_nums.append(f'House {house_num}')
            data_points.append(len(result['features_data']))

        plt.bar(house_nums, data_points, color='skyblue')
        plt.title('各家庭数据量比较')
        plt.xlabel('家庭')
        plt.ylabel('数据点数')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_report.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    print("UK-DALE 自动数据分析系统")

    #创建处理器实例
    processor = UKDALEProcessor()

    #自动处理所有可用的家庭数据
    results = processor.process_all_households(sample_freq='1H')

    if results:
        print(f"\n所有家庭数据处理完成!")
        print(f"成功处理 {len(results)} 个家庭")
        print(f"输出文件保存在: {processor.output_dir}")
    else:
        print(f"\n没有成功处理任何家庭数据")

    return processor


if __name__ == "__main__":
    processor = main()