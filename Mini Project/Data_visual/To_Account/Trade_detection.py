import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob

# pandas显示完整输出内容
pd.set_option('display.max_columns', 1000)  # 最大行数
pd.set_option('display.width', 1000)  # 字段宽度
pd.set_option('display.max_colwidth', 1000)  # 字段内容宽度


'''
区分显示商家用户
'''


# 设置保存拆分文件的目录路径
xlsx_files = glob.glob('Trader/*.xlsx')

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

for file_name in xlsx_files:
    if file_name.endswith('.xlsx'):
        data = pd.read_excel(file_name)
        print(file_name)

        # 1. 交易金额分析
        plt.figure(figsize=(10, 6))
        sns.histplot(data['monopoly_money_amount'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Monopoly Money Amount')
        plt.xlabel('Monopoly Money Amount')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # 读取CSV文件
        # Calculate the frequency of each unique monopoly_money_amount value
        amount_frequency = data['monopoly_money_amount'].value_counts().reset_index()
        amount_frequency.columns = ['monopoly_money_amount', 'frequency']

        # Reshape the frequency data for K-means clustering
        frequencies = amount_frequency['frequency'].values.reshape(-1, 1)


        def fit_kmeans(data, n_clusters):
            try:
                # 尝试使用指定的簇数量进行 K-Means 聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
                print(f"成功使用{n_clusters}个簇进行聚类")
                return kmeans
            except ValueError as e:
                # 如果样本数量少于簇的数量，则递减簇的数量并重试
                if 'n_samples=1' in str(e) and n_clusters > 1:
                    print(f"样本数量少于簇的数量{n_clusters}，尝试使用{n_clusters - 1}个簇")
                    return fit_kmeans(data, n_clusters - 1)
                else:
                    # 如果 n_clusters 已经减到 1 但仍然失败，抛出异常
                    raise


        # 尝试使用初始的簇数量
        n_clusters_initial = 3
        kmeans_frequency = fit_kmeans(frequencies, n_clusters_initial)
        if len(kmeans_frequency.cluster_centers_) > 1:
            # 根据聚类结果，为数据分配标签
            amount_frequency['frequency_category'] = kmeans_frequency.labels_
            cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
            sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)
            frequency_mapping_frequency = {sorted_center_indices_frequency[i]: f"Frequency {i + 1}" for i in
                                           range(len(cluster_centers_frequency))}
            amount_frequency['frequency_label'] = amount_frequency['frequency_category'].apply(
                lambda x: frequency_mapping_frequency[x])
        else:
            amount_frequency['frequency_label'] = 'Frequency 1'


        # 有多个聚类中心时，根据聚类结果分离数据
        frequency_labels = amount_frequency['frequency_label'].unique()
        separated_amounts = {}
        for label in frequency_labels:
            amounts = amount_frequency[amount_frequency['frequency_label'] == label]['monopoly_money_amount'].values
            separated_amounts[label] = amounts
            print(f"{label}:", amounts[:20])


        # # Assign the labels to the frequency data
        # amount_frequency['frequency_category'] = kmeans_frequency.labels_
        #
        # # Determine the cluster centers to identify which label corresponds to low, medium, and high frequency
        # cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
        # sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)
        #
        # # Map sorted indices to frequency categories
        # frequency_mapping_frequency = {sorted_center_indices_frequency[0]: "Low Frequency",
        #                                sorted_center_indices_frequency[1]: "Medium Frequency",
        #                                sorted_center_indices_frequency[2]: "High Frequency"}
        #
        # # Re-assign the correct frequency labels based on cluster centers
        # amount_frequency['frequency_label'] = amount_frequency['frequency_category'].apply(
        #     lambda x: frequency_mapping_frequency[x])
        #
        # # Separate the data into three categories based on corrected labels
        # low_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "Low Frequency"][
        #     'monopoly_money_amount'].values
        # medium_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "Medium Frequency"][
        #     'monopoly_money_amount'].values
        # high_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "High Frequency"][
        #     'monopoly_money_amount'].values

        print(frequency_mapping_frequency)


'''
        ''''''
        已经过滤出了低，中，高，三个通道数据的值后，接下来的思路就是
        1、直接对三个通道的频率值进行判断

        2、再将三个通道对应的值放回到原数据集中，分别进行分析
        ''''''

        ''''''
        方法1、直接对现有数据进行分析
        ''''''

        # # 均值和标准差
        # stats_by_category = amount_frequency.groupby('frequency_label')['monopoly_money_amount'].agg(
        #     ['mean', 'std']).reset_index()
        # print(stats_by_category)
        #
        # # Z-score
        # from scipy.stats import zscore
        #
        # # Applying zscore within each frequency category for the transaction amounts
        # amount_frequency['z_score'] = amount_frequency.groupby('frequency_label')['monopoly_money_amount'].transform(
        #     lambda x: zscore(x, ddof=1))
        #
        # # LOF
        # from sklearn.neighbors import LocalOutlierFactor
        #
        # # Initialize the LOF model
        # lof_model = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.1)
        #
        # # Apply LOF separately for each frequency category and add LOF scores to the dataframe
        # for label in amount_frequency['frequency_label'].unique():
        #     subset = amount_frequency[amount_frequency['frequency_label'] == label][
        #         'monopoly_money_amount'].values.reshape(-1,
        #                                                 1)
        #     lof_scores = lof_model.fit_predict(subset)
        #     amount_frequency.loc[amount_frequency['frequency_label'] == label, 'lof_score'] = lof_scores
        #
        # print(amount_frequency)

        ''''''
        方法2、在原数据集中对三通道数据进行分析
        ''''''

        # 分离原数据集为三个不同频率的子集
        high_freq_transactions = data[data['monopoly_money_amount'].isin(high_freq_amounts)]
        medium_freq_transactions = data[data['monopoly_money_amount'].isin(medium_freq_amounts)]
        low_freq_transactions = data[data['monopoly_money_amount'].isin(low_freq_amounts)]

        # 检查每个子集的数据量
        high_freq_count = high_freq_transactions.shape[0]
        medium_freq_count = medium_freq_transactions.shape[0]
        low_freq_count = low_freq_transactions.shape[0]

        print(high_freq_count, medium_freq_count, low_freq_count)

        print(high_freq_transactions['monopoly_money_amount'].describe(),
              medium_freq_transactions['monopoly_money_amount'].describe(),
              low_freq_transactions['monopoly_money_amount'].describe())


        def calculate_mad(series):
            return series.mad()


        def data_span_detection(freq_transactions):
            transaction_mean = freq_transactions['monopoly_money_amount'].mean()
            transaction_std = freq_transactions['monopoly_money_amount'].std()
            middle_amount = (freq_transactions['monopoly_money_amount'].max() + freq_transactions[
                'monopoly_money_amount'].min()) / 2
            flag = 0
            if (transaction_mean - 2 * transaction_std > middle_amount):
                flag += 1
            elif (middle_amount > transaction_mean + 2 * transaction_std):
                flag += 2
            else:
                pass
            return flag


        def dispersion_detection(freq_transactions):
            # Calculate MAD for each frequency category
            freq_mad = calculate_mad(freq_transactions['monopoly_money_amount'])
            print(freq_mad)

            if (freq_mad < 0.5):
                flag = data_span_detection(freq_transactions)
                if flag == 1:
                    print('下界越界')
                elif flag == 2:
                    print('上界越界')
                elif flag == 3:
                    print('双边越界')
                else:
                    print('数据波动正常')

                return flag
            else:
                flag = data_span_detection(freq_transactions)
                if flag == 1:
                    print('下界越界')
                elif flag == 2:
                    print('上界越界')
                elif flag == 3:
                    print('双边越界')
                else:
                    print('数据波动正常')

                return flag


        # def outlier_detection(high_freq_transactions, medium_freq_transactions, low_freq_transactions):
        high_freq_dispersion = dispersion_detection(high_freq_transactions)  # 是否存在数据非常离散的情况
        medium_freq_dispersion = dispersion_detection(medium_freq_transactions)  # 是否存在数据非常离散的情况
        low_freq_dispersion = dispersion_detection(low_freq_transactions)  # 是否存在数据非常离散的情况

        if (high_freq_dispersion == 0 and medium_freq_dispersion == 0 and low_freq_dispersion == 0):
            print('该数据集波动正常')
        else:
            print('该数据集波动异常')

        # 交易时序

        # Convert transaction date to datetime format for trend analysis
        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

        # Analyze the trend of transaction amounts over time for medium and low frequency transactions
        # Group by date and calculate the average transaction amount for each day
        high_freq_trend = high_freq_transactions.groupby('not_happened_yet_date')[
            'monopoly_money_amount'].mean().reset_index()
        medium_freq_trend = medium_freq_transactions.groupby('not_happened_yet_date')[
            'monopoly_money_amount'].mean().reset_index()
        low_freq_trend = low_freq_transactions.groupby('not_happened_yet_date')[
            'monopoly_money_amount'].mean().reset_index()

        # Plotting the trend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7))

        plt.plot(high_freq_trend['not_happened_yet_date'], high_freq_trend['monopoly_money_amount'],
                 label='High Frequency Transactions', color='green')
        plt.plot(medium_freq_trend['not_happened_yet_date'], medium_freq_trend['monopoly_money_amount'],
                 label='Medium Frequency Transactions', color='blue')
        plt.plot(low_freq_trend['not_happened_yet_date'], low_freq_trend['monopoly_money_amount'],
                 label='Low Frequency Transactions', color='red')

        plt.title('Trend of Transaction Amounts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Transaction Amount')
        plt.legend()
        plt.grid(True)

        plt.show()

        import matplotlib.pyplot as plt

        # Convert the date column to datetime type
        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

        # Group by date and count transactions
        daily_transactions = data.groupby('not_happened_yet_date').size()

        # Plot
        plt.figure(figsize=(15, 6))
        daily_transactions.plot(kind='line', title='Daily Transactions Count', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Number of Transactions')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Convert the date column to datetime type
        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

        # Add a column to identify whether the transaction occurred on a weekday or a weekend
        data['day_of_week'] = data['not_happened_yet_date'].dt.dayofweek
        data['type_of_day'] = data['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

        # Calculate the average number of transactions and average transaction amount per day for weekdays and weekends
        weekday_data = data[data['type_of_day'] == 'Weekday']
        weekend_data = data[data['type_of_day'] == 'Weekend']

        # Group by date and calculate metrics
        weekday_transactions = weekday_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
            ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})
        weekend_transactions = weekend_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
            ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})

        print(weekend_transactions.describe())


        def analyze_transactions_variance(transactions, label):
            """
            Analyzes the variance of transactions data and prints out the results.

            Parameters:
            - transactions: Pandas Series with transaction data.
            - label: Label for the transaction data being analyzed.
            """
            # Calculate the middle point of the transaction amounts
            middle = (transactions.max() + transactions.min()) / 2
            # Calculate the bounds for normal variance
            lower_bound = transactions.mean() - 2 * transactions.std()
            upper_bound = transactions.mean() + 2 * transactions.std()

            # Determine if the middle point is outside the normal bounds
            if middle > upper_bound:
                print(f'{label} 上界越界')
            elif middle < lower_bound:
                print(f'{label} 下界越界')
            else:
                print(f'{label} 波动正常')


        # 需要先计算weekday和weekend的交易数据的统计量
        # 假设已经有weekday_transactions和weekend_transactions数据集，以下为示例调用
        analyze_transactions_variance(weekday_transactions['Transactions'], '工作日时序交易数额')
        analyze_transactions_variance(weekday_transactions['Average Amount'], '工作日时序交易数量')
        analyze_transactions_variance(weekend_transactions['Transactions'], '周末时序交易数额')
        analyze_transactions_variance(weekend_transactions['Average Amount'], '周末时序交易数量')


'''