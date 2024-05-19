# # Re-importing necessary libraries and re-defining the dataset since the code execution state was reset
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 构建数据框架
# data = {
#     'Monopoly Money Amount': [
#         2.40, 2.65, 1.45, 2.45, 2.15, 2.25, 2.55, 1.95, 1.80, 2.20,
#         4.60, 4.80, 4.35, 5.05, 4.85, 4.20, 4.65, 4.45, 4.10, 4.40,
#         5.10, 4.90, 3.85, 5.30, 4.70, 4.00, 4.95, 4.55, 5.20, 4.50,
#         3.60, 3.90, 4.25, 4.15, 3.65, 4.05, 3.95, 4.75, 3.75, 5.00,
#         3.70, 3.40, 3.25, 4.30, 2.90
#     ],
#     'Frequency': [
#         73057, 72925, 36881, 36819, 36812, 36756, 36628, 36430, 36380, 36176,
#         2452, 2409, 2001, 1923, 1917, 1497, 1482, 1451, 1402, 1247,
#         1211, 1166, 1043, 1027, 1025, 1014, 1014, 1014, 974, 775,
#         762, 699, 511, 507, 499, 497, 496, 495, 480, 479,
#         468, 449, 445, 239, 225
#     ]
# }
#
# df = pd.DataFrame(data)
#
# # Prepare data for clustering
# X = df[['Monopoly Money Amount', 'Frequency']].values
# # Standardize data to improve DBSCAN performance
# X_scaled = StandardScaler().fit_transform(X)
#
# # Apply DBSCAN clustering algorithm
# # Initial attempt with a starting value for eps and min_samples, which may need adjustment based on the clustering outcome
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# clusters = dbscan.fit_predict(X_scaled)
#
# # Add clustering results back to the original dataframe
# df['Cluster'] = clusters
#
# # Plotting the clustering results
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Monopoly Money Amount'], df['Frequency'], c=df['Cluster'], cmap='viridis', label='Cluster')
# plt.title('DBSCAN Clustering of Monopoly Money Data')
# plt.xlabel('Monopoly Money Amount')
# plt.ylabel('Frequency')
# plt.colorbar(label='Cluster')
# plt.show()
#
# print(df['Cluster'].value_counts())


import numpy as np
# import glob
#
# import pandas as pd
#
# xlsx_files = glob.glob('account_A_LOCAL_COFFEE_SHOP.xlsx')
# for file_name in xlsx_files:
#     if file_name.endswith('.xlsx'):
#         df = pd.read_excel(file_name)
#         # # 提取monopoly_money_amount列的数据
#         # monopoly_money_amount_data = df['monopoly_money_amount']
#         # frequency_counts = monopoly_money_amount_data.value_counts()
#         #
#         # # 将频次统计结果转换为DataFrame，便于查看和分析
#         # frequency_df = frequency_counts.reset_index()
#         # frequency_df.columns = ['Monopoly Money Amount', 'Frequency']
#
# print(df['not_happened_yet_date'].describe())
#
# # 重新计算IQR
# Q1 = df['monopoly_money_amount'].quantile(0.25)
# Q3 = df['monopoly_money_amount'].quantile(0.75)
# IQR = Q3 - Q1
#
# # 定义异常值的阈值
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # 确定异常值的范围
# print(
#     f"Based on IQR method, values below {lower_bound:.2f} or above {upper_bound:.2f} could be considered as outliers.")
#
# # 检查数据集中是否存在超出这些阈值的值
# potential_outliers = df[(df['monopoly_money_amount'] < lower_bound) | (df['monopoly_money_amount'] > upper_bound)]
#
# # 计算超出阈值的交易金额的比例
# outliers_percentage = len(potential_outliers) / len(df) * 100
#
# print(f"Percentage of potential outliers: {outliers_percentage:.2f}%")
#
# # 判断是否有超出阈值的交易金额
# if len(potential_outliers) > 0:
#     print("There are potential outliers in the dataset based on the IQR method.")
# else:
#     print("There are no potential outliers in the dataset based on the IQR method.")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 替换以下路径为你的CSV文件路径
csv_file_path = 'To_Account/Trader/account_A_CAFE.xlsx'

# 读取CSV文件
df = pd.read_excel(csv_file_path)

# 1. 交易金额分析
plt.figure(figsize=(10, 6))
sns.histplot(df['monopoly_money_amount'], bins=30, kde=True, color='blue')
plt.title('Distribution of Monopoly Money Amount')
plt.xlabel('Monopoly Money Amount')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#
# # 基本统计数据
# df['monopoly_money_amount'].describe()


#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# # 替换以下路径为你的CSV文件路径
# csv_file_path = 'random_sampled_data_50000.csv'
#
# # 读取CSV文件
# data = pd.read_csv(csv_file_path)
#
# # Set the style of seaborn
# sns.set(style="whitegrid")
#
# # Plot the distribution of transaction amounts
# plt.figure(figsize=(10, 6))
# sns.histplot(data['monopoly_money_amount'], bins=50, kde=True)
# plt.title('Distribution of Transaction Amounts')
# plt.xlabel('Transaction Amount')
# plt.ylabel('Frequency')
# plt.show()
#
# # Calculate basic statistics: mean and standard deviation
# mean_amount = data['monopoly_money_amount'].mean()
# std_dev_amount = data['monopoly_money_amount'].std()
# max_amount=data['monopoly_money_amount'].max()
# min_amount=data['monopoly_money_amount'].min()
#
# print(mean_amount, std_dev_amount)
# # Define the range as mean +/- 2 standard deviations
# lower_bound = mean_amount - 2 * std_dev_amount
# upper_bound = mean_amount + 2 * std_dev_amount
#
# print(lower_bound, upper_bound)
#
# def outlier_direction(lower_bound,upper_bound,max_amount,min_amount):
#     if(min_amount<=0):
#         print('Error transaction')
#
#     elif(min_amount<lower_bound*)
#
#


# Re-import necessary libraries and re-load the data due to code execution state reset
import pandas as pd
from sklearn.cluster import KMeans

# Load the data again
csv_file_path = 'To_Account/Trader/account_A_CAFE.xlsx'

# 读取CSV文件
data = pd.read_excel(csv_file_path)
# Calculate the frequency of each unique monopoly_money_amount value
amount_frequency = data['monopoly_money_amount'].value_counts().reset_index()
amount_frequency.columns = ['monopoly_money_amount', 'frequency']

# Reshape the frequency data for K-means clustering
frequencies = amount_frequency['frequency'].values.reshape(-1, 1)

# Use KMeans to cluster the frequency data into 3 categories
kmeans_frequency = KMeans(n_clusters=3, random_state=0).fit(frequencies)

# Assign the labels to the frequency data
amount_frequency['frequency_category'] = kmeans_frequency.labels_

# Determine the cluster centers to identify which label corresponds to low, medium, and high frequency
cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)

# Map sorted indices to frequency categories
frequency_mapping_frequency = {sorted_center_indices_frequency[0]: "Low Frequency",
                               sorted_center_indices_frequency[1]: "Medium Frequency",
                               sorted_center_indices_frequency[2]: "High Frequency"}

# Re-assign the correct frequency labels based on cluster centers
amount_frequency['frequency_label'] = amount_frequency['frequency_category'].apply(
    lambda x: frequency_mapping_frequency[x])

# Separate the data into three categories based on corrected labels
low_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "Low Frequency"][
    'monopoly_money_amount'].values
medium_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "Medium Frequency"][
    'monopoly_money_amount'].values
high_freq_amounts = amount_frequency[amount_frequency['frequency_label'] == "High Frequency"][
    'monopoly_money_amount'].values

print(frequency_mapping_frequency, low_freq_amounts[:10], medium_freq_amounts[:10], high_freq_amounts[:10])

'''
已经过滤出了低，中，高，三个通道数据的值后，接下来的思路就是
1、直接对三个通道的频率值进行判断

2、再将三个通道对应的值放回到原数据集中，分别进行分析
'''

'''
方法1、直接对现有数据进行分析
'''

# 均值和标准差
stats_by_category = amount_frequency.groupby('frequency_label')['monopoly_money_amount'].agg(
    ['mean', 'std']).reset_index()
print(stats_by_category)

# Z-score
from scipy.stats import zscore

# Applying zscore within each frequency category for the transaction amounts
amount_frequency['z_score'] = amount_frequency.groupby('frequency_label')['monopoly_money_amount'].transform(
    lambda x: zscore(x, ddof=1))

# LOF
from sklearn.neighbors import LocalOutlierFactor

# Initialize the LOF model
lof_model = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.1)

# Apply LOF separately for each frequency category and add LOF scores to the dataframe
for label in amount_frequency['frequency_label'].unique():
    subset = amount_frequency[amount_frequency['frequency_label'] == label]['monopoly_money_amount'].values.reshape(-1,
                                                                                                                    1)
    lof_scores = lof_model.fit_predict(subset)
    amount_frequency.loc[amount_frequency['frequency_label'] == label, 'lof_score'] = lof_scores

print(amount_frequency)

'''
方法2、在原数据集中对三通道数据进行分析
'''

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
high_freq_trend = high_freq_transactions.groupby('not_happened_yet_date')['monopoly_money_amount'].mean().reset_index()
medium_freq_trend = medium_freq_transactions.groupby('not_happened_yet_date')[
    'monopoly_money_amount'].mean().reset_index()
low_freq_trend = low_freq_transactions.groupby('not_happened_yet_date')['monopoly_money_amount'].mean().reset_index()

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



# # Calculate overall averages
# overall_averages = pd.DataFrame({
#     'Type of Day': ['Weekday', 'Weekend'],
#     'Average Transactions': [weekday_transactions['Transactions'].mean(), weekend_transactions['Transactions'].mean()],
#     'Average Transaction Amount': [weekday_transactions['Average Amount'].mean(), weekend_transactions['Average Amount'].mean()]
# })
#
# print(overall_averages)






