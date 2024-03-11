import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

'''
###################  Merchant  #################

这个对象类的意义是在提取所有可能的使用到的关于商家的显式信息和隐式信息
所有方法的编译均保证操作的原子化，后续可以更具需求继承该类

主要包含数据提取和分析，以及数据可视化两部分：
提取分析中的算法逻辑：1、聚类，实现对不同交易类型的切割。2、分析是否越界，确定合适参数。3、时序分析，确定交易模式

################################################


'''


class Merchant_transaction_frequency:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.setup_pandas_options()

    @staticmethod
    def setup_pandas_options():
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)

    def calculate_sse(self, data, max_clusters):
        '''

        :param data:
        :param max_culsters:
        :return:
        '''
        if max_clusters > len(data):
            return self.calculate_sse(data, max_clusters - 1)
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            sse.append(kmeans.inertia_)
        return sse

    def find_knee_point(self, sse):
        knee_locator = KneeLocator(np.arange(0, len(sse)), sse, curve='convex', direction='decreasing', online=True)
        if knee_locator.elbow is None:
            print("No clear elbow point found.")
            # Handle the case where no elbow is found
            # This could be returning None, using a default value, etc.
            return 1
        else:
            return knee_locator.elbow + 1

    def fit_kmeans(self, data):
        '''
        K聚类，递归查询最优聚类簇数
        :param data:
        :param n_clusters:
        :return:
        '''
        sse = self.calculate_sse(data, max_clusters=5)

        plt.figure()
        plt.plot(range(1, len(sse) + 1), sse, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        self.knee_locator = self.find_knee_point(sse)

        print("optimal_clusters: ", self.knee_locator)

        try:
            # 尝试使用指定的簇数量进行 K-Means 聚类
            kmeans_optimal = KMeans(n_clusters=self.knee_locator, random_state=0).fit(data)
            print(f"成功使用{self.knee_locator}个簇进行聚类")
            return kmeans_optimal
        except ValueError as e:
            raise

    def data_categorisation(self):
        '''

        :return:
        '''
        print("file_path: ", self.file_path)

        self.amount_frequency = self.data['monopoly_money_amount'].value_counts().reset_index()
        self.amount_frequency.columns = ['monopoly_money_amount', 'frequency']

        # Reshape the frequency data for K-means clustering
        frequencies = self.amount_frequency['frequency'].values.reshape(-1, 1)
        # 尝试使用初始的簇数量
        kmeans_frequency = self.fit_kmeans(frequencies)
        if len(kmeans_frequency.cluster_centers_) > 1:
            # 根据聚类结果，为数据分配标签
            self.amount_frequency['frequency_category'] = kmeans_frequency.labels_
            cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
            sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)
            frequency_mapping_frequency = {sorted_center_indices_frequency[i]: f"Frequency {i + 1}" for i in
                                           range(len(cluster_centers_frequency))}
            self.amount_frequency['frequency_label'] = self.amount_frequency['frequency_category'].apply(
                lambda x: frequency_mapping_frequency[x])
        else:
            self.amount_frequency['frequency_label'] = 'Frequency 1'

        self.data_frequency_copy = pd.merge(self.data, self.amount_frequency, on='monopoly_money_amount')
        print('merge: ', self.data_frequency_copy)

        # 有多个聚类中心时，根据聚类结果分离数据
        frequency_labels = self.amount_frequency['frequency_label'].unique()
        self.freq_dispersion = []
        for label in frequency_labels:
            amounts = self.amount_frequency[self.amount_frequency['frequency_label'] == label][
                'monopoly_money_amount'].values
            print('amounts: ', amounts)
            self.freq_dispersion.append(self.dispersion_detection(amounts))
            print(f"{label}:", amounts[:20])

    def data_result_show(self):
        for i in self.freq_dispersion:
            if i != 0:
                print("数据波动异常")
                break
        print("数据波动正常")

    def calculate_mad(self, array):
        mean = np.mean(array)
        deviations = np.abs(array - mean)
        mad = np.mean(deviations)
        return mad

    def data_span_detection(self, freq_transactions):
        transaction_mean = freq_transactions.mean()
        transaction_std = freq_transactions.std()
        middle_amount = (freq_transactions.max() + freq_transactions.min()) / 2
        flag = 0
        if (transaction_mean - 2 * transaction_std > middle_amount):
            flag += 1
        elif (middle_amount > transaction_mean + 2 * transaction_std):
            flag += 2
        else:
            pass
        return flag

    def dispersion_detection(self, amounts):
        # Calculate MAD for each frequency category
        freq_mad = self.calculate_mad(amounts)

        if (freq_mad < 0.5):
            flag = self.data_span_detection(amounts)
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
            flag = self.data_span_detection(amounts)
            if flag == 1:
                print('下界越界')
            elif flag == 2:
                print('上界越界')
            elif flag == 3:
                print('双边越界')
            else:
                print('数据波动正常')

            return flag

    def analysis_by_weekday(self):
        self.weekday_transactions = self.weekday_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
            ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})
        self.weekend_transactions = self.weekend_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
            ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})
        print(self.weekday_transactions['Transactions'].mean(), '工作日时序交易数量')
        print(self.weekday_transactions['Average Amount'].mean(), '工作日时序交易数额')
        print(self.weekend_transactions['Transactions'].mean(), '周末时序交易数量')
        print(self.weekend_transactions['Average Amount'].mean(), '周末时序交易数额')

    def split_by_weekday(self):
        '''
        时序分析参考的参数可以是：时序平均交易额，时序交易量。
        加上时间轴，3个参数，是不是可以采用三维图像和三维超平面去切分呢？？
        :return:
        '''
        self.data['not_happened_yet_date'] = pd.to_datetime(self.data['not_happened_yet_date'])

        self.data['day_of_week'] = self.data['not_happened_yet_date'].dt.dayofweek
        self.data['type_of_day'] = self.data['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

        # Calculate the average number of transactions and average transaction amount per day for weekdays and weekends
        self.weekday_data = self.data[self.data['type_of_day'] == 'Weekday']
        self.weekend_data = self.data[self.data['type_of_day'] == 'Weekend']

        self.analysis_by_weekday()

        # Group by date and calculate metrics
        # weekday_transactions = weekday_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
        #     ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})
        # weekend_transactions = weekend_data.groupby('not_happened_yet_date')['monopoly_money_amount'].agg(
        #     ['count', 'mean']).rename(columns={'count': 'Transactions', 'mean': 'Average Amount'})
        #
        # print(weekday_transactions['Transactions'], '工作日时序交易数额')
        # print(weekday_transactions['Average Amount'], '工作日时序交易数量')
        # print(weekend_transactions['Transactions'], '周末时序交易数额')
        # print(weekend_transactions['Average Amount'], '周末时序交易数量')

    def amount_frequency_visualisation(self):
        """
        可视化当前账号交易金额和频率
        :return: null
        """

        self.amount_frequency.sort_values('monopoly_money_amount', inplace=True)

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(self.amount_frequency['monopoly_money_amount'], self.amount_frequency['frequency'], marker='o',
                 color='blue')
        plt.title(f'Line Plot of Monopoly Money Amount vs Frequency  -  {self.file_path}')
        plt.xlabel('Monopoly Money Amount')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def transaction_distribution_visualisation(self):
        """
        可视化聚类后不同通道的数据：平均交易额度，交易量，交易占比
        :return:null
        """

    # def time_frequency_visualisation(self):
    #     """
    #     可视化不同通道的时序交易频率
    #     :return: null
    #     """
    #     color_colum = ['green', 'blue', 'red', 'purple']
    #     frequency = ['High', 'Medium', 'Low', 'Ex-Low']
    #     plt.figure(figsize=(14, 7))
    #
    #     for i in range(self.knee_locator):
    #         # data_subset = self.data_frequency_copy[self.data_frequency_copy['frequency_category'] == i]
    #         daily_frequency = self.data_frequency_copy.groupby('not_happened_yet_date').size()
    #         plt.plot(daily_frequency.index, daily_frequency.values,
    #                  label=f'{frequency[i]} Frequency Transactions', color=color_colum[i])
    #     plt.title('Trend of Transaction Amounts Over Time')
    #     plt.xlabel('Date')
    #     plt.ylabel('Average Transaction Amount')
    #     plt.legend()
    #     plt.show()

    def time_frequency_visualisation(self):
        # 为了在3D图中使用，需要一个整数表示日期
        self.weekday_transactions['Day'] = range(len(self.weekday_transactions))
        self.weekend_transactions['Day'] = range(len(self.weekend_transactions))

        # 创建3D图
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制数据点
        ax.scatter(self.weekday_transactions['Day'], self.weekday_transactions['Transactions'],
                   self.weekday_transactions['Average Amount'],c='red')
        ax.scatter(self.weekend_transactions['Day'], self.weekend_transactions['Transactions'],
                   self.weekend_transactions['Average Amount'],c='blue')

        # 设置轴标签
        ax.set_xlabel('Day')
        ax.set_ylabel('Transactions')
        ax.set_zlabel('Average Amount')

        # 设置标题
        plt.title('Daily Transactions and Average Amount Over Time')
    def visualisation(self, amount_frequency, transaction_distribution, time_frequency):
        '''
        可视化插件启动
        :param amount_frequency:
        :param transaction_distribution:
        :param time_frequency:
        :return:
        '''
        if (amount_frequency == 1):
            self.amount_frequency_visualisation()
        if (transaction_distribution == 1):
            self.transaction_distribution_visualisation()
        if (time_frequency == 1):
            self.time_frequency_visualisation()


def main():
    xlsx_files = glob.glob('../Trader/*.xlsx')

    for file_name in xlsx_files:
        if file_name.endswith('.xlsx'):
            merchant_transaction_frequency = Merchant_transaction_frequency(file_name)
            merchant_transaction_frequency.data_categorisation()
            merchant_transaction_frequency.split_by_weekday()
            merchant_transaction_frequency.visualisation(1, 0, 1)


if __name__ == '__main__':
    main()
