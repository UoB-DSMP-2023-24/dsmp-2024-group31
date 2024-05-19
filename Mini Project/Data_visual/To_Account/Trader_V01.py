import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor


class Trader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.setup_pandas_options()

    @staticmethod
    def setup_pandas_options():
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)

    def plot_monopoly_money_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['monopoly_money_amount'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Monopoly Money Amount')
        plt.xlabel('Monopoly Money Amount')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def fit_kmeans(self, data, n_clusters):
        try:
            # 尝试使用指定的簇数量进行 K-Means 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
            print(f"成功使用{n_clusters}个簇进行聚类")
            return kmeans
        except ValueError as e:
            # 如果样本数量少于簇的数量，则递减簇的数量并重试
            if 'n_samples=1' in str(e) and n_clusters > 1:
                print(f"样本数量少于簇的数量{n_clusters}，尝试使用{n_clusters - 1}个簇")
                return self.fit_kmeans(data, n_clusters - 1)
            else:
                # 如果 n_clusters 已经减到 1 但仍然失败，抛出异常
                raise

    def assign_frequency_labels_and_separate_data(self):
        # 有多个聚类中心时，根据聚类结果分离数据
        frequency_labels = self.data['frequency_label'].unique()
        separated_amounts = {}
        for label in frequency_labels:
            amounts = self.data[self.data['frequency_label'] == label]['monopoly_money_amount'].values
            separated_amounts[label] = amounts
            print(f"{label}:", amounts[:10])

        return separated_amounts

    def perform_kmeans_clustering(self):
        # Calculate the frequency of each unique monopoly_money_amount value
        amount_frequency = self.data['monopoly_money_amount'].value_counts().reset_index()
        amount_frequency.columns = ['monopoly_money_amount', 'frequency']

        # Reshape the frequency data for K-means clustering
        frequencies = amount_frequency['frequency'].values.reshape(-1, 1)

        # 尝试使用初始的簇数量
        n_clusters_initial = 3
        kmeans_frequency = self.fit_kmeans(frequencies, n_clusters_initial)
        # 如果聚类数为1，则直接将所有数据视为一个通道
        if len(kmeans_frequency.cluster_centers_) > 1:
            # 根据聚类结果，为数据分配标签
            self.data['frequency_category'] = kmeans_frequency.labels_
            cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
            sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)
            frequency_mapping_frequency = {sorted_center_indices_frequency[i]: f"Frequency {i + 1}" for i in
                                           range(len(cluster_centers_frequency))}
            self.data['frequency_label'] = self.data['frequency_category'].apply(
                lambda x: frequency_mapping_frequency[x])
        else:
            self.data['frequency_label'] = 'Frequency 1'

        # Assign the labels to the frequency data
        amount_frequency['frequency_category'] = kmeans_frequency.labels_

        # Determine the cluster centers to identify which label corresponds to low, medium, and high frequency
        cluster_centers_frequency = kmeans_frequency.cluster_centers_.flatten()
        sorted_center_indices_frequency = np.argsort(cluster_centers_frequency)

    def analyze_transactions_variance(self,transactions, label):
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

    def analyze_transactions(self):
        amount_frequency = self.perform_kmeans_clustering()

        def data_span_detection(freq_transactions):
            # Your existing logic for data span detection
            pass

        def dispersion_detection(freq_transactions):
            # Your existing logic for dispersion detection
            pass

        # Analysis for each frequency category
        for label in ['Low', 'Medium', 'High']:
            freq_data = self.data[self.data['monopoly_money_amount'].isin(
                amount_frequency[amount_frequency['frequency_label'] == label]['monopoly_money_amount'])]
            print(f"Analysis for {label} Frequency Transactions:")
            dispersion_detection(freq_data)

    def trend_analysis(self):
        # Your existing trend analysis logic
        plt.figure(figsize=(14, 7))
        for label, color in zip(['High', 'Medium', 'Low'], ['green', 'blue', 'red']):
            subset = self.data[self.data['monopoly_money_amount'].isin(
                self.perform_kmeans_clustering()[self.perform_kmeans_clustering()['frequency_label'] == label][
                    'monopoly_money_amount'])]
            trend = subset.groupby('not_happened_yet_date')['monopoly_money_amount'].mean().reset_index()
            plt.plot(trend['not_happened_yet_date'], trend['monopoly_money_amount'],
                     label=f'{label} Frequency Transactions', color=color)

        plt.title('Trend of Transaction Amounts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Transaction Amount')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    xlsx_files = glob.glob('Trader/*.xlsx')

    for file_name in xlsx_files:
        if file_name.endswith('.xlsx'):
            trader = Trader(file_name)
            trader.plot_monopoly_money_distribution()
            trader.analyze_transactions()
            trader.trend_analysis()


if __name__ == "__main__":
    main()
