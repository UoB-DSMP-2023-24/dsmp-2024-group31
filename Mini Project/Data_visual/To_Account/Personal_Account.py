import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


class Personal_Account:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.data['not_happened_yet_date'] = pd.to_datetime(self.data['not_happened_yet_date'])
        self.setup_pandas_options()

        self.visualisation_flag = {'monthly_visualisation': 0}

    @staticmethod
    def setup_pandas_options():
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        pd.set_option('display.max_rows', 100)  # 设置最大显示的行数

    def data_split_by_regular(self):
        # meth01:直接通过date 和amout参数进行归一，再查询是否为数据集包含的月份数，如果是那就是周期性数据
        # 再通过周期性数据，对原始数据集进行映射分割
        '''
        从原始数据集中筛出以月为周期的数据，并将这些数据剔除出原数据集。
        周期性数据默认为一特征，不再进入后续特征选择和机器学习。
        :return:
        '''

        data = self.data
        # 确保 'not_happened_yet_date' 是日期格式，并提取年月和日
        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])
        data['year_month'] = data['not_happened_yet_date'].dt.to_period('M')
        data['day'] = data['not_happened_yet_date'].dt.day

        data['is_periodic'] = 0

        data['transaction_num'] = data.groupby(['from_totally_fake_account', 'not_happened_yet_date']).cumcount() + 1

        grouped = data.groupby(['from_totally_fake_account', 'to_randomly_generated_account',
                                'monopoly_money_amount', 'day'])
        monthly_counts = grouped['year_month'].nunique()

        periodic_criteria = monthly_counts[monthly_counts > 8].reset_index()

        # 标记周期性数据
        for _, row in periodic_criteria.iterrows():
            data.loc[(data['from_totally_fake_account'] == row['from_totally_fake_account']) &
                     (data['to_randomly_generated_account'] == row['to_randomly_generated_account']) &
                     (data['monopoly_money_amount'] == row['monopoly_money_amount']) &
                     (data['day'] == row['day']), 'is_periodic'] = 1

        # 分离周期性和非周期性数据
        self.periodic_data = data[data['is_periodic'] == 1]
        self.non_periodic_data = data[data['is_periodic'] == 0]
        self.data = pd.concat([self.periodic_data, self.non_periodic_data], ignore_index=True)
        if self.periodic_data.size != 0:
            self.monthly_feature_visualisation()

        # 显示周期性数据和非周期性数据的前几行
        # print(periodic_data)
        # print(non_periodic_data.head())

    def monthly_feature_visualisation(self):
        if self.visualisation_flag['monthly_visualisation'] == 1:
            # 将日期转换为数值型（例如，从某个固定日期起的天数）
            reference_date = pd.to_datetime('2025-01-01')  # 可以选择一个合适的参考日期
            self.data['date_num'] = (self.data['not_happened_yet_date'] - reference_date).dt.days
            for account in self.periodic_data['from_totally_fake_account'].unique():
                data = self.data[self.data['from_totally_fake_account'] == account]
                print(self.file_path)

                print(account)

                print('periodic_data\n',self.periodic_data)

                print('non_periodic_data\n',self.non_periodic_data)

                # 创建一个新的 figure
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 获取x, y, z值
                xs = data['date_num']
                ys = data['transaction_num']
                zs = data['monopoly_money_amount']

                # 根据 is_periodic 为 1 的情况确定颜色
                colors = ['red' if is_periodic else 'blue' for is_periodic in data['is_periodic']]

                # 绘制散点图
                ax.scatter(xs, ys, zs, c=colors)

                ax.set_xlabel('Date (days since 2025-01-01)')
                ax.set_ylabel('Transaction Number in same day')
                ax.set_zlabel('Monopoly Money Amount')

                ax.set_title(f'3D Scatter Plot of Transactions account:{account}')
                # 显示图形
                plt.show()

    # def monthly_feature_visualisation(self):
    #     if self.visualisation_flag['monthly_visualisation'] == 1:
    #         # 将日期转换为数值型
    #         reference_date = pd.to_datetime('2025-01-01')
    #         self.data['date_num'] = (self.data['not_happened_yet_date'] - reference_date).dt.days
    #         for account in self.periodic_data['from_totally_fake_account'].unique():
    #             data = self.data[self.data['from_totally_fake_account'] == account]
    #
    #             # 创建新的 figure
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111, projection='3d')
    #
    #             # 获取x, y, z值
    #             xs = data['date_num']
    #             ys = data['transaction_num']
    #             zs = np.zeros_like(xs)  # 基底位置设为 0
    #             dx = dy = np.ones_like(zs)  # 单位大小的立方体
    #             dz = data['monopoly_money_amount']
    #
    #             # 定义颜色
    #             colors = ['red' if is_periodic else 'blue' for is_periodic in data['is_periodic']]
    #
    #             # 绘制柱状图
    #             ax.bar3d(xs, ys, zs, dx, dy, dz, color=colors)
    #
    #             # 设置轴标签
    #             ax.set_xlabel('Date (days since 2025-01-01)')
    #             ax.set_ylabel('Transaction Number')
    #             ax.set_zlabel('Monopoly Money Amount')
    #
    #             # 设置标题
    #             ax.set_title(f'3D Bar Chart of Transactions for Account: {account}')
    #
    #             # 显示图形
    #             plt.show()

    # # 将日期转换为数值型（例如，从某个固定日期起的天数）
    # reference_date = pd.to_datetime('2025-01-01')  # 可以选择一个合适的参考日期
    # self.data['date_num'] = (self.data['not_happened_yet_date'] - reference_date).dt.days
    #
    # # 创建一个新的 figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 遍历每个周期性账户
    # for account in self.periodic_data['from_totally_fake_account'].unique():
    #     # 获取当前账户的数据
    #     account_data = self.data[self.data['from_totally_fake_account'] == account]
    #
    #     # 获取x, y, z值
    #     xs = account_data['date_num']
    #     ys = account_data['transaction_num']
    #     zs = account_data['monopoly_money_amount']
    #
    #     # 根据 is_periodic 为 1 的情况确定颜色
    #     colors = ['red' if is_periodic else 'blue' for is_periodic in account_data['is_periodic']]
    #
    #     # 绘制柱状图
    #     ax.bar3d(xs, ys, np.zeros_like(zs), 1, 1, zs, color=colors)
    #
    # # 设置轴标签
    # ax.set_xlabel('Date (days since 2025-01-01)')
    # ax.set_ylabel('Transaction Number')
    # ax.set_zlabel('Monopoly Money Amount')
    #
    # # 显示图形
    # plt.show()

    def visualisation(self, amount_frequency):
        '''
        可视化插件启动
        :param amount_frequency:
        :param transaction_distribution:
        :param time_frequency:
        :return:
        '''
        self.visualisation_flag["monthly_visualisation"] = amount_frequency


def main():
    xlsx_files = glob.glob('Personal/*.xlsx')

    for file_name in xlsx_files:
        if file_name.endswith('.xlsx'):
            print('file_name: ', file_name)

            personal_account = Personal_Account(file_name)
            personal_account.visualisation(1)

            personal_account.data_split_by_regular()


if __name__ == '__main__':
    main()
