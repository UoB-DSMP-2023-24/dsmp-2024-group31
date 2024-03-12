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

    @staticmethod
    def setup_pandas_options():
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)

    def data_split_by_regular(self):
        # meth01:直接通过date 和amout参数进行归一，再查询是否为数据集包含的月份数，如果是那就是周期性数据
        #再通过周期性数据，对原始数据集进行映射分割
        '''
        从原始数据集中筛出以月为周期的数据，并将这些数据剔除出原数据集。
        周期性数据默认为一特征，不再进入后续特征选择和机器学习。
        :return:
        '''

        data=self.data
        # 确保 'not_happened_yet_date' 是日期格式，并提取年月和日
        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])
        data['year_month'] = data['not_happened_yet_date'].dt.to_period('M')
        data['day'] = data['not_happened_yet_date'].dt.day

        grouped = data.groupby(['from_totally_fake_account', 'to_randomly_generated_account',
                                'monopoly_money_amount', 'day'])
        monthly_counts = grouped['year_month'].nunique()

        periodic_criteria = monthly_counts[monthly_counts > 8].index

        periodic_data = data.set_index(['from_totally_fake_account', 'to_randomly_generated_account',
                                        'monopoly_money_amount', 'day'])
        periodic_data = periodic_data.loc[periodic_criteria].reset_index()

        periodic_data = periodic_data.drop_duplicates(subset=['from_totally_fake_account',
                                                              'to_randomly_generated_account',
                                                              'monopoly_money_amount', 'day', 'year_month'])

        non_periodic_data = data.drop(periodic_data.index)
        self.data=non_periodic_data


        # 显示周期性数据和非周期性数据的前几行
        # print(periodic_data)
        # print(non_periodic_data.head())

def main():
    xlsx_files = glob.glob('Personal/*.xlsx')

    for file_name in xlsx_files:
        if file_name.endswith('.xlsx'):
            print('file_name: ',file_name)
            personal_account=Personal_Account(file_name)
            personal_account.data_split_by_regular()


if __name__ == '__main__':
    main()
