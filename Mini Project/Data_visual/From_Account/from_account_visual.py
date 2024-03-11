import pandas as pd

# # 载入数据集
# file_path = '../fake_transactional_data_24.csv'
# data = pd.read_csv(file_path)

# # 获取所有唯一的账号
# unique_accounts = data['to_randomly_generated_account'].unique()
# # 为每个独特的账号创建一个Excel文件
# for account in unique_accounts:
#     # 筛选出当前账号的所有交易
#     account_data = data[data['to_randomly_generated_account'] == account]
#
#     # 定义文件名
#     file_name = f'account_{(account)}.xlsx'
#
#     # 将筛选出的数据保存为Excel文件
#     account_data.to_excel(file_name, index=False)
#
# print("文件拆分完成。")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置保存拆分文件的目录路径
split_files_directory = 'E:\Study in UK\Mini Project\Data_visual\From_Account\Personal_Account'

# 遍历目录中的每个Excel文件
for file_name in os.listdir(split_files_directory):
    if file_name.endswith('.xlsx'):
        # 载入Excel文件
        file_path = os.path.join(split_files_directory, file_name)
        data = pd.read_excel(file_path)

        data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

        # 使用Seaborn来创建带有分类颜色的散点图
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='not_happened_yet_date', y='monopoly_money_amount', hue='to_randomly_generated_account',
                        data=data, palette='viridis')

        plt.title('Monopoly Money Amount Over Time by Account')
        plt.xlabel('Date')
        plt.ylabel('Monopoly Money Amount')
        plt.xticks(rotation=45)  # 旋转x轴标签以改善可读性
        plt.legend(title='Account', bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放置在图表外侧

        plt.tight_layout()  # 调整整体布局以适应标签和图例
        plt.show()
