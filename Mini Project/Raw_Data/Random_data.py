import pandas as pd

# 加载数据集
file_path = 'fake_transactional_data_1000000.xlsx'  # 替换为您的数据集路径
data = pd.read_excel(file_path)
data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

# 随机选择1000条记录
sampled_data = data.sample(n=500000,random_state=100)

# 保存为新的xlsx文件
output_file_path = '../Data_visual/random_sampled_data_50000.csv'  # 您希望保存文件的路径和名称
sampled_data.to_csv(output_file_path, index=False)
