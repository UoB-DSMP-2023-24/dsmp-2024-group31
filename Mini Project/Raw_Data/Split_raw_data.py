import pandas as pd

# 载入数据集
data = pd.read_csv('fake_transactional_data_24.csv')  # 替换为您的文件路径

# 将 'not_happened_yet_date' 列转换为标准时间格式
data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'])

# 对 'from_totally_fake_account' 列进行处理，确保没有 ".0"
def convert_account(account):
    try:
        # 尝试将账号转换为整数，然后转换为字符串
        return str(int(account))
    except ValueError:
        # 如果转换失败（例如，如果账号包含字母），则直接返回字符串形式
        return str(account)

data['to_randomly_generated_account'] = data['to_randomly_generated_account'].apply(convert_account)

# 按照 'from_totally_fake_account' 分组并保存每个组到单独的文件
for account, group in data.groupby('to_randomly_generated_account'):
    file_name = f'account_{account}.xlsx'
    group.to_excel(file_name, index=False)