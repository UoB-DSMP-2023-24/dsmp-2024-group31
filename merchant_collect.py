import pandas as pd
import os

def save_transactions_by_merchant(file_path):
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 确保'Merchant'目录存在
    output_dir = 'Merchant'
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤掉'Third Party Name'为空的行
    filtered_data = data.dropna(subset=['Third Party Name'])
    
    # 按'Third Party Name'分组
    grouped = filtered_data.groupby('Third Party Name')
    
    # 为每个分组保存一个CSV文件
    for name, group in grouped:
        # 文件名格式：{Third Party Name}_transactions.csv
        filename = f"{name}_transactions.csv"
        file_path = os.path.join(output_dir, filename)
        group.to_csv(file_path, index=False)

# 调用函数，输入数据文件的路径
save_transactions_by_merchant('cleaned_simulated_transaction_2024.csv')