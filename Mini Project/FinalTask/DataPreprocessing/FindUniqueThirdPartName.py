import os

import pandas as pd
import numpy as np
import missingno as msno
@staticmethod
def setup_pandas_options():
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

df= pd.read_csv('../RawData/simulated_transaction_2024_Processed(without salary).csv')
print(df.info())
print(df.shape)
print(df.head())

msno.matrix(df)

unique_values = df['Third Party Name'].unique()

unique_values_list = unique_values.tolist()


unique_Third_Party_Name_list = [value for value in unique_values if pd.notna(value)]

print(unique_Third_Party_Name_list)

categories = {
    'Retail': ['Head','CeX','Coop Local','Boutique', 'Tesco', 'Selfridges', 'Sainsbury', 'Amazon', 'The Works', 'Blackwell\'s', 'Hobby Lobby',
               'Revella', 'Hobbycraft', 'Etsy', 'Sports Direct', 'Boots', 'Mothercare', 'Millets', 'Mountain Warehouse',
               'HMV','Happy Days Home','Mamas & Papas'],
    'Food & Drink': ['Costa Coffee', 'Starbucks', 'The Crown', 'Rose & Crown', 'Kings Arms', 'Deliveroo', 'JustEat',
                     'Coffee #1', 'Frankie & Bennies'],
    'Finance': ['Premier Finance', 'Halifax', 'LBG'],
    'Healthcare': ['Westport Care Home', 'Green Park Academy', 'Sunny Care Nursery', 'Lloyds Pharmacy',
                   'University College Hospital', 'Vision Express', 'Remedy plus care', 'Specsavers'],
    'Education': ['CPA','Lavender Primary', 'Green Park', 'Town High', 'RugbyFields'],
    'Entertainment & Media': ['Gamestation','SquareOnix','Blizzard', 'Xbox', 'Mojang Studios', 'Disney', 'Netflix'],
    'Fitness': ['PureGym', 'Grand Union BJJ'],
    'Art & Craft': ['Brilliant Brushes','Kew House','Cass Art', 'Craftastic', 'Fitted Stitch', 'A Yarn Story', 'Five Senses Art', 'Collector Cave','A Cut Above', 'Lavender Fields',],
    'Pet Care': ['Pets Corner', 'Pets at Home', 'Jollyes'],
    'Hotel':['Victoria Park'],
    'Apparel Stores': ['Topshop', 'Matalan', 'Gap Kids', 'Reebok', 'JD Sports', 'Loosely Fitted', 'Stitch By Stitch',
                       'Foyles', 'Wool','Fat Face','A Cut Above','North Face',]
}

# 创建一个空字典来存储结果
categorized_names = {category: [] for category in categories}

# 对每个名称进行分类
for name in unique_Third_Party_Name_list:
    categorized = False
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in name.lower():
                categorized_names[category].append(name)
                categorized = True
                break
        if categorized:
            break
    if not categorized:
        # 如果找不到匹配的分类，我们可以将其放在一个'其他'类别中
        if 'Other' not in categorized_names:
            categorized_names['Other'] = []
        categorized_names['Other'].append(name)

# 打印分类后的结果
for category, names in categorized_names.items():
    print(f"{category}: {names}")


base_path = '../SplitedData'
for category in categories:
    category_path = os.path.join(base_path, category)
    os.makedirs(category_path, exist_ok=True)

# 分类和保存文件
for name in unique_Third_Party_Name_list:
    categorized = False
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in name.lower():
                df[df['Third Party Name'] == name].to_csv(os.path.join(base_path, category, f"{name}.csv"), index=False)
                categorized = True
                break
        if categorized:
            break
    if not categorized:
        # 如果找不到匹配的分类，放在'其他'类别中
        other_path = os.path.join(base_path, 'Other')
        os.makedirs(other_path, exist_ok=True)
        df[df['Third Party Name'] == name].to_csv(os.path.join(other_path, f"{name}.csv"), index=False)

# 打印一条消息确认操作完成
print("Files have been categorized and saved successfully.")