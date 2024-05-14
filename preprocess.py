import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

Cafe = ['A_CAFE', 'A_LOCAL_COFFEE_SHOP', 'CAFE', 'COFFEE_SHOP', 'GOURMET_COFFEE_SHOP', 'HIPSTER_COFFEE_SHOP', 'PRETENTIOUS_COFFEE_SHOP', 
        'TOTALLY_A_REAL_COFFEE_SHOP']

Supermarket = ['A_SUPERMARKET', 'DEPARTMENT_STORE', 'EXPRESS_SUPERMARKET', 'LARGE_SUPERMARKET', 'THE_SUPERMARKET']

Other_shop = ['ACCESSORY_SHOP', 'COOKSHOP', 'DIY_STORE', 'GREENGROCER', 'GYM', 'HOME_IMPROVEMENT_STORE', 
              'JEWLLERY_SHOP', 'PET_SHOP', 'PET_TOY_SHOP']

Drink = ['BAR', 'COCKTAIL_BAR', 'G&T_BAR', 'LIQUOR_STORE', 'LOCAL_PUB', 'LOCAL_WATERING_HOLE', 'PUB', 'TEA_SHOP', 'WHISKEY_BAR', 
         'WHISKEY_SHOP', 'WINE_BAR', 'WINE_CELLAR']

Book_shop = ['BOOKSHOP', 'COMIC_BOOK_SHOP', 'LOCAL_BOOKSHOP', 'NERDY_BOOK_STORE', 'SECOND_HAND_BOOKSHOP']

Butcher = ['BUTCHER', 'BUTCHERS']

Restaurant = ['CHINESE_RESTAURANT', 'CHINESE_TAKEAWAY', 'INDIAN_RESTAURANT', 'KEBAB_SHOP', 'LOCAL_RESTAURANT', 
              'LUNCH_PLACE', 'LUNCH_VAN', 'RESTAURANT', 'RESTAURANT_VOUCHER', 'ROASTERIE', 'SANDWICH_SHOP', 
              'SEAFOOD_RESAURANT', 'STEAK_HOUSE', 'TAKEAWAY_CURRY', 'TAKEAWAY', 'TO_BEAN_OR_NOT_TO_BEAN', 
              'TURKEY_FARM', 'WE_HAVE_BEAN_WEIGHTING']

Media = ['CINEMA', 'DVD_SHOP', 'GAME_SHOP', 'STREAMING_SERVICE', 'VIDEO_GAME_STORE']

Fashion = ['CLOTHES_SHOP', 'FASHION_SHOP', 'FASHIONABLE_SPORTSWARE_SHOP', 'KIDS_CLOTHING_SHOP', 'RUNNING_SHOP', 
           'SPORT_SHOP', 'TRAINER_SHOP']

Electronic = ['ELECTRONIC_SHOP', 'HIPSTER_ELECTRONICS_SHOP', 'TECH_SHOP']

Child = ['CHILDRENDS_SHOP', 'KIDS_ACTIVITY_CENTRE', 'SCHOOL_SUPPLY_STORE', 'TOY_SHOP']

Flower = ['FLORIST']

categories_dict = {
    'Cafe': Cafe,
    'Supermarket': Supermarket,
    'Other_shop': Other_shop,
    'Drink': Drink,
    'Book_shop': Book_shop,
    'Butcher': Butcher,
    'Restaurant': Restaurant,
    'Media': Media,
    'Fashion': Fashion,
    'Electronic': Electronic,
    'Child': Child,
    'Flower': Flower
}

def preprocess(file_name, column_name):
    df = pd.read_csv(file_name)
    grouped_data = df.groupby(column_name)
    # column_name == 'from_totally_fake_account'
    # for source_account, group in grouped_data:
    #     # 输出每个源账户的交易数据
    #     print(f"Source Account: {source_account}")
    #     print(group)
    #     print('\n')
    for source_account, group in grouped_data:
        # 设置保存路径
        save_path = f'data/{source_account}_transactions.csv'
    
        # 将分组数据保存到新的CSV文件
        group.to_csv(save_path, index=False)
        print(f"Data for Source Account {source_account} saved to {save_path}")

def create_tag(row):
    if str(row['to_randomly_generated_account']).isdigit():
        return 'Personal'
    
    for category, items in categories_dict.items():
        if row['to_randomly_generated_account'] in items:
            return category
    
    else: 
        return 'Other'

def rfm_analysis(data, output_file):
    # 转换日期格式
    data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'], format='%d/%m/%Y')

    # 计算 Recency，取最近一次购买的日期
    recency_df = data.groupby('from_totally_fake_account')['not_happened_yet_date'].max().reset_index()
    recency_df['Recency'] = (pd.to_datetime('2026-1-1') - recency_df['not_happened_yet_date']).dt.days

    # 计算 Frequency，即购买次数
    frequency_df = data.groupby('from_totally_fake_account')['to_randomly_generated_account'].count().reset_index()
    frequency_df.rename(columns={'to_randomly_generated_account': 'Frequency'}, inplace=True)

    # 计算 Monetary，即总购买金额
    monetary_df = data.groupby('from_totally_fake_account')['monopoly_money_amount'].sum().reset_index()
    monetary_df.rename(columns={'monopoly_money_amount': 'Monetary'}, inplace=True)

    # 合并 Recency、Frequency、Monetary 到一个 DataFrame
    rfm_df = pd.merge(recency_df[['from_totally_fake_account', 'Recency']],
                      frequency_df[['from_totally_fake_account', 'Frequency']],
                      on='from_totally_fake_account')

    rfm_df = pd.merge(rfm_df, monetary_df[['from_totally_fake_account', 'Monetary']],
                      on='from_totally_fake_account')

    # 计算 R 的最小值、F 的总和、M 的总和
    r_summary = recency_df['Recency'].min()
    f_summary = frequency_df['Frequency'].sum()
    m_summary = monetary_df['Monetary'].sum()

    # 添加汇总结果为一行
    summary_row = pd.DataFrame({'from_totally_fake_account': ['Summary'],
                                'Recency': [r_summary],
                                'Frequency': [f_summary],
                                'Monetary': [m_summary]})

    # 将汇总结果添加到 RFM DataFrame
    rfm_df = pd.concat([rfm_df, summary_row], ignore_index=True)

    # 将结果保存到 CSV 文件
    rfm_df.to_csv(output_file, index=False)
    print(f"RFM analysis results saved to {output_file}")

    return rfm_df

def RFM_process_files(input_folder='data', output_folder='RFM_data'):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹下所有文件
    files = [f for f in os.listdir(input_folder) if f.endswith('_transactions.csv')]

    for file in files:
        # 生成输出文件名
        output_file = os.path.join(output_folder, f"{file.replace('_transactions.csv', '_rfm_results.csv')}")

        # 读取源文件
        file_path = os.path.join(input_folder, file)
        data = pd.read_csv(file_path)

        # 执行 RFM 分析
        result_df = rfm_analysis(data, output_file = output_file)
    

# # 读取CSV文件
# df = pd.read_csv('sub_fake_transactional_data_24.csv')

# # 按照源账户分组
# grouped_data = df.groupby('from_totally_fake_account')

# for source_account, group in grouped_data:
#     # 输出每个源账户的交易数据
#     print(f"Source Account: {source_account}")
#     print(group)
#     print('\n')
        
if __name__ == "__main__":
    # df = pd.read_csv('fake_transactional_data_24.csv')

    # preprocess('fake_transactional_data_24.csv', 'to_randomly_generated_account')

    # df['tag'] = df.apply(create_tag, axis=1)
    # df.to_csv('new_transactions.csv', index=False)

    RFM_process_files('data', 'RFM_data')



