import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    
def read_rfm_data(folder_path):
    all_files = os.listdir(folder_path)
    rfm_files = [file for file in all_files if file.endswith('_rfm_results.csv')]
    
    # Empty list to hold dataframes
    list_of_dataframes = []
    
    # Read each file and append to list
    for file in rfm_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        list_of_dataframes.append(df)
    
    # Concatenate all dataframes into one
    full_rfm_df = pd.concat(list_of_dataframes, ignore_index=True)
    
    return full_rfm_df

def rfm_score(folder_path, file):
    data = pd.read_csv(file)
    summary_row = data[data['from_totally_fake_account'] == 'Summary']

    # 创建一个新的 DataFrame 包含要添加的新行
    new_row = pd.DataFrame({'from_totally_fake_account': ['Score'], 'Recency': [np.nan], 'Frequency': [np.nan], 'Monetary': [np.nan]})

    # 使用 append 方法将新行添加到 DataFrame 末尾
    data = pd.concat([data, new_row], ignore_index=True)
    r_bins = [0, 73, 146, 219, 292, 365]
    f_bins = [0, 64, 128, 192, 254, 850000]
    m_bins = [0, 5000, 10000, 15000, 19190, 10640000]
    labels = [5, 4, 3, 2, 1]

    if not summary_row.empty:
        summary_recency = summary_row['Recency'].iloc[0]
        summary_frequency = summary_row['Frequency'].iloc[0]
        summary_monetary = summary_row['Monetary'].iloc[0]

        r_score = pd.cut([summary_recency], bins=r_bins, labels=labels, right=False)[0]

        f_score = pd.cut([summary_frequency], bins=f_bins, labels=labels, right=False)[0]
        f_score = 6 - f_score if pd.notnull(f_score) else f_score

        m_score = pd.cut([summary_monetary], bins=m_bins, labels=labels, right=False)[0]
        m_score = 6 - m_score if pd.notnull(m_score) else m_score

    score_index = data[data['from_totally_fake_account'] == 'Score'].index
    if not score_index.empty:
        data.loc[score_index, 'Recency'] = r_score
        data.loc[score_index, 'Frequency'] = f_score
        data.loc[score_index, 'Monetary'] = m_score
    
    output_file_path = os.path.join(folder_path, os.path.basename(file))
    data.to_csv(output_file_path, index=False)

    print(f"Save file to {output_file_path}")




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

    # RFM_process_files('data', 'RFM_data')
    folder_path = 'RFM_data'
    output_folder = 'RFM_Score'
    # Read the RFM data
    rfm_data = read_rfm_data(folder_path)
    # Remove the summary rows to avoid skewing the analysis
    rfm_data = rfm_data[rfm_data['from_totally_fake_account'] == 'Summary']
    # print(rfm_data)
    rfm_data_cleaned = rfm_data
    rfm_data_cleaned['Recency'] = pd.to_numeric(rfm_data_cleaned['Recency'])
    rfm_data_cleaned['Frequency'] = pd.to_numeric(rfm_data_cleaned['Frequency'])
    rfm_data_cleaned['Monetary'] = pd.to_numeric(rfm_data_cleaned['Monetary'])
    
    # # Sort the data for visualization
    # sorted_by_recency = rfm_data_cleaned.sort_values('Recency', ascending=True)
    # sorted_by_frequency = rfm_data_cleaned.sort_values('Frequency', ascending=True)
    # sorted_by_monetary = rfm_data_cleaned.sort_values('Monetary', ascending=True)

    # f_sum = rfm_data_cleaned.Frequency.sum()
    # m_sum = rfm_data_cleaned.Monetary.sum()

    # f_total = 0
    # for index, row in sorted_by_frequency[::-1].iterrows():
    #     f_total += row['Frequency']
    #     if f_total >= 0.8*f_sum:
    #         break
    # f_row = row

    # m_total = 0
    # for index, row in sorted_by_monetary[::-1].iterrows():
    #     m_total += row['Monetary']
    #     if m_total >= 0.8*m_sum:
    #         break
    # m_row = row
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)  # 输入文件完整路径
        output_path = os.path.join(output_folder, file)  # 输出文件完整路径
        rfm_score(output_folder, file_path)
        







