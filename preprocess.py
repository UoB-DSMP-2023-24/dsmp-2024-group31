import pandas as pd
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
    elif row['to_randomly_generated_account'] in Cafe:
        return 'Cafe'
    elif row['to_randomly_generated_account'] in Supermarket:
        return 'Supermarket'
    elif row['to_randomly_generated_account'] in Other_shop:
        return 'Other Shop'
    elif row['to_randomly_generated_account'] in Drink:
        return 'Drink'
    elif row['to_randomly_generated_account'] in Book_shop:
        return 'Book Shop'
    elif row['to_randomly_generated_account'] in Butcher:
        return 'Butcher'
    elif row['to_randomly_generated_account'] in Restaurant:
        return 'Restaurant'
    elif row['to_randomly_generated_account'] in Media:
        return 'Media'
    elif row['to_randomly_generated_account'] in Fashion:
        return 'Fashion'
    elif row['to_randomly_generated_account'] in Electronic:
        return 'Electronic'
    elif row['to_randomly_generated_account'] in Child:
        return 'Child'
    elif row['to_randomly_generated_account'] in Flower:
        return 'Flower'
    else:
        # 如果都不匹配，可以指定默认的标签
        return 'Other'

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
    df = pd.read_csv('fake_transactional_data_24.csv')
    preprocess('fake_transactional_data_24.csv', 'to_randomly_generated_account')
    df['tag'] = df.apply(create_tag, axis=1)
    df.to_csv('new_transactions.csv', index=False)


