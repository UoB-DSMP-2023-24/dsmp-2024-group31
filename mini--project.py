import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 导入CSV文件
data = pd.read_csv('E:\\ATB2\\miniproject\\data\\fake_transactional_data_24.csv')


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

#计算不同类别的商家的月均销售额
# 将商店分类
is_cafe = data['to_randomly_generated_account'].isin(categories_dict['Cafe'])
cafe_sales = data[is_cafe]

is_supermarket = data['to_randomly_generated_account'].isin(categories_dict['Supermarket'])
supermarket_sales = data[is_supermarket]

is_other_shop = data['to_randomly_generated_account'].isin(categories_dict['Other_shop'])
other_shop_sales = data[is_other_shop]

is_drink = data['to_randomly_generated_account'].isin(categories_dict['Drink'])
drink_sales = data[is_drink]

is_book_shop = data['to_randomly_generated_account'].isin(categories_dict['Book_shop'])
book_shop_sales = data[is_book_shop]

is_butcher = data['to_randomly_generated_account'].isin(categories_dict['Butcher'])
butcher_sales = data[is_butcher]

is_restaurant = data['to_randomly_generated_account'].isin(categories_dict['Restaurant'])
restaurant_sales = data[is_restaurant]

is_media = data['to_randomly_generated_account'].isin(categories_dict['Media'])
media_sales = data[is_media]

# 将日期信息转换为日期时间对象
cafe_sales['not_happened_yet_date'] = pd.to_datetime(cafe_sales['not_happened_yet_date'], format='%d/%m/%Y')

supermarket_sales['not_happened_yet_date'] = pd.to_datetime(supermarket_sales['not_happened_yet_date'], format='%d/%m/%Y')

other_shop_sales['not_happened_yet_date'] = pd.to_datetime(other_shop_sales['not_happened_yet_date'], format='%d/%m/%Y')

drink_sales['not_happened_yet_date'] = pd.to_datetime(drink_sales['not_happened_yet_date'], format='%d/%m/%Y')

book_shop_sales['not_happened_yet_date'] = pd.to_datetime(book_shop_sales['not_happened_yet_date'], format='%d/%m/%Y')

butcher_sales['not_happened_yet_date'] = pd.to_datetime(butcher_sales['not_happened_yet_date'], format='%d/%m/%Y')

restaurant_sales['not_happened_yet_date'] = pd.to_datetime(restaurant_sales['not_happened_yet_date'], format='%d/%m/%Y')

media_sales['not_happened_yet_date'] = pd.to_datetime(media_sales['not_happened_yet_date'], format='%d/%m/%Y')

# 按月份分组，并计算每个商店每月的销售额
cafe_sales['month'] = cafe_sales['not_happened_yet_date'].dt.to_period('M')
cafe_monthly_sales = cafe_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

supermarket_sales['month'] = supermarket_sales['not_happened_yet_date'].dt.to_period('M')
supermarket_monthly_sales = supermarket_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

other_shop_sales['month'] = other_shop_sales['not_happened_yet_date'].dt.to_period('M')
other_shop_monthly_sales = other_shop_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

drink_sales['month'] = drink_sales['not_happened_yet_date'].dt.to_period('M')
drink_monthly_sales = drink_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

book_shop_sales['month'] = book_shop_sales['not_happened_yet_date'].dt.to_period('M')
book_shop_monthly_sales = book_shop_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

butcher_sales['month'] = butcher_sales['not_happened_yet_date'].dt.to_period('M')
butcher_monthly_sales = butcher_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

restaurant_sales['month'] = restaurant_sales['not_happened_yet_date'].dt.to_period('M')
restaurant_monthly_sales = restaurant_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

media_sales['month'] = media_sales['not_happened_yet_date'].dt.to_period('M')
media_monthly_sales = media_sales.groupby(['to_randomly_generated_account', 'month'])['monopoly_money_amount'].sum().reset_index()

# 计算各个类别商店每月销售额的平均值
cafe_average_monthly_sales = cafe_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

supermarket_average_monthly_sales = supermarket_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

other_shop_average_monthly_sales = other_shop_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

drink_average_monthly_sales = drink_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

book_shop_average_monthly_sales = book_shop_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

butcher_average_monthly_sales = butcher_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

restaurant_average_monthly_sales = restaurant_monthly_sales.groupby('month')['monopoly_money_amount'].mean()

media_average_monthly_sales = media_monthly_sales.groupby('month')['monopoly_money_amount'].mean()


# 打印结果
print("Cafe 月均销售额：")
print(cafe_average_monthly_sales)

print("\nSupermarket 月均销售额：")
print(supermarket_average_monthly_sales)

print("\nOther_shop 月均销售额：")
print(other_shop_average_monthly_sales)

print("\nDrink 月均销售额：")
print(drink_average_monthly_sales)

print("\nBook_shop 月均销售额：")
print(book_shop_average_monthly_sales)

print("\nButcher 月均销售额：")
print(butcher_average_monthly_sales)

print("\nRestaurant 月均销售额：")
print(restaurant_average_monthly_sales)

print("\nMedia 月均销售额：")
print(media_average_monthly_sales)

# 将 Period 对象转换为浮点数
cafe_average_monthly_sales.index = cafe_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

supermarket_average_monthly_sales.index = supermarket_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

other_shop_average_monthly_sales.index = other_shop_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

drink_average_monthly_sales.index = drink_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

book_shop_average_monthly_sales.index = book_shop_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

butcher_average_monthly_sales.index = butcher_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

restaurant_average_monthly_sales.index = restaurant_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

media_average_monthly_sales.index = media_average_monthly_sales.index.to_timestamp().to_numpy().astype(float)

#尝试将上述表格合并成一个，但是失败，已经将month从period类型改为浮点数

# 可视化月均销售额
plt.figure(figsize=(10, 6))
plt.plot(cafe_average_monthly_sales.index, cafe_average_monthly_sales.values, label='Cafe', color='blue', marker='o')
plt.plot(supermarket_average_monthly_sales.index, supermarket_average_monthly_sales.values, label='Supermarket', color='green', marker='s')
plt.plot(other_shop_average_monthly_sales.index, other_shop_average_monthly_sales.values, label='Other_shop', color='red', marker='*')
plt.plot(drink_average_monthly_sales.index, drink_average_monthly_sales.values, label='Drink', color='yellow', marker='+')
plt.plot(book_shop_average_monthly_sales.index, book_shop_average_monthly_sales.values, label='Book_shop', color='orange', marker='^')
plt.plot(butcher_average_monthly_sales.index, butcher_average_monthly_sales.values, label='Butcher', color='pink', marker='.')
plt.plot(restaurant_average_monthly_sales.index, restaurant_average_monthly_sales.values, label='Restaurant', color='brown', marker=',')
plt.plot(media_average_monthly_sales.index, media_average_monthly_sales.values, label='Media', color='purple', marker='1')

plt.title('Average Monthly Sales of Different Kinds Of Merchant')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.legend()
plt.xticks([])
plt.grid(True)
plt.tight_layout()
plt.show()