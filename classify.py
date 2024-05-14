'''
'Halifax' 'LBG' 'Blizzard' 'Xbox' 'Mojang Studios' 'PureGym' 'Disney'
 'Netflix' 'Grand Union BJJ' 'Amazon' 'SquareOnix' 'Deliveroo' 'JustEat'  
 'Coop Local' 'Selfridges' 'Sainsbury' 'AMAZON' 'The Works' "Blackwell's" 
 'Fat Face' 'Topshop' 'Matalan' 'Foyles' 'Tesco' 'Wool' 'Hobby Lobby'     
 'Revella' 'Sainsbury Local' 'Starbucks' 'Loosely Fitted'
 'Stitch By Stitch' 'Coffee #1' 'Hobbycraft' 'A Yarn Story' 'Craftastic'
 'Kings Arms' 'Costa Coffee' 'The Crown' 'Lloyds Pharmacy' 'Rose & Crown'
 'Fitted Stitch' 'Gamestation' 'CeX' 'Etsy' 'Five Senses Art'
 'Sports Direct' 'Cass Art' 'Brilliant Brushes' 'Boots' 'Reebok'
 'JD Sports' 'Head' 'Frankie & Bennies' 'Gap Kids' 'North Face'
 'Town High' 'Collector Cave' 'Barbiee Boutique' 'RugbyFields'
 'Mamas & Papas' 'Lavender Primary' 'Remedy plus care' 'Specsavers'
 'Kew House' 'HMV' 'Vision Express' 'Millets' 'Pets Corner' 'Mothercare'
 'A Cut Above' 'Happy Days Home' 'Mountain Warehouse' 'Victoria Park'
 'University College Hospital'
'''
import pandas as pd
import os

# 商户类型映射字典
category_merchant_mapping = {
    'Food&Drink': {'Starbucks', 'Costa Coffee', 'The Crown', 'Kings Arms', 'Rose & Crown', 
                   'Frankie & Bennies', 'JustEat', 'Deliveroo', 'Coffee #1'},
    'Clothing': {'Fat Face', 'Topshop', 'Matalan', 'Selfridges', 'North Face', 'Reebok', 
                 'JD Sports', 'Gap Kids', 'Mothercare', 'Millets', 'Mountain Warehouse', 'Revella', 'Sports Direct', 'Wool'},
    'Grocery': {'Coop Local', 'Tesco', 'AMAZON', 'Amazon', 'Sainsbury', 'Sainsbury Local'},
    'Sports': {'PureGym', 'Loosely Fitted', 'Fitted Stitch', 'RugbyFields', 'Head'},
    'Entertainment': {'Blizzard', 'Xbox', 'Mojang Studios', 'Gamestation', 'CeX', 'HMV', 'Netflix', 'Disney'},
    'Art': {'Grand Union BJJ', 'SquareOnix', 'The Works', 'Blackwell\'s', 
            'Foyles', 'A Yarn Story', 'Five Senses Art', 'Cass Art', 'Brilliant Brushes', 
            'Collector Cave', 'Barbiee Boutique', 'Victoria Park'},
    'Beauty': {'Lloyds Pharmacy', 'Specsavers', 'Vision Express', 'A Cut Above', 'Remedy plus care', 'Boots'},
    'Housekeeping': {'Lavender Primary', 'Happy Days Home', 'Pets Corner', 'Hobby Lobby', 
                     'Craftastic', 'Hobbycraft', 'Etsy', 'Stitch By Stitch', 'Mamas & Papas'},
    'Bank': {'Halifax', 'LBG'},
    'Others': {'University College Hospital', 'Kew House', 'Town High'}
}

# 反向映射商户名称到类型
merchant_to_category = {merchant: category for category, merchants in category_merchant_mapping.items() for merchant in merchants}
# 定义函数用于映射商户类型
def map_merchant_type(row):
    if pd.isna(row['Third Party Name']) and pd.notna(row['Third Party Account No']):
        return 'Personal'
    return merchant_to_category.get(row['Third Party Name'], 'Unknown')

# 指定数据所在的文件夹路径
folder_path = 'simulated_data'
new_folder_path = 'processed_simulated_data'
# 处理文件夹中的每个文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 映射商户类型
        data['Merchant Type'] = data.apply(map_merchant_type, axis=1)
        
        # 保存处理后的数据到新文件
        new_file_path = os.path.join(new_folder_path, 'processed_' + filename)
        data.to_csv(new_file_path, index=False)
        print(f"Processed {filename} and saved as {new_file_path}")


