import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the CSV file
file_path = 'cleaned_simulated_transaction_2024.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime and extract the month
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Month'] = data['Date'].dt.month

# Define the category to merchant mapping
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


# Filter the transactions for the merchants listed
all_merchants = set(merchant for sublist in category_merchant_mapping.values() for merchant in sublist)
filtered_data = data.loc[data['Third Party Name'].isin(all_merchants)].copy()

# Calculate the absolute value of the transactions for revenue
filtered_data.loc[:, 'Absolute Amount'] = filtered_data['Amount'].abs()

# Group by merchant and month to calculate monthly revenue
monthly_revenue = filtered_data.groupby(['Third Party Name', 'Month'])['Absolute Amount'].sum().reset_index()

# Pivot the data for easier plotting
pivot_table = monthly_revenue.pivot(index='Month', columns='Third Party Name', values='Absolute Amount').fillna(0)

# Plot individual graphs for each category
for category, merchants in category_merchant_mapping.items():
    plt.figure(figsize=(12, 6))  # 增加宽度，以适应图表内容
    # Use the 'tab20' colormap and generate as many colors as there are merchants
    color_palette = cm.get_cmap('tab20', len(merchants))
    for idx, merchant in enumerate(merchants):
        if merchant in pivot_table.columns:
            plt.plot(pivot_table.index, pivot_table[merchant], marker='o', label=merchant, color=color_palette(idx))
    # plt.title(f'Monthly Revenue for {category}')
    plt.xlabel('Month')
    plt.ylabel('Revenue (£)')
    plt.legend(title="Merchants", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 调整rect参数以减少图表右侧的空白区域
    plt.show()
