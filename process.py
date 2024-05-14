import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 引入numpy库来帮助调整刻度

# 加载数据
file_path = 'processed_simulated_data\processed_858989281.0_transactions.csv'
data = pd.read_csv(file_path)

# 转换日期格式并提取月份
data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
data['Month'] = data['Date'].dt.to_period('M')

# 按月份和商户类型分组，计算金额总和
grouped_data = data.groupby(['Month', 'Merchant Type'])['Amount'].sum().unstack(fill_value=0)

# 定义清晰的颜色调色板
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

# 绘制图表
fig, ax = plt.subplots(figsize=(9, 6))
grouped_data.plot(kind='line', ax=ax, marker='o', color=colors[:len(grouped_data.columns)])
# ax.set_title('Monthly Amount by Merchant Type')
ax.set_xlabel('Month')
ax.set_ylabel('Amount')
ax.grid(True)

# 设置纵轴刻度更密集
ax.yaxis.set_major_locator(plt.MaxNLocator(15))  # 增加刻度数量为15

plt.xticks(rotation=45)
plt.legend(title='Merchant Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
