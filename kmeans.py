from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File paths of the uploaded CSV files
folder_path = 'RFM_Score'
all_files = os.listdir(folder_path)

# Initialize lists to hold the data points and their names
data_points = []
point_names = []

# Iterate over each file and extract the data points and names
for file in all_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    score_row = df[df['from_totally_fake_account'] == 'Score']
    if not score_row.empty:
        data_points.append(score_row[['Recency', 'Frequency', 'Monetary']].values[0])
        point_name = file.split('_rfm_results.csv')[0]
        point_names.append(point_name)

# Convert the list of data points into a numpy array
data_points_array = np.array(data_points)

# 在数据点上添加轻微的随机噪声 (jittering)
jitter_strength = 0.01  # 这个值可以根据需要调整
jittered_data_points = data_points_array + np.random.normal(0, jitter_strength, data_points_array.shape)

# 假设我们决定将数据点分成3个聚类
kmeans = KMeans(n_clusters=5)

# 执行K-Means聚类
kmeans.fit(data_points_array)

# 获取聚类标签
kmeans_labels = kmeans.labels_

# 绘制带有K-Means聚类标签的点云图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.rainbow(np.linspace(0, 1, len(set(kmeans_labels))))

for point, label, point_name in zip(data_points_array, kmeans_labels, point_names):
    ax.scatter(point[0], point[1], point[2], color=colors[label])
    # ax.text(point[0], point[1], point[2], point_name)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('3D Scatter Plot with K-Means Clusters')
plt.show()
