from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 准备数据集
data = {
    'Monopoly Money Amount': [
        2.40, 2.65, 1.45, 2.45, 2.15, 2.25, 2.55, 1.95, 1.80, 2.20,
        4.60, 4.80, 4.35, 5.05, 4.85, 4.20, 4.65, 4.45, 4.10, 4.40,
        5.10, 4.90, 3.85, 5.30, 4.70, 4.00, 4.95, 4.55, 5.20, 4.50,
        3.60, 3.90, 4.25, 4.15, 3.65, 4.05, 3.95, 4.75, 3.75, 5.00,
        3.70, 3.40, 3.25, 4.30, 2.90
    ],
    'Frequency': [
        73057, 72925, 36881, 36819, 36812, 36756, 36628, 36430, 36380, 36176,
        2452, 2409, 2001, 1923, 1917, 1497, 1482, 1451, 1402, 1247,
        1211, 1166, 1043, 1027, 1025, 1014, 1014, 1014, 974, 775,
        762, 699, 511, 507, 499, 497, 496, 495, 480, 479,
        468, 449, 445, 239, 225
    ]
}
df = pd.DataFrame(data)

# 应用KMeans聚类
sse = {}
silhouette_coefficients = []
K = range(2, 11)  # 尝试的聚类数范围
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    sse[k] = kmeans.inertia_  # SSE到簇中心的距离
    score = silhouette_score(df, kmeans.labels_)
    silhouette_coefficients.append(score)

# 绘制肘部法则图
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# 绘制轮廓系数图
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_coefficients, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient Method For Optimal k')
plt.show()
