import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 数据加载和预处理
data = pd.read_csv('simulated_data/113747882.0_transactions.csv')  # 替换为您的文件路径
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
amount_data = data['Amount'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_amount = scaler.fit_transform(amount_data)

# 创建时间序列数据集的函数
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# 准备数据集
look_back = 3
X, Y = create_dataset(scaled_amount, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# 模型评估和预测
# ... (根据需要添加代码)


# 模型评估：计算测试误差
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)  # 反缩放到原始金额
Y_test_orig = scaler.inverse_transform([Y_test])  # 反缩放原始测试标签

# 计算均方误差
test_score = np.sqrt(mean_squared_error(Y_test_orig[0], test_predictions[:, 0]))
print('Test RMSE: %.2f' % (test_score))

# 可视化预测结果和实际值
plt.figure(figsize=(12,6))
plt.plot(Y_test_orig[0], label='Actual Amount')
plt.plot(test_predictions[:, 0], label='Predicted Amount')
plt.title('LSTM Model Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend()
plt.show()

# 使用模型进行未来的交易预测（假设你有未来的输入数据）
# future_predictions = model.predict(future_input_data)
# future_predictions = scaler.inverse_transform(future_predictions)
# ... 进行未来交易预测
