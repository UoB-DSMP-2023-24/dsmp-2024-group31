import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping


# 数据加载和聚合
data = pd.read_csv('data/TOY_SHOP_transactions.csv')  # 你的文件路径
data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'], format='%d/%m/%Y')
daily_totals = data.groupby('not_happened_yet_date')['monopoly_money_amount'].sum().reset_index()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_totals['monopoly_money_amount'].values.reshape(-1, 1))

# 创建时间序列数据集的函数
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# 准备数据集
look_back = 20  # 基于你的需要调整这个参数
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
early_stop = EarlyStopping(monitor='loss', patience=10)
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, Y_train, epochs=30, batch_size=1, verbose=2, callbacks = [early_stop])

# 模型评估和预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化预测结果
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# 计算均方根误差
train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))

# 初始化预测绘图数组
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan

# 填充训练集预测
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

# 填充测试集预测
# 修复可能的错误：确保测试预测的插入点是正确的
test_predict_index_start = len(train_predict) + (look_back)
test_predict_index_end = test_predict_index_start + len(test_predict)
test_predict_plot[test_predict_index_start:test_predict_index_end, :] = test_predict

plt.figure(figsize=(15,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original data')
plt.plot(train_predict_plot, label='Training set prediction')
plt.plot(test_predict_plot, label='Test set prediction')
plt.title('TOY_SHOP LSTM Predictions vs Actual')
plt.xlabel('Days')
plt.ylabel('Monopoly Money Amount')
plt.legend()
plt.show()
