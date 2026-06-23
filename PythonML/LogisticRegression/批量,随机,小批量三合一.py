import numpy as np
# 特征（X）和标签（y=0/1）
X = np.array([[1], [2], [3], [4], [5]])  # 特征（需注意：逻辑回归通常加截距项，这里简化）
y = np.array([0, 0, 0, 1, 1])            # 二分类标签
n = len(X)
# sigmoid函数
def sigmoid(z):
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))


print("===== 批量梯度下降（BGD） =====")
w, b = 0.0, 0.0  # 初始化参数
rate = 0.1
epochs = 1000

for i in range(epochs):
    # 计算线性得分和预测概率
    z = w * X + b
    y_hat = sigmoid(z)
    # 计算对数损失（仅监控，不参与梯度更新）
    loss = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    # 批量梯度计算（核心：所有样本的平均梯度）
    dw = (1/n) * np.sum((y_hat - y) * X)
    db = (1/n) * np.sum(y_hat - y)
    # 更新参数
    w -= rate * dw
    b -= rate * db
    if (i+1) % 200 == 0:
        print(f'第{i+1}次迭代,w={w:.4f},b={b:.4f},损失={loss:.6f}')


print("\n===== 随机梯度下降（SGD） =====")
w, b = 0.0, 0.0
rate = 0.01  # SGD学习率更小，避免震荡
epochs = 1000

for i in range(epochs):
    # 随机选1个样本
    idx = np.random.randint(0, n)
    xi = X[idx]
    yi = y[idx]
    # 单样本计算
    z_i = w * xi + b
    y_hat_i = sigmoid(z_i)
    # 单样本梯度（无求和、无除以n）
    dw = (y_hat_i - yi) * xi
    db = (y_hat_i - yi)
    # 更新参数
    w -= rate * dw
    b -= rate * db
    # 计算全局损失（仅监控）
    z_all = w * X + b
    y_hat_all = sigmoid(z_all)
    loss = -np.mean(y * np.log(y_hat_all) + (1-y) * np.log(1-y_hat_all))
    if (i+1) % 200 == 0:
        print(f'第{i+1}次迭代,w={w:.4f},b={b:.4f},损失={loss:.6f}')


print("\n===== 小批量梯度下降（MBGD）（工业界主流） =====")
w, b = 0.0, 0.0
rate = 0.05
epochs = 1000
batch_size = 2  # 每次选2个样本

for i in range(epochs):
    # 随机选一批样本
    idx = np.random.choice(n, batch_size, replace=False)
    X_batch = X[idx]
    y_batch = y[idx]
    # 批量计算
    z_batch = w * X_batch + b
    y_hat_batch = sigmoid(z_batch)
    # 小批量梯度
    dw = (1/batch_size) * np.sum((y_hat_batch - y_batch) * X_batch)
    db = (1/batch_size) * np.sum(y_hat_batch - y_batch)
    # 更新参数
    w -= rate * dw
    b -= rate * db
    # 全局损失监控
    z_all = w * X + b
    y_hat_all = sigmoid(z_all)
    loss = -np.mean(y * np.log(y_hat_all) + (1-y) * np.log(1-y_hat_all))
    if (i+1) % 200 == 0:
        print(f'第{i+1}次迭代,w={w:.4f},b={b:.4f},损失={loss:.6f}')