import numpy as np

# ===================== 线性回归（MSE损失）=====================
# 准备数据（必须转为numpy数组，而非普通列表）
X = np.array([1,2,3])
y = np.array([3,5,7])
n = len(X)
# 初始化参数
w = 0
b = 0
rate = 0.1
epochs  = 500

print("===== 线性回归（MSE损失） =====")
for i in range(epochs):
    # 计算预测值
    yp = w * X + b
    # 计算均方误差MSE
    MSE = np.mean((yp - y) ** 2)  # 逻辑回归用对数损失(交叉熵/逻辑损失)
    # 计算梯度（对w和b的偏导数）（固定其他参数，只看当前参数w/b变化对损失的影响）
    dw = (2/n) * np.sum((yp - y) * X)  # w的梯度(error对w的偏导数): 2/n * ∑error*X    单样本: 2*error*X
    db = (2/n) * np.sum(yp - y)        # b的梯度(error对b的偏导数): 2/n * ∑error      单样本: 2*error
    # 更新参数（核心：沿梯度反方向更新）
    w = w - rate * dw
    b = b - rate * db
    if (i + 1) % 50 == 0: # 每50次打印一次
        print(f'第{i + 1}次迭代,w={w:.4f},b={b:.4f},MSE={MSE:.6f}')


# ===================== 逻辑回归（对数损失）=====================
# 分类任务数据（y是0/1标签，和回归数据区分）
X_l = np.array([1,2,3])
y_l = np.array([0,0,1])  # 修正：逻辑回归必须用0/1标签
n_l = len(X_l)
w_l = 0
b_l = 0
rate_l = 0.5  # 逻辑回归学习率需调大，否则收敛慢
epochs_l = 1000  # 增加迭代次数，确保收敛

def sigmoid(z):
    z = np.clip(z, -100, 100) # 防止exp溢出，限制z的范围
    return 1/(1+np.exp(-z))
print("\n===== 逻辑回归（对数损失） =====")
for i in range(epochs_l):
    # 计算预测值
    yp = w_l * X_l + b_l # 计算线性得分
    yp_l = sigmoid(yp)   # 转成0~1概率
    # 对数损失（不是MSE）
    loss = -np.mean(y_l * np.log(yp_l) + (1-y_l) * np.log(1-yp_l))
    # 对数损失的梯度公式（非MSE,无2倍系数）
    dw = (1/n_l) * np.sum((yp_l - y_l) * X_l)  # w的梯度(error对w的偏导数): 1/n * ∑error*X    单样本: error*X
    db = (1/n_l) * np.sum(yp_l - y_l)          # b的梯度(error对b的偏导数): 1/n * ∑error      单样本: error
    # 更新参数（核心：沿梯度反方向更新）
    w_l = w_l - rate_l * dw
    b_l = b_l - rate_l * db
    if (i + 1) % 100 == 0: # 每100次打印一次
        print(f'第{i + 1}次迭代,w={w_l:.4f},b={b_l:.4f},对数损失={loss:.6f}')


print("\n===== 闭式解（仅线性回归） =====")
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x_squared = np.sum(X ** 2)
# 权重w
w_ols = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
# 截距b
b_ols = (sum_y - w_ols * sum_x) / n
print(f"最优权重闭式解 w = {w_ols:.2f}")
print(f"最优截距闭式解 b = {b_ols:.2f}")


# ===================== 最终结果对比 =====================
print("\n===== 最终结果对比 =====")
print(f"线性回归（MSE）：w={w:.2f}, b={b:.2f}")
print(f"逻辑回归（对数损失）：w={w_l:.2f}, b={b_l:.2f}")
print(f"线性回归闭式解：w={w_ols:.2f}, b={b_ols:.2f}")