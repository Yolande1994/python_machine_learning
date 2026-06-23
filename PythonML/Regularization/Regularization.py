import numpy as np
import matplotlib.pyplot as plt

# 1. 造带噪声的模拟数据（故意制造过拟合风险）
np.random.seed()
x = np.linspace(0, 5, 15)  # 少量数据更容易过拟合
y = 2 * x + 1 + np.random.randn(15) * 1.2  # 加大噪声


# 2. 定义梯度下降函数（核心复用）
def gradient_descent(use_regularization, alpha=0.1):
    w = 0.0
    b = 0.0
    lr = 0.01
    epochs = 1500

    for _ in range(epochs):
        y_pred = w * x + b
        # 损失计算：有无正则二选一
        if use_regularization:
            loss = np.mean((y - y_pred) ** 2) + alpha * (w ** 2)
        else:
            loss = np.mean((y - y_pred) ** 2)

        # 梯度计算：有无正则二选一
        if use_regularization:
            dw = -2 * np.mean(x * (y - y_pred)) + 2 * alpha * w
        else:
            dw = -2 * np.mean(x * (y - y_pred))
        db = -2 * np.mean(y - y_pred)

        # 更新参数
        w -= lr * dw
        b -= lr * db
    return w, b


# 3. 分别训练：无正则 vs 有L2正则
w_no_reg, b_no_reg = gradient_descent(use_regularization=False)
w_with_reg, b_with_reg = gradient_descent(use_regularization=True, alpha=0.3)

# 4. 画图对比
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='red', label='原始数据', s=50)
# 无正则拟合线
plt.plot(x, w_no_reg * x + b_no_reg, 'g-', linewidth=2, label=f'无正则 (w={w_no_reg:.2f})')
# 有正则拟合线
plt.plot(x, w_with_reg * x + b_with_reg, 'b-', linewidth=2, label=f'有L2正则 (w={w_with_reg:.2f})')

plt.title('无正则 vs L2正则 权重对比')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 输出关键对比
print('===== 权重对比 =====')
print(f'无正则 w = {w_no_reg:.3f}')
print(f'有L2正则 w = {w_with_reg:.3f}')