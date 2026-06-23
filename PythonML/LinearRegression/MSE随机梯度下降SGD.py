# 随机梯度下降版（SGD）
x = [1, 2, 3]
y = [2, 4, 6]

w = 0.0
alpha = 0.1
max_epoch = 3  # 迭代轮数（每轮遍历所有样本）

print("===== 随机梯度下降迭代过程 =====")
print(f"初始权重 w = {w:.2f}")

for epoch in range(max_epoch):
    print(f"第{epoch+1}轮（遍历所有样本）：")
    for i in range(len(x)):  # 每遍历1个样本，w就变1次 (SGD：“边走边更”—— 每看一个样本，就根据这个样本的反馈调整方向)
        xi = x[i]
        yi = y[i]
        yp = w * xi  # 计算单个样本的预测值
        # 梯度公式: 2 * x * (yp - y)  只用单个样本的梯度，不用平均   (误差是单样本损失)
        gradient = 2 * xi * (yp - yi)
        # 更新权重 w = w - alpha * 梯度(单样本)
        w = w - alpha * gradient
        # 当前误差
        error = (yp - yi) ** 2
        print(f"  样本{i+1}：梯度={gradient:.4f}，w更新为{w:.4f}，误差={error:.4f}")

print("===== SGD最终结果 =====")
print(f"w ≈ {w:.4f}")


#===================== 核心：两种损失函数的区别 =====================
# 场景       损失函数         损失公式（单样本）                   梯度公式（单样本，对w）
# 线性回归    平方损失（单样本） L=       (w⋅x−y)²                  2⋅x⋅(yp − y)
# 线性回归    均方误差（MSE）   L= (1/N)Σ(w⋅x−y)²              (2/N)⋅x⋅(yp − y)
# 逻辑回归    对数损失         L=−[y⋅log(y^)+(1−y)⋅log(1−y^)]       x⋅(yp − y)