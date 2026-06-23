# 准备数据
x = [1, 2, 3]  # 特征值
y = [2, 4, 6]  # 真实值（最优w=2）

w = 0.0  # 初始权重（从0开始）
alpha = 0.1  # 学习率（步长）
max_iter = 3  # 迭代次数（多迭代几次看收敛效果）
n = len(x)  # 样本数量

print("===== 批量梯度下降迭代过程 =====")
print(f"初始权重 w = {w:.2f}")


for i in range(max_iter):  # 遍历完所有样本，w才变1次 (BGD：“看完所有再更”—— 把所有样本的反馈汇总、求平均，再调整一次方向)
    # 1：计算所有样本的预测值 yp = w*x
    yp = [w * xi for xi in x]
    #print(yp)
    # 2：计算每个样本的梯度（误差对w的导数）
    # 梯度公式: 2 * x * (w*x - y) = 2 * x * (yp - y)   (误差是均方误差MSE)
    gradients = [2 * xi * (ypi - yi) for xi, ypi, yi in zip(x, yp, y)]
    #print(gradients)
    # 3：计算平均梯度（批量梯度下降的核心：用所有样本的平均梯度）
    avg_gradient = sum(gradients) / n
    # 4：更新权重 w = w - alpha * 平均梯度
    w = w - alpha * avg_gradient
    # 5：计算当前总误差（平方误差和），看误差是否越来越小
    total_error = sum([(yp_i - yi) ** 2 for yp_i, yi in zip(yp, y)])
    # 打印每一轮的结果（和手动演算对比）
    print(f"第{i + 1}轮迭代：")
    print(f"  平均梯度 = {avg_gradient:.4f}")
    print(f"  更新后 w = {w:.4f}")
    print(f"  当前总误差 = {total_error:.4f}\n")

# 4. 输出最终结果
print("===== 最终结果 =====")
print(f"梯度下降找到的最优权重 w ≈ {w:.4f}")
print(f"真实最优权重 w = 2.0")

#===================== 核心：两种损失函数的区别 =====================
# 场景       损失函数         损失公式（单样本）                   梯度公式（单样本，对w）
# 线性回归    平方损失（单样本） L=       (w⋅x−y)²                  2⋅x⋅(yp − y)
# 线性回归    均方误差（MSE）   L= (1/N)Σ(w⋅x−y)²              (2/N)⋅x⋅(yp − y)
# 逻辑回归    对数损失         L=−[y⋅log(y^)+(1−y)⋅log(1−y^)]       x⋅(yp − y)