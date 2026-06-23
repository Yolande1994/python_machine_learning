# 导入所需库
import numpy as np
import matplotlib.pyplot as plt  # 绘图神器
# ========== 设置matplotlib支持中文 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文（Windows系统）
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from sklearn.datasets import make_regression  # 生成模拟回归数据
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 步骤1：生成模拟数据（1个特征，方便可视化）
# X：特征（比如房屋面积），y：目标值（比如房价）
X, y = make_regression(
    n_samples=100,  # 100个数据点
    n_features=1,   # 1个特征
    noise=20,       # 加入少量噪声，让数据更接近真实场景
    random_state=42 # 固定随机种子，结果可复现
)

# 步骤2：划分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 步骤3：创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)  # 核心：拟合数据，找到最优w和b

# 步骤4：查看模型参数（w和b）
print(f"权重(w)：{model.coef_[0]:.2f}")
print(f"偏置(b)：{model.intercept_:.2f}")

# 步骤5：用模型预测测试集
y_pred = model.predict(X_test)

# 步骤6：评估模型性能
mse = mean_squared_error(y_test, y_pred)  # 均方误差（越小越好）
r2 = r2_score(y_test, y_pred)            # 决定系数（越接近1越好）
print(f"均方误差(MSE)：{mse:.2f}")
print(f"决定系数(R²)：{r2:.2f}")

# 步骤7：可视化结果（直观看拟合效果）
plt.scatter(X_test, y_test, color='blue', label='真实值')  # 测试集真实数据点
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测直线')  # 模型拟合的直线
plt.xlabel('特征X（比如房屋面积）')
plt.ylabel('目标y（比如房价）')
plt.title('线性回归拟合效果')
plt.legend()
plt.show()