# 1. 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 解决matplotlib中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
from sklearn.datasets import load_iris  # 自带的鸢尾花数据集
from sklearn.model_selection import train_test_split  # 拆分训练/测试集
from sklearn.tree import DecisionTreeClassifier  # 决策树模型（简单易理解）
from sklearn.metrics import accuracy_score  # 评估准确率

# 2. 加载数据
iris = load_iris()
X = iris.data  # 特征：花的萼片/花瓣长度宽度
y = iris.target  # 标签：花的种类（0/1/2）

# 3. 拆分数据：80%训练，20%测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state固定拆分结果
)

# 4. 训练模型
model = DecisionTreeClassifier()  # 初始化模型
model.fit(X_train, y_train)  # 用训练集学习

# 5. 预测+评估
y_pred = model.predict(X_test)  # 用测试集预测
acc = accuracy_score(y_test, y_pred)  # 计算准确率
print(f"模型准确率：{acc * 100:.2f}%")  # 应该能到96%以上

# 6. 简单可视化（看特征分布）
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("萼片长度")
plt.ylabel("萼片宽度")
plt.title("鸢尾花特征分布")
plt.show()