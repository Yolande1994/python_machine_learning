from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 加载经典分类数据集（鸢尾花，3个类别，4个特征）
iris = load_iris()
X = iris.data  # 特征数据（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
y = iris.target  # 分类标签（0/1/2 对应3种鸢尾花）

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 初始化并训练分类决策树
dt_clf = DecisionTreeClassifier(
    criterion='gini',  # 用基尼系数衡量纯净度
    max_depth=3,  # 限制树的最大深度，防止过拟合
    random_state=42
)
dt_clf.fit(X_train, y_train)

# 4. 可视化这颗决策树（直观看到“提问”和“结论”）
plt.figure(figsize=(12, 8))  # figure创建一个空白的绘图窗口  figsize=(12, 8)：设置画布的尺寸
plot_tree(dt_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True,rounded=True)
plt.show()

# 5. 简单评估模型效果
accuracy = dt_clf.score(X_test, y_test)
print(f"模型在测试集上的准确率：{accuracy:.2f}")