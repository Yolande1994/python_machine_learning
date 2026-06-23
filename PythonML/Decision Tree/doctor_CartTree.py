from CART_Tree import CARTTree
import pandas as pd

# 创建模型
tree = CARTTree()

# 读取训练数据  (高性能数值计算,用sklearn/XGBoost等库训练模型 用numpy更好;  纯Python实现的模型,需要灵活修改数据 用list更好)
#X = pd.read_csv('../tree/train_X.csv').to_numpy()
#y = pd.read_csv('../tree/train_y.csv').to_numpy().ravel()
X = pd.read_csv('../doctor/train_X.csv').values.tolist()
y = pd.read_csv('../doctor/train_y.csv').values.flatten().tolist()
train_data = [xi + [yi] for xi, yi in zip(X, y)]

# 特征名
feature_labels = ['气色', '气味', '脉象','体温']

# 训练模型
tree.fit(train_data, feature_labels)

# 读取测试数据
test_X = pd.read_csv('../doctor/test_X.csv').values.tolist()
test_y = pd.read_csv('../doctor/test_y.csv').values.flatten().tolist()

# 诊断（预测）
result = tree.predict(test_X)

# 打印输出诊断结果，与实际结果比较
print("\n编号,诊断值,实际值,结果")
labels = ['女娃', '男孩', '没有怀孕']
predictOK_Num = 0
i = 0
while i < len(test_y): # test_y是列表，没有shape属性，用len(test_y)
    if result[i] == test_y[i]:
        predictOK_Num += 1
        okOrNo = "准确"
    else:
        okOrNo = "错误"
    print(f"{i+1},{labels[int(result[i])]},{labels[int(test_y[i])]},{okOrNo}")
    i += 1

if i > 0:
    print(f"诊断准确率: {predictOK_Num / i * 100:.2f}%")
else:
    print("没有测试数据")