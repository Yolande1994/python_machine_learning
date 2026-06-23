from Tree_Kmeans import ContinuousDiscretizer,mytree
import pandas as pd

# 读取训练数据  (高性能数值计算,用sklearn/XGBoost等库训练模型 用numpy更好;  纯Python实现的模型,需要灵活修改数据 用list更好)
X = pd.read_csv('../doctor/train_X.csv').values.tolist()
y = pd.read_csv('../doctor/train_y.csv').values.flatten().tolist()
# 合并
train_data = [xi + [yi] for xi, yi in zip(X, y)]
# 特征名
feature_labels = ['气色', '气味', '脉象','体温']
# 连续特征的索引
continuous_indexes = [0, 1, 2, 3]

# 创建树模型
tree = mytree()
# 创建离散器
discretizer = ContinuousDiscretizer(n_clusters=3, random_state=42)  # K<3 信息不足导致欠拟合错判，K>3 过度细分引入噪声导致过拟合错判。
# 将连续数据转为离散数据
train_data_discrete = discretizer.fit_transform(train_data, continuous_indexes)
# 训练树模型
trained_tree = tree.fit(train_data_discrete, feature_labels)

# 读取测试数据
test_X = pd.read_csv('../doctor/test_X.csv').values.tolist()
test_y = pd.read_csv('../doctor/test_y.csv').values.flatten().tolist()
# 将测试数据转为离散数据
new_sample_discrete = discretizer.transform(test_X, continuous_indexes)

# 诊断（预测）
result = tree.predict(new_sample_discrete)

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