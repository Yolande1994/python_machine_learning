from GaussianNB import GaussianNB_Engineering_demo
from my_write import GNB
import pandas as pd

# 创建模型
doctor = GNB()

# 读取训练数据
X = pd.read_csv('../doctor/train_X.csv').to_numpy()
y = pd.read_csv('../doctor/train_y.csv').to_numpy().ravel()

# 训练模型
doctor.fit(X, y)

# 读取测试数据
test_X = pd.read_csv('../doctor/test_X.csv').to_numpy()
test_y = pd.read_csv('../doctor/test_y.csv').to_numpy().ravel()

# 诊断（预测）
result = doctor.predict2(test_X)

# 打印输出诊断结果，与实际结果比较
labels = ['女娃', '男孩', '没有怀孕']
predictOKNum = 0
i = 0
print("\n编号,诊断值,实际值,结果")

while i < test_y.shape[0]:
    if result[i] == test_y[i]:
        predictOKNum += 1
        okOrNo = "准确"
    else:
        okOrNo = "错误"
    print(f"{i+1},{labels[int(result[i])]},{labels[int(test_y[i])]},{okOrNo}")
    i += 1

if i > 0:
    print(f"诊断准确率: {predictOKNum / i * 100:.2f}%")
else:
    print("没有测试数据")