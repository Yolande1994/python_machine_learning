import pandas as pd

# 读取训练集（有标签Survived，用来训练模型）
train = pd.read_csv("../data/train.csv")
# 读取测试集（没有标签，后面用来预测）
test = pd.read_csv("../data/test.csv")
#print(train.head())  # print("训练集前5行：") 先看一眼训练集前5行，确认读对了

# 查看每列有多少缺失 + 数据类型
print("\n数据信息（重点看缺失值）：")
print(train.info())
print(test.info())
# 查看数值列统计（看Age、Fare大概范围）
print("\n数值列统计：")
print(train.describe())
'''
info()     是数据结构检查：帮你定位 “哪里有问题”（缺失值、非数值列）；
describe() 是数值分布分析：帮你确定 “怎么解决问题”（用中位数填充、是否处理极端值）；
这两个函数是数据分析的 “黄金组合”，任何数据集拿到手，第一步都要先跑这两个函数，再做后续清洗 / 建模。
'''

#=================== 数据清洗（核心步骤！逐个解决） ===================
# 先把训练集和测试集合起来，一起清洗
all_data = pd.concat([train, test], ignore_index=True)
print("合并后数据形状：", all_data.shape)

# 处理Age缺失（用中位数填充，最稳）
age_median = all_data['Age'].median()  # 计算Age的中位数
# 用中位数填充缺失
all_data['Age'] = all_data['Age'].fillna(age_median)
print(f"用中位数 {age_median} 填充Age缺失")

# 处理Embarked缺失（用众数填充，出现最多的港口）
embarked_mode = all_data['Embarked'].mode()[0]  # 找出出现最多的登船港口
# 填充
all_data['Embarked'] = all_data['Embarked'].fillna(embarked_mode)
print(f"用众数 {embarked_mode} 填充Embarked缺失")

# 处理Fare缺失（测试集里有1个，一起洗了）
fare_median = all_data['Fare'].median()  # Fare也用中位数填充
all_data['Fare'] = all_data['Fare'].fillna(fare_median)

# 删除没用的列（对预测没帮助，直接删）
# 删掉：乘客ID、姓名、船票号、船舱号（缺失太多）
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
all_data = all_data.drop(drop_cols, axis=1)
print("\n删除无用列后，剩下的特征：")
print(all_data.columns.tolist())

# 把文字特征 → 转成数字（模型必须！）
# 性别 Sex：male→0，female→1
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1}) # map 映射：字典里写好对应关系
# 登船港口 Embarked：S→0, C→1, Q→2
all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 把数据再分开回: 训练集 / 测试集
# 训练集：原来train的行数（891行）
train_clean = all_data[:len(train)]
# 测试集：剩下的
test_clean = all_data[len(train):]
print("\n清洗后训练集形状：", train_clean.shape)
print("清洗后测试集形状：", test_clean.shape)

# 最后检查
print("\n最终清洗后训练集信息（无缺失、全是数字）：")
print(train_clean.info())
print("\n清洗后训练集前5行：")
print(train_clean.head())