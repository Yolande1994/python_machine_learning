# ===================== 第一步：导入所有库 =====================
# 数据清洗用的库
import pandas as pd
# 画图核心库
import matplotlib.pyplot as plt
# 更美观的画图库
import seaborn as sns
# 逻辑回归模型核心库
from sklearn.linear_model import LogisticRegression
# 拆分训练集/验证集（用来算准确率）
from sklearn.model_selection import train_test_split
# 计算准确率的指标
from sklearn.metrics import accuracy_score
import numpy as np

# ===================== 第二步：读取数据 =====================
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# ===================== 第三步：可视化分析（画图） =====================
# 【设置画图风格】解决中文显示问题
#sns.set_style("whitegrid")  # 设置画图背景风格，更美观
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 【创建画布】一张大图（2行2列）包含4张小图
plt.figure(figsize=(16, 12))  # 画布大小：宽16，高12

# --- 子图1：性别 vs 生还率（最核心！）---
plt.subplot(2, 2, 1)  # 2行2列，第1张图
# x='Sex'定横轴（性别分组），hue='Survived'定颜色（二级细分），data=train提供数据源，sns.countplot()底层自动完成:分组→计数→绘柱状图
sns.countplot(x='Sex', hue='Survived', data=train, palette='Set2')
plt.title('1. 性别与生还率关系', fontsize=14)
plt.xlabel('性别', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.legend(['遇难', '生还'], fontsize=12)  # 手动设置图例，更直观

# --- 子图2：船舱等级 vs 生还率 ---
plt.subplot(2, 2, 2)  # 2行2列，第2张图
sns.countplot(x='Pclass', hue='Survived', data=train, palette='Set2')
plt.title('2. 船舱等级与生还率关系', fontsize=14)
plt.xlabel('船舱等级 (1=头等, 2=二等, 3=三等)', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.legend(['遇难', '生还'], fontsize=12)

# --- 子图3：年龄分布（按生还/遇难分组）---
plt.subplot(2, 2, 3)  # 2行2列，第3张图
sns.histplot(data=train, x='Age', hue='Survived', multiple='stack', bins=20, palette='Set2')
plt.title('3. 年龄与生还率分布', fontsize=14)
plt.xlabel('年龄', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.legend(['遇难', '生还'], fontsize=12)

# --- 子图4：家庭人数 vs 生还率（先临时构造FamilySize特征）---
train_temp = train.copy()  # 临时复制一份，不影响原始数据
train_temp['FamilySize'] = train_temp['SibSp'] + train_temp['Parch'] + 1  # +1是把自己算上
plt.subplot(2, 2, 4)  # 2行2列，第4张图
sns.countplot(x='FamilySize', hue='Survived', data=train_temp, palette='Set2')
plt.title('4. 家庭人数与生还率关系', fontsize=14)
plt.xlabel('家庭总人数 (含自己)', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.legend(['遇难', '生还'], fontsize=12)

# 【保存+显示图片】
plt.tight_layout()  # 自动调整子图间距，避免重叠
#plt.savefig('titanic_eda.png', dpi=300)  # 保存为高清图片，dpi=300
print("\n===== 可视化分析完成！图片已保存为 titanic_eda.png =====")
#plt.show()  # 弹出窗口显示图片


# ===================== 第四步：数据清洗 =====================
# ========== 合并清洗 ==========
all_data = pd.concat([train, test], ignore_index=True) # ignore_index=True：忽略原行索引，合并后生成从0开始的全新连续索引，避免重复
# 填充缺失值
# 年龄：用中位数填充（中位数对异常值不敏感，比均值更稳定）
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
# 登船港口：用众数填充（分类特征用出现次数最多的值）
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
# 票价：用中位数填充
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

# 删除无用列
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
all_data = all_data.drop(drop_cols, axis=1)
# 文字转数字
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ========== 拆分回训练集/测试集 ==========
# 训练集（有Survived标签）
train_clean = all_data[:len(train)]
# 测试集（无Survived标签，后续可预测）
test_clean = all_data[len(train):]
# 删掉测试集里的Survived列（本来就是NaN）
test_clean = test_clean.drop('Survived', axis=1)


# ===================== 第五步：准备模型训练数据 =====================
# 特征X：所有列除了Survived（模型用来预测的依据）
X = train_clean.drop('Survived', axis=1)
# 标签y：只有Survived列（模型要预测的目标：0=死亡，1=生还）
y = train_clean['Survived']
# 拆分训练集和验证集（80%训练，20%验证，用来算准确率）
random_state=42 # 固定随机种子，保证每次运行代码时，拆分的数据集完全相同，方便复现和调试
# train_test_split()：把一份带标签的数据集（特征X + 标签y）随机拆分成「训练集」和「验证集/测试集」，用于模型训练和效果评估
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("训练集特征形状：", X_train.shape)  # 输出 (712, 7)：712行数据，7个特征
print("验证集特征形状：", X_val.shape)    # 输出 (179, 7)：179行数据，7个特征
X = np.hstack([np.ones((X_train.shape[0], 1)),X_train]) # 增加全1列
X_v = np.hstack([np.ones((X_val.shape[0], 1)), X_val])  # 增加全1列
print(X.shape, X_v.shape)

# ===================== 第六步：训练逻辑回归模型 =====================
# 手写逻辑回归模型
def sigmoid(z):
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))

def LR_fit(X, y, rate=0.01, max_iter=1000):
    m, n = X.shape
    w = np.zeros(n)
    for i in range(max_iter):
        z = X @ w
        y_p = sigmoid(z)
        gradient = (1/m) * X.T @ (y_p-y)
        w = w - rate * gradient
        # 可选：打印损失值监控训练
        if i % 100 == 0:
            loss = -np.mean(y * np.log(y_p + 1e-8) + (1 - y) * np.log(1 - y_p + 1e-8))
            print(f"迭代次数: {i}, Loss: {loss:.4f}")
    return w

# 手写预测函数
def predict(X, W):
    z = X @ W
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

'''
# 初始化逻辑回归模型（默认参数）
lr_model = LogisticRegression(random_state=42, max_iter=200)
# 用训练数据训练模型
lr_model.fit(X_train, y_train)
print("模型训练完成！")
'''

# ===================== 第七步：计算原始准确率 =====================
# 用手写
W = LR_fit(X, y_train, rate=0.01, max_iter=7000)
y_pred = predict(X_v, W)

# 用模型
#y_pred = lr_model.predict(X_val)
# 计算准确率（预测对的数量 / 总数量）
accuracy = accuracy_score(y_val, y_pred)
# 打印结果（保留4位小数）
print(f"\n===== 无特征工程的原始准确率：{accuracy:.4f} =====") # 正常结果应该在 0.78~0.80 之间（比如 0.7877）