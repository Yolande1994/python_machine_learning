"""
数据预处理+特征工程模块
功能：清洗脏数据、构造有效特征，输出模型可直接使用的数据集
学习重点：工业界数据处理规范、特征工程的核心思路、避免数据泄露
"""
import pandas as pd

def preprocess_data(train_path: str = "data/train.csv", test_path: str = "data/test.csv"):
    """  完整的数据清洗 + 特征工程流程
    :param train_path: 训练集路径，标注为字符串类型，默认 "data/train.csv"
    :param test_path:  测试集路径，标注为字符串类型，默认 "data/test.csv"
    :return: 训练集特征、训练集标签、测试集特征
    """
    print("="*30 + " 开始数据预处理+特征工程 " + "="*30)
    # 1. 读取原始数据
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    # 【工业界规范】合并训练集+测试集一起处理，避免数据泄露(分开处理:测试集的统计值（如中位数）会泄露到训练过程，导致泛化能力下降. 只用全局信息:模型学到的是真本事，泛化能力才强)
    all_data = pd.concat([train_raw, test_raw], ignore_index=True)
    print(f"合并后总数据量：{all_data.shape[0]}行，{all_data.shape[1]}列")

    # ===================== 第一部分：基础数据清洗 =====================
    # 1. 填充缺失值（用全量数据的统计值，避免泄露）
    # 年龄：用中位数填充（中位数对异常值不敏感，比均值更稳定）
    all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
    # 登船港口：用众数填充（分类特征用出现次数最多的值）
    all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
    # 票价：用中位数填充
    all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
    # 2. 分类特征编码（文字转数字，模型只能识别数字）
    all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
    all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # ===================== 第二部分：核心特征工程（提分关键） =====================
    # 特征1：家庭总人数 FamilySize (灵感来自EDA：2-4人的小家庭生还率最高，独自出行/大家庭生还率低)
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

    # 特征2：是否独自出行 IsAlone (灵感来自EDA：独自出行的人死亡率极高，单独拎出来让模型更容易学习)
    all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)  # astype(int):True/False → 0/1

    # 特征3：年龄分箱 AgeBin (灵感来自EDA：儿童（<12岁）生还率极高，老人生还率低；分箱可以消除年龄异常值的影响)
    all_data['AgeBin'] = pd.cut(
        all_data['Age'],
        bins=[0, 12, 18, 35, 60, 100],  # 按年龄划分区间：儿童、少年、青年、中年、老年
        labels=[0, 1, 2, 3, 4]          # 给每个区间编码
    ).astype(int)  # 'AgeBin'这一列的值就是每个乘客对应的年龄区间编码

    # 特征4：乘客头衔 Title (核心强特征！头衔代表了社会地位、性别、年龄，是泰坦尼克号提分的关键)
    # 从Name字段提取头衔（例如 Braund, Mr. Owen Harris → 提取出 Mr）
    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False) # ' ([A-Za-z]+)\.': 匹配 “一个空格+一串字母+一个句点”，只提取中间的“一串字母”  +:匹配1次
    # 把罕见头衔合并成Other，避免维度爆炸，同时让模型更容易学习到规律，而不是被少数特例干扰
    rare_title = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    all_data['Title'] = all_data['Title'].replace(rare_title, 'Other')
    # 同义头衔合并
    all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
    all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
    all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
    # 头衔编码（文字转数字）
    all_data['Title'] = all_data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}).fillna(0)

    # 特征5：票价分箱 FareBin (灵感来自EDA：票价越高生还率越高，分箱消除票价极端值的影响)
    all_data['FareBin'] = pd.qcut(  # pd.cut():按数值平均分  pd.qcut():按样本数量平均分
        all_data['Fare'],
        q=4,  # 按四分位数分箱，保证每个区间人数一致
        labels=[0, 1, 2, 3]
    ).astype(int)  # 'FareBin'这一列的值就是每个乘客对应的票价区间编码

    # ===================== 第三部分：最终数据处理 =====================
    # 1. 删除无用列（对预测无帮助、已经提取过特征的列）
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    all_data = all_data.drop(drop_cols, axis=1)

    # 2. 拆分回训练集和测试集
    train_data = all_data[:len(train_raw)]  # 前891行是训练集（有标签）
    test_data = all_data[len(train_raw):]   # 后面的是测试集（无标签）

    # 3. 拆分特征和标签
    train_features = train_data.drop('Survived', axis=1)   # 模型输入：所有特征
    train_label = train_data['Survived']                   # 模型预测：是否生还
    test_features = test_data.drop('Survived', axis=1)     # 最终提交用的测试集特征

    print(f"最终训练集特征维度：{train_features.shape}")
    print(f"最终测试集特征维度：{test_features.shape}")
    print(f"最终使用的特征列表：{list(train_features.columns)}")
    print("数据预处理+特征工程完成！")
    print("="*70 + "\n")

    return train_features, train_label, test_features

# 单独运行这个文件时，会执行预处理
if __name__ == "__main__":
    preprocess_data()