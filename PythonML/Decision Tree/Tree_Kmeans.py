from math import log2
import numpy as np
import pandas as pd
"""
K-means 核心逻辑:
1. 随机选K个初始聚类中心
2. 算距离:计算每个点到各中心的距离,各点归属于距离最近的簇
3. 挪位置:各簇以簇内点的平均值,刷新簇中心
4. 重复2-3直到收敛
"""
# =========================== 一. 手写 K-means  =================================
class SimpleKMeans:  # 仅支持单特征聚类，适配连续特征离散化场景
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter      # 最大迭代次数（防止死循环）
        self.random_state = random_state  # 固定随机种子（无论运行多少次，初始中心和聚类标签都和第一次完全一样，结果可复现）
        self.centers = None  # 最终聚类中心
        self.labels = None   # 每个点的聚类标签（所属簇的索引）

    def _euclidean_distance(self, x1, x2):    # 计算两个数的欧氏距离（仅单特征下,欧氏距离=绝对值差）
        return abs(x1 - x2)

    def fit(self, X):  # 仅支持单特征一维数组
        # 1. 初始化：设置随机种子，随机选K个初始中心
        np.random.seed(self.random_state)
        # 从输入的样本数据 X 中，随机挑选 self.n_clusters 个不重复的样本，作为 K-means 的初始聚类中心
        self.centers = np.random.choice(X, size=self.n_clusters, replace=False)
        for _ in range(self.max_iter):
            # 2. 分配标签：每个点归到最近的中心
            labels = []
            for x in X:
                # 计算当前点到所有中心的距离
                distances = [self._euclidean_distance(x, c) for c in self.centers]
                # 选距离最小的中心的索引作为标签
                labels.append(np.argmin(distances))  # np.argmin()返回索引  np.min()返回值
            self.labels = np.array(labels)
            # 3. 更新中心：每个簇的平均值作为新中心
            new_centers = []
            for i in range(self.n_clusters):
                # 取当前簇的所有点
                cluster_points = X[self.labels == i]
                if len(cluster_points) == 0:  # 防止空簇（随机重选一个点）
                    cluster_points = np.random.choice(X, size=1)
                new_centers.append(np.mean(cluster_points))  # 计算当前簇的所有样本的平均值（新的聚类中心），添加到new_centers列表中
            new_centers = np.array(new_centers)
            # 4. 收敛判断：中心不再变化则停止
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers

    # 预测新样本的聚类标签(输入新样本,返回所属类标签)
    def predict(self, X):
        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, c) for c in self.centers]
            labels.append(np.argmin(distances))
        return np.array(labels)


# ===================== 二. 连续特征离散化工具类（基于手写K-means） =====================
class ContinuousDiscretizer:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters  # 每个特征聚类数
        self.max_iter = max_iter      # K-means最大迭代
        self.random_state = random_state
        self.kmeans_models = {}      # {特征索引: 手写K-means模型}

    # 训练 + 转换: 对连续特征离散化  (输入原始数据与连续特征索引,返回离散化后数据)
    def fit_transform(self, data, feature_indexes):
        # 转数组方便处理（保留原格式）
        data = np.array(data, dtype=object)  # object:混合类型
        # 遍历每个连续特征，单独聚类
        for idx in feature_indexes:
            # 提取该特征的所有值（转成一维数组）              [: →取所有行, idx →取第 idx 列]
            feat_values = data[:, idx].astype(float) # [:, idx] →取第idx列的所有值，得到一个一维数组
            # 初始化并训练手写K-means
            kmeans = SimpleKMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=self.random_state)
            kmeans.fit(feat_values)
            # 保存模型（用于预测新样本）
            self.kmeans_models[idx] = kmeans
            # 替换原连续值为聚类标签（离散化）
            data[:, idx] = kmeans.labels  # .labels: 访问实例的self.labels类属性
        # 转回list，适配决策树
        return data.tolist()

    # 用训练好的K-means,对新样本离散化
    def transform(self, new_samples, feature_indexes):
        # 统一处理成二维数组(兼容单/多样本)
        if isinstance(new_samples[0], (int, float)):
            new_samples = [new_samples]  # 把「单个一维样本」包装成「二维列表」   [1,2,3] → [[1,2,3]]
        data = np.array(new_samples, dtype=object)
        # 遍历连续特征，用训练好的模型预测标签
        for idx in feature_indexes:
            feat_values = data[:, idx].astype(float)
            labels = self.kmeans_models[idx].predict(feat_values)
            data[:, idx] = labels
        return data.tolist()


# =========================== 三. 原决策树类（无修改） ===========================
# 决策树整体逻辑步骤：
# 1. 计算熵：衡量数据的混乱程度（越乱熵越高）
# 2. 分裂数据集：按指定特征和值把数据拆分成子集
# 3. 选最优分裂特征：找分裂后熵最小的特征
# 4. 构建树：递归用最优特征分裂数据，直到数据纯净
# 5. 预测：用建好的树判断新样本
class mytree:
    def __init__(self):
        self.Tree = None
        self.feature_labels = None

    # 计算熵
    def calculate_entropy(self, input_data):
        if isinstance(input_data[0], (list, tuple)):
            List = [i[-1] for i in input_data]  # 如果输入数据集，自动提取标签
        else:
            List = input_data  # 如果输入标签列表，直接使用
        labels_count = {}
        for label in List:
            labels_count[label] = labels_count.get(label, 0) + 1  # 统计标签出现次数
        entropy = 0.0
        for label in labels_count:
            prob = labels_count[label] / len(List)
            # 熵公式：H = -Σ(p * log2(p))
            entropy -= prob * log2(prob)
        return entropy

    # 分裂数据集：按指定特征和值把数据拆分成子集
    def split_data(self, data, index, value):
        subdata = []
        for row in data:
            if row[index] == value:  # 只保留指定值
                newrow = row[:index] + row[index + 1:]  # 去掉当前列
                subdata.append(newrow)
        return subdata

    # 选最优分裂特征：遍历所有特征 → 计算每个特征分裂后的加权总熵 → 选熵最小的特征
    def choose_best_feature(self, data):
        bestEntropy = float('inf')  # 初始化最大熵(只有二分类最大值为1,多分类可以超过1)
        bestFeature = -1  # 初始化最优特征
        features_number = len(data[0]) - 1  # 特征数量
        for i in range(features_number):  # 遍历所有特征 → 提取特征值 → 去重 → 计算加权熵 → 比较更新最优特征
            featList = [row[i] for row in data]  # 每个特征对应的所有特征值
            featVals = set(featList)  # 特征值去重
            newEntropy = 0.0
            for value in featVals:
                subdata = self.split_data(data, i, value)  # 按特征索引,特征值分裂数据
                prob = len(subdata) / float(len(data))  # 子集占比
                # 加权总熵：∑(子集熵 * 子集权重)
                newEntropy += self.calculate_entropy(subdata) * prob
            if newEntropy < bestEntropy:  # 更新最优特征,熵越小越好
                bestEntropy = newEntropy
                bestFeature = i
        return bestFeature

    # 构建树：递归用最优特征分裂数据，直到数据纯净
    def create_tree(self, data, labels):
        ClassList = [i[-1] for i in data]  # 提取所有标签,判断是否纯净
        # 递归停止条件1: 所有标签相同,数据纯净
        if ClassList.count(ClassList[0]) == len(ClassList):
            return ClassList[0]
        # 递归停止条件2: 没有特征可分,多数投票决定
        if len(data[0]) == 1:
            classcount = {}
            for vote in ClassList:
                classcount[vote] = classcount.get(vote, 0) + 1
            sortedClassCount = sorted(classcount.items(), key=lambda x: x[1], reverse=True)  # 按标签数降序排列
            return sortedClassCount[0][0]  # 第一个元组的标签

        bestFeat = self.choose_best_feature(data)  # 选最优特征(索引)
        bestFeatName = labels[bestFeat]  # 最优特征名
        Tree = {bestFeatName: {}}  # 初始化树  {键:bestFeatName,值:空字典}
        label_copy = labels.copy()  # 复制特征列表（避免修改原列表）
        del (label_copy[bestFeat])  # 删除已用的特征（避免重复分裂）
        # 递归构建子树
        featValues = [i[bestFeat] for i in data]  # 最优特征列的值
        uniqueValues = set(featValues)  # 对值去重
        for value in uniqueValues:
            # 递归：用分裂后的子集构建子树
            Tree[bestFeatName][value] = self.create_tree(self.split_data(data, bestFeat, value), label_copy)
        return Tree

    def fit(self, data, labels):
        self.feature_labels = labels
        self.Tree = self.create_tree(data, labels)  # 将训练好的树赋值给类属性
        return self.Tree

    def predict(self, new_sample):
        if isinstance(new_sample[0], (int, float)):  # 处理单个样本
            return self._predict(self.Tree, new_sample)
        else:  # 处理多个样本
            results = []
            for sample in new_sample:
                results.append(self._predict(self.Tree, sample))
            return results

    # 预测：用建好的树判断新样本
    def _predict(self, tree, new_sample):
        first_feature = list(tree.keys())[0]  # 取当前层决策特征
        feature_branches = tree[first_feature]  # 取该特征的子字典(分支)
        feat_index = self.feature_labels.index(first_feature)  # 特征索引
        sample_feat_value = new_sample[feat_index]  # 新样本的该特征值

        # 新增：判断特征值是否在分支中，不在则返回当前分支的多数类 (在遇到未见过的特征值时，能自动返回该分支下的多数类标签)
        if sample_feat_value not in feature_branches:
            # 提取当前分支下所有叶子节点的标签，返回出现最多的那个
            labels = [v for v in feature_branches.values() if not isinstance(v, dict)]
            if not labels:
                return 0  # 无标签时返回默认值
            # 统计每个标签的出现次数
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            # 返回出现次数最多的标签
            return max(label_counts, key=label_counts.get)

        branch_result = feature_branches[sample_feat_value]  # 进入对应分支(结果/子字典)
        # 如果分支结果是字典（还有子树），递归预测；否则直接返回标签
        if isinstance(branch_result, dict):
            return self._predict(branch_result, new_sample)
        else:
            return branch_result


# ===================== 4. 测试代码 =====================
if __name__ == "__main__":
    # 测试数据：[薪资(连续), 工作年限(连续), 学历(离散), 入职推荐(标签)]
    '''
    train_data = [[8000.0  , 1.5, 0, 'no' ],
                  [15000.0 , 3.0, 1, 'yes'],
                  [12000.0 , 2.0, 0, 'yes'],
                  [9000.0  , 1.0, 1, 'no' ],
                  [20000.0 , 5.0, 1, 'yes'],
                  [7000.0  , 0.5, 0, 'no' ],
                  [18000.0 , 4.0, 0, 'yes'],
                  [10000.0 , 1.8, 1, 'no' ]]
    '''

    # 清道夫训练数据
    X = pd.read_csv('../doctor/train_X.csv').values.tolist()
    y = pd.read_csv('../doctor/train_y.csv').values.flatten().tolist()
    #train_data = np.hstack((X,y))
    train_data = [xi + [yi] for xi, yi in zip(X, y)]

    # 特征名
    #feature_labels = ['薪资', '工作年限', '学历']
    feature_labels = ['气色', '气味', '脉象', '体温']

    # 连续特征的索引
    #continuous_indexes = [0, 1]  # 0=薪资、1=工作年限 是连续特征
    continuous_indexes = [0, 1, 2, 3]

    # 1. 离散化连续特征
    discretizer = ContinuousDiscretizer(n_clusters=3, random_state=42)
    train_data_discrete = discretizer.fit_transform(train_data, continuous_indexes)
    print("离散化后的训练数据：")
    for row in train_data_discrete:
        print(row)

    # 2. 训练决策树
    tree = mytree()
    trained_tree = tree.fit(train_data_discrete, feature_labels)
    print("\n训练好的决策树：", trained_tree)

    # 3. 预测新样本
    test_X = pd.read_csv('../doctor/test_X.csv').values.tolist()    # 清道夫待预测样本
    #new_sample_continuous = [[13000.0, 2.5, 1], [7500.0, 0.8, 0]]

    new_sample_discrete = discretizer.transform(test_X, continuous_indexes)
    print("\n离散化后的新样本：", new_sample_discrete)
    result = tree.predict(new_sample_discrete)
    print("预测结果：", result)