from math import log2
import pandas as pd

# 决策树整体逻辑步骤：
# 1. 计算熵：衡量数据的混乱程度（越乱熵越高）
# 2. 分裂数据集：按指定特征和值把数据拆分成子集
# 3. 选最优分裂特征：找分裂后熵最小的特征
# 4. 构建树：递归用最优特征分裂数据，直到数据纯净
# 5. 预测：用建好的树判断新样本

class CARTTree:
    def __init__(self):
        self.Tree = None
        self.feature_labels = None

    # 计算熵
    def calculate_entropy(self, input_data):
        if isinstance(input_data[0], (list, tuple)):
            List = [i[-1] for i in input_data]  # 如果输入数据集，自动提取标签
        else:
            List = input_data                   # 如果输入标签列表，直接使用
        labels_count = {}
        for label in List:
            labels_count[label] = labels_count.get(label,0) + 1    # 统计标签出现次数
        entropy = 0.0
        for label in labels_count:
            prob = labels_count[label]/len(List)
            # 熵公式：H = -Σ(p * log2(p))
            entropy -= prob * log2(prob)
        return entropy


    # 分裂数据集(离散特征)：按指定特征和值把数据拆分成子集
    def split_data(self, data, index, value):
        subdata = []
        for row in data:
            if row[index]==value:  # 只保留指定值
                newrow = row[:index] + row[index+1:]  # 去掉当前列
                subdata.append(newrow)
        return subdata

    # 二分分裂(连续特征)：把数据按阈值分成两部分
    def split_continuous_data(self, data, index, threshold):
        left_data = []   # 小于等于阈值的样本
        right_data = []  # 大于阈值的样本
        for row in data:
            if row[index] <= threshold:
                new_row = row[:index] + row[index+1:]  # 删除该特征列
                left_data.append(new_row)
            else:
                new_row = row[:index] + row[index+1:]
                right_data.append(new_row)
        return left_data, right_data


    # 选最优分裂特征：遍历所有特征 → 计算每个特征分裂后的加权总熵(加权熵=子集熵×子集占比) → 选熵最小的特征
    # 同时支持离散 + 连续特征
    def choose_best_feature(self, data):
        best_Entropy = float('inf') # 初始化最优熵
        best_Feature = -1           # 初始化最优特征
        best_Threshold = None       # 保存连续特征最优阈值
        features_number = len(data[0]) -1  # 特征数量
        for i in range(features_number):   # 遍历所有特征 → 提取特征值并去重 → 计算加权熵 → 比较更新最优特征
            feat_values = [row[i] for row in data]  # 提取每个特征的所有特征值
            # 判断是否为连续特征（简单判断：如果不是整数，就视为连续）
            if isinstance(feat_values[0], float):
                # 1. 提取该特征的所有值并排序去重
                sorted_unique_vals = sorted(set(feat_values))
                if len(sorted_unique_vals) == 1:  # 此特征的所有样本值都一样,无法做分裂,且此特征对分类没任何帮助
                    continue  # 只有一个值，无法分裂，跳过
                current_best_entropy = float('inf')  # 当前特征列最优熵
                current_best_threshold = None        # 当前特征列最优阈值
                # 2. 遍历所有可能的阈值（相邻值的中点）
                for j in range(len(sorted_unique_vals) - 1):
                    threshold = (sorted_unique_vals[j] + sorted_unique_vals[j+1]) / 2
                    # 3. 按阈值分裂
                    left, right = self.split_continuous_data(data, i, threshold)
                    # 4. 计算加权熵
                    prob_left = len(left) / len(data)
                    prob_right = len(right) / len(data)
                    entropy = prob_left * self.calculate_entropy(left) + prob_right * self.calculate_entropy(right)
                    # 5. 更新当前特征最优阈值
                    if entropy < current_best_entropy:
                        current_best_entropy = entropy
                        current_best_threshold = threshold
                # 对比所有特征的熵，更新全局最优
                if current_best_entropy < best_Entropy:
                    best_Entropy = current_best_entropy
                    best_Feature = i
                    best_Threshold = current_best_threshold

            # 离散特征：原来的逻辑
            else:
                unique_vals = set(feat_values)  # 特征值去重
                newEntropy = 0.0
                for value in unique_vals:
                    subdata = self.split_data(data,i,value)  # 按特征索引,特征值分裂数据
                    prob = len(subdata)/float(len(data))     # 子集占比
                    # 加权总熵：∑(子集熵 * 子集权重)
                    newEntropy += self.calculate_entropy(subdata) * prob
                if newEntropy < best_Entropy:  # 更新最优特征,熵越小越好
                    best_Entropy = newEntropy
                    best_Feature = i
        return best_Feature,best_Threshold  # 返回：最优特征索引 + 最优阈值（连续特征才有）


    # 构建树：递归用最优特征分裂数据，直到数据纯净
    def create_tree(self, data, labels):
        ClassList = [i[-1] for i in data]  # 提取所有标签,判断是否纯净
        # 递归停止条件1: 所有标签相同,数据纯净
        if ClassList.count(ClassList[0]) == len(ClassList):
            return ClassList[0]
        # 递归停止条件2: 只剩下标签列,没有特征可分,统计各标签个数,多数投票决定
        if len(data[0]) == 1:
            classcount = {}
            for vote in ClassList:
                classcount[vote] = classcount.get(vote,0) + 1
            sortedClassCount = sorted(classcount.items(), key=lambda x: x[1], reverse=True)  # 按标签数降序排列
            return sortedClassCount[0][0]  # 第一个元组的标签

        bestFeat, bestThreshold = self.choose_best_feature(data)  # 选最优特征（可能是连续的）（索引）
        bestFeatName = labels[bestFeat]    # 最优特征名
        Tree = {bestFeatName:{}}           # 初始化树  {键:bestFeatName,值:空字典}
        label_copy = labels.copy()         # 复制特征列表（避免修改原列表）
        del(label_copy[bestFeat])          # 删除已用的特征（避免重复使用）
        # 递归构建子树(连续/离散)      ' 赋值逻辑：Tree[bestFeatName][键] = 值 '
        if bestThreshold is not None:  # 判断是否为连续特征
            left_data, right_data = self.split_continuous_data(data, bestFeat, bestThreshold)
            Tree[bestFeatName]['<= ' + str(round(bestThreshold, 2))] = self.create_tree(left_data, label_copy)  # 键是区间判断，值是递归生成的子树 / 叶子节点
            Tree[bestFeatName]['> ' + str(round(bestThreshold, 2))] = self.create_tree(right_data, label_copy) # 分支键:[], 分支值: =右边
            # 把 Tree 想象成一个 “文件夹”：
            # Tree = {'体温': {}} → 创建了一个名为「体温」的空子文件夹；
            # Tree['体温']['<= 1.35'] = 0    → 往「体温」文件夹里放了一个名为「<= 1.35」的文件（内容是 0）；
            # Tree['体温']['> 1.35'] = {...} → 再往「体温」文件夹里放了一个名为「> 1.35」的子文件夹（内容是子树）。

        else:  # 离散特征：原来的逻辑
            featValues = [i[bestFeat] for i in data]  # 最优特征列的值
            uniqueValues = set(featValues)            # 对值去重
            for value in uniqueValues:
                # 递归：用分裂后的子集构建子树
                Tree[bestFeatName][value] = self.create_tree(self.split_data(data, bestFeat, value), label_copy)
        return Tree


    def fit(self,data,labels):
        self.feature_labels = labels
        self.Tree = self.create_tree(data, labels)  # 将训练好的树赋值给类属性
        return self.Tree


    def predict(self,new_sample):
        if isinstance(new_sample[0], (int, float)):  # 处理单个样本
            return self._predict(self.Tree,new_sample)
        else:                                        # 处理多个样本
            results = []
            for sample in new_sample:
                results.append(self._predict(self.Tree,sample))
            return results
    # 预测：用建好的树判断新样本
    def _predict(self,tree,new_sample):
        first_feature = list(tree.keys())[0]    # 取当前层决策特征
        feature_branches = tree[first_feature]  # 取该特征的子字典(分支)
        feat_index = self.feature_labels.index(first_feature)    # 特征索引
        sample_feat_value = new_sample[feat_index]  # 新样本的该特征值
        # 判断是否为连续特征的分支
        if list(feature_branches.keys())[0].startswith('<='):  # 检查第一个分支键是否以 '<= ' 开头，以此判断当前特征是连续特征还是离散特征
            # 连续特征
            branch_keys = list(feature_branches.keys())
            threshold_str = branch_keys[0].split(' ')[1]
            threshold = float(threshold_str)
            if sample_feat_value <= threshold:
                branch_result = feature_branches[branch_keys[0]]
            else:
                branch_result = feature_branches[branch_keys[1]]
        else:  # 离散特征
            branch_result = feature_branches[sample_feat_value]  # 进入对应分支(结果/子字典)
        # 如果分支结果是字典（还有子树），递归预测；否则直接返回标签
        if isinstance(branch_result, dict):
            return self._predict(branch_result, new_sample)
        else:
            return branch_result



if __name__ == "__main__":
    # 格式：技能(1好,0差), 态度(1好,0差), 招聘结果(yes/no)
    train_data = [[1, 1, 1,'yes'],
                  [1, 1, 0,'yes'],
                  [0, 1, 0,'no'],
                  [1, 0, 0,'no'],
                  [1, 0, 0,'no']]
    X = pd.read_csv('../doctor/train_X.csv').values.tolist()
    y = pd.read_csv('../doctor/train_y.csv').values.flatten().tolist()
    #train_X = numpy.hstack((X,y))
    train_X = [xi+[yi] for xi,yi in zip(X,y)]

    # 特征名
    #feature_labels = ['技能', '态度','专业']
    feature_labels = ['气色', '气味', '脉象','体温']

    # 新样本
    #new_sample = [[0, 1, 0],[1, 1, 1]]
    #new_sample = [5.8, 2.8, 5.1, 2.4]
    test_X = pd.read_csv('../doctor/test_X.csv').values.tolist()

    # 训练树
    cart = CARTTree()
    #trained_tree = cart.fit(train_data, feature_labels)
    trained_tree = cart.fit(train_X, feature_labels)

    # 预测
    result = cart.predict(test_X)
    print('树结构:',trained_tree)
    print('预测结果:',result)