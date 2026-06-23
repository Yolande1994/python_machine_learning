from math import log2

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


    # 分裂数据集：按指定特征和值把数据拆分成子集
    def split_data(self, data, index, value):
        subdata = []
        for row in data:
            if row[index]==value:  # 只保留指定值
                newrow = row[:index] + row[index+1:]  # 去掉当前列
                subdata.append(newrow)
        return subdata


    # 选最优分裂特征：遍历所有特征 → 计算每个特征分裂后的加权总熵(加权熵=子集熵×子集占比) → 选熵最小的特征
    def choose_best_feature(self, data):
        bestEntropy = 1.0  # 初始化最大熵(只有二分类最大值为1,多分类可以超过1)
        bestFeature = -1   # 初始化最优特征
        features_number = len(data[0]) -1  # 特征数量
        for i in range(features_number):   # 遍历所有特征 → 提取特征值 → 去重 → 计算加权熵 → 比较更新最优特征
            featList = [row[i] for row in data]  # 每个特征对应的所有特征值
            featVals = set(featList)  # 特征值去重
            newEntropy = 0.0
            for value in featVals:
                subdata = self.split_data(data,i,value)  # 按特征索引,特征值分裂数据
                prob = len(subdata)/float(len(data))     # 子集占比
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
                classcount[vote] = classcount.get(vote,0) + 1
            sortedClassCount = sorted(classcount.items(), key=lambda x: x[1], reverse=True)  # 按标签数降序排列
            return sortedClassCount[0][0]  # 第一个元组的标签

        bestFeat = self.choose_best_feature(data)  # 选最优特征(索引)
        bestFeatName = labels[bestFeat]            # 最优特征名
        Tree = {bestFeatName:{}}                   # 初始化树  {键:bestFeatName,值:空字典}
        label_copy = labels.copy()                 # 复制特征列表（避免修改原列表）
        del(label_copy[bestFeat])                  # 删除已用的特征（避免重复分裂）
        # 递归构建子树
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
    # 特征名
    feature_labels = ['技能', '态度','专业']
    # 新样本
    new_sample = [[0, 1, 0],[1, 1, 1]]
    # 训练树
    dt = mytree()
    trained_tree = dt.fit(train_data,feature_labels)
    # 预测
    result = dt.predict(new_sample)
    print('树:',trained_tree)
    print('预测结果:',result)