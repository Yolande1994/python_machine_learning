# 导入以2为底的对数函数（熵计算的核心工具）
from math import log2

# 格式：[技能(1=好,0=差), 态度(1=好,0=差), 招聘结果(yes/no)]
train_data = [
    [1, 1, 'yes'],
    [1, 1, 'yes'],
    [0, 1, 'no'],
    [1, 0, 'no'],
    [1, 0, 'no']
]
# 特征名称（和数据列对应）
feature_labels = ['技能', '态度']
# 第6个新样本：技能差，态度好
new_sample = [0, 1]

# =============================  决策树核心模块  ===================================
# 决策树整体逻辑步骤：
# 1. 计算熵：衡量数据的混乱程度（越乱熵越高）
# 2. 分裂数据集：按指定特征和值把数据拆分成子集
# 3. 选最优特征：找分裂后熵最小的特征（决策树的"决策点"）
# 4. 构建树：递归用最优特征分裂数据，直到数据纯净（树的主体）
# 5. 预测：用建好的树判断新样本（树的最终用途）

# ------------------------ 模块1：计算香农熵（核心：衡量混乱度） ------------------------
def calculate_shannon_entropy(labels):
    """
    计算标签列表的香农熵（数据越乱，熵值越大）
    :param labels: 标签列表，如 ['yes','yes','no','no','no']
    :return: 熵值（float），0表示数据完全纯净，1表示最混乱
    """
    # 统计每个标签出现的次数（键=标签，值=次数）
    label_counts = {}
    for label in labels:  # 不存在则默认0，存在则取原值，统一+1
        label_counts[label] = label_counts.get(label, 0) + 1
    entropy = 0.0  # 初始化熵值
    total = len(labels)  # 总样本数
    # 按熵公式计算：H = -Σ(p * log2(p))
    for count in label_counts.values():
        prob = count / total  # 单个标签的概率
        entropy -= prob * log2(prob)  # 熵的核心计算
    return entropy


# ------------------------ 模块2：分裂数据集（核心：生成分支） ------------------------
def split_dataset(dataSet, index, value):
    """
    按指定特征分裂数据集（决策树的"分支"操作）
    :param dataSet: 原始数据集，如 [[1,1,'yes'], [0,1,'no']]
    :param index: 特征索引（0=第一个特征，1=第二个特征）
    :param value: 特征值（如 1=技能好，0=技能差）
    :return: 分裂后的子集（去掉当前特征列）
    """
    retDataSet = []
    for featVec in dataSet:
        # 筛选：只保留特征值等于指定值的样本
        if featVec[index] == value:
            # 降维：去掉当前分裂用的特征列（避免重复分裂）
            reducedFeatVec = featVec[:index] + featVec[index + 1:] # [:index] 取“要删列”左边的所有元素； [index+1:] 取“要删列”右边的所有元素；
            retDataSet.append(reducedFeatVec)
    return retDataSet


# ------------------------ 模块3：选择最优分裂特征（核心：找决策点） ------------------------
def choose_best_feature_to_split(dataSet):
    """
    选择最优分裂特征（找最能"理清混乱"的特征）
    :param dataSet: 原始数据集
    :return: 最优特征的索引（0/1/...）
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量（最后一列是标签，要排除）
    bestEntropy = 1.0  # 初始化分裂最优熵
    bestFeature = -1  # 初始化最优特征索引
    # 遍历所有特征，找信息增益最大的那个
    for i in range(numFeatures):
        # 提取当前特征的所有值
        featList = [sample[i] for sample in dataSet]    # [1,1,0,1,1]/[1,1,1,0,0]
        uniqueVals = set(featList)  # 去重（集合自动去重，只保留唯一值，避免重复计算） # {0, 1}/{0, 1}
        newEntropy = 0.0  # 初始化分裂后的总熵
        # 按当前特征的每个值分裂数据，计算加权熵
        for value in uniqueVals:
            subDataSet = split_dataset(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 子集占总数据的比例（权重）
            # 加权计算总熵：∑(子集熵 * 子集权重)
            newEntropy += calculate_shannon_entropy([s[-1] for s in subDataSet]) * prob  # 对“态度”特征来说：newEntropy=(0.4×0)+(0.6×0.918)≈0.551
            print(f'特征{i}值为{value}的子集:{subDataSet} 权重:{prob} 子集熵:{calculate_shannon_entropy([s[-1] for s in subDataSet])} 加权总熵:{newEntropy}')
        # 更新最优特征
        if newEntropy < bestEntropy:
            bestEntropy = newEntropy
            bestFeature = i
    return bestFeature
print(choose_best_feature_to_split(train_data))


# ------------------------ 模块4：构建决策树（核心：递归生成树） ------------------------
def create_tree(dataSet, labels):
    """
    核心逻辑：每一层选最优分裂特征 → 按特征值分裂数据集 → 对每个子集递归构建子树 → 把子树 / 标签填入字典，最终形成嵌套字典结构的决策树。
    :param dataSet: 数据集
    :param labels: 特征名称列表（如 ['技能','态度']）
    :return: 决策树（字典形式，如 {'态度': {0: 'no', 1: {'技能': {0: 'no', 1: 'yes'}}}}）
    """
    # 提取所有标签（判断是否纯净）
    classList = [sample[-1] for sample in dataSet]
    # 递归停止条件1：所有标签相同（数据纯净，无需分裂）
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归停止条件2：没有更多特征可分（按多数投票决定）
    if len(dataSet[0]) == 1: # 样本里只剩下最后一列（标签），所有特征都已被用来分裂过，没有更多特征可用
        classCount = {}  # 用字典统计每个标签出现的次数
        for vote in classList: # 简单多数投票：返回出现次数最多的标签
            classCount[vote] = classCount.get(vote, 0) + 1
        # 按次数排序，返回最多的那个      # classCount.items():把字典转成可迭代的元组（键值对）  key=:sorted()的一个参数，指定排序的依据   lambda x:x[1]:取第二个元素(值)来排序
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)   # 例:sorted自动遍历到('yes',2) → x[1] = 2
        return sortedClassCount[0][0]  # 取元组第一位的标签

    # 核心步骤：选最优特征
    bestFeat = choose_best_feature_to_split(dataSet)  # 索引
    bestFeatName = labels[bestFeat]  # 最优特征的名称（如'态度'）
    # 初始化决策树（字典结构：{特征名: {特征值: 子树/标签}}）
    Tree = {bestFeatName: {}}  # {键:bestFeatName,值:空字典}
    print('树:',Tree)
    # 复制特征标签列表（避免修改原列表）
    labels_copy = labels[:]  # 不能直接 labels_copy=centers_indexes, 列表、字典、集合是可变对象, = 是贴标签, [:]/.copy() 才是真复制
    del (labels_copy[bestFeat])  # 删除已用的特征（避免重复分裂）
    # 提取最优特征的所有值，递归构建子树
    featValues = [sample[bestFeat] for sample in dataSet]
    uniqueVals = set(featValues)  # 集合去重
    for value in uniqueVals:
        # 递归：用分裂后的子集构建子树
        Tree[bestFeatName][value] = create_tree(split_dataset(dataSet, bestFeat, value), labels_copy)
        # Tree[bestFeatName] → 找到已有子字典比如Tree = {'态度': {}}
        # [value] = ... → 把 value（如0）作为新键，把等号右边的结果（如'no'）作为值，添加到第一步找到的子字典中. Tree = {'态度': {0: 'no'}}
        # 嵌套赋值：先找Tree['态度']（空字典），再给它的键0赋值'no' / 键1赋值{'技能': {0: 'no', 1: 'yes'}}
        # Tree['态度'][0] = 'no'  →  输出：{'态度': {0: 'no'}}
        print(f'键:{Tree.keys()}  值:{Tree[bestFeatName][value]}')
        print('树:',Tree)
    #print('树:',Tree)

    return Tree
# ┌─────────────────────────────────────────┐
# │  第一层递归（根节点）：选特征“态度”            │
# ├───────────────┬─────────────────────────┤
# │ 态度=0         │ 态度=1                   │
# │ 第二层递归      │ 第二层递归                │
# │ 标签全为no      │ 选特征“技能”              │
# │ 返回：no       ├──────────────┬──────────┤
# │               │ 技能=0       │ 技能=1    │
# │               │ 第三层递归    │ 第三层递归  │
# │               │ 标签全为no    │ 标签全为yes│
# │               │ 返回：no     │ 返回：yes  │
# └───────────────┴─────────────┴──────────┘

# ------------------------ 模块5：预测（核心：用树判断新样本） ------------------------
def predict(tree, labels, new_sample):
    """
    用决策树预测新样本（按树的规则一步步查，直到找到标签）
    :param tree: 训练好的决策树
    :param labels: 特征名称列表
    :param new_sample: 新样本（如 [0,1]）
    :return: 预测结果（如 'no'）
    """
    # 取树的第一个决策特征（如'态度'）（决策树的每个层级只有一个根特征，所以这一步总是取当前层级的决策特征）
    first_feature = list(tree.keys())[0]  # '态度'   递归后:'技能'
    # 从当前树字典中，取出该特征对应的子字典（分支）（如tree['态度']: {0: 'no', 1: {'技能': {...}}}）
    feature_branches = tree[first_feature]  # {0: 'no', 1: {'技能': {0: 'no', 1: 'yes'}}}   递归后:{0: 'no', 1: 'yes'}
    # 找该特征在标签列表中的索引（如labels.index('态度') 会返回 1）
    feat_index = labels.index(first_feature)  # 1    递归后:0
    # 取新样本的该特征值（如 [0,1] 中'态度'的值是1）
    sample_feat_value = new_sample[feat_index]  # 1    递归后:0
    # 进入对应分支
    branch_result = feature_branches[sample_feat_value]  #  {'技能': {0: 'no', 1: 'yes'}}
    # 如果分支结果是字典（还有子树），递归预测；否则直接返回标签
    if isinstance(branch_result, dict):
        classLabel = predict(branch_result, labels, new_sample) # predict({'技能': {...}}, centers_indexes, [0,1])
    else:
        classLabel = branch_result     # 递归后:'no'
    return classLabel



# ======================== 新手友好的测试代码 ========================
if __name__ == "__main__":
    # 1. 准备数据（5个训练样本）
    # 格式：[技能(1=好,0=差), 态度(1=好,0=差), 招聘结果(yes/no)]
    train_data = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [0, 1, 'no'],
        [1, 0, 'no'],
        [1, 0, 'no']
    ]
    # 特征名称（和数据列对应）
    feature_labels = ['技能', '态度']
    # 第6个新样本：技能差，态度好
    new_sample = [0, 1]

    # 2. 训练决策树
    print("===== 训练决策树 =====")
    dt_tree = create_tree(train_data, feature_labels)
    print("生成的决策树结构：", dt_tree)
    # 树结构解读：
    # {'态度': {0: 'no', 1: {'技能': {0: 'no', 1: 'yes'}}}}
    # → 第一步看态度：态度差(0)→拒绝(no)；态度好(1)→看技能
    # → 技能差(0)→拒绝(no)；技能好(1)→录用(yes)

    # 3. 预测新样本
    print("\n===== 预测新样本 =====")
    result = predict(dt_tree, feature_labels, new_sample)
    print(f"新样本 [技能差, 态度好] 的预测结果：{result}")
