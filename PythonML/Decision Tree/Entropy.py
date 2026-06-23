from math import log2  # 导入math库中的log2函数，专门用于计算以2为底的对数（熵的计算默认用2为底）

def calculate_shannon_entropy(labels):
    """
    计算给定标签列表的香农熵（Shannon Entropy）
    香农熵是衡量数据"混乱程度"的指标，公式如下：
    H = - Σ (p_i * log2(p_i))
    其中：   - H：香农熵值
            - p_i：第i个类别在数据集中出现的概率
            - Σ：求和符号，对所有类别的(p_i * log2(p_i))结果求和
            - 负号：保证熵值为非负数（因为0<p_i<1时，log2(p_i)是负数）
    参数: centers_indexes (list): 一维标签列表，例如 [1, 1, 0, 0, 0]
    返回: float: 香农熵值（值越大，数据越混乱；值为0时数据完全纯净）
    """
    # 1. 第一步：统计每个标签出现的次数（用于后续计算概率）
    label_counts = {}   # 创建空字典，键=标签值，值=该标签出现的次数
    # 遍历输入的标签列表，逐个统计
    for label in labels:
        # 如果标签还没在字典里，先初始化次数为0
        if label not in label_counts:
            label_counts[label] = 0
        # 每遇到一次该标签，次数+1
        label_counts[label] += 1
    # 2. 第二步：初始化熵值为0.0（浮点数，保证计算精度）
    entropy = 0.0
    # 计算标签的总数量（分母，用于算概率）
    total = len(labels)
    # 3. 第三步：按照熵的公式逐类别计算，最后求和
    # 遍历每个标签的出现次数
    for count in label_counts.values():
        # 计算该标签的概率：出现次数 / 总数量
        prob = count / total
        # 公式核心：累加 "概率 * log2(概率)"，并取相反数
        # 等价于 entropy = entropy - (prob * log2(prob))
        entropy -= prob * log2(prob)
    return entropy

#计算多个数据集合的熵之和 list是一个二维数组
def calculate_shannon_entropys(lsts):
    total_ent = 0.0
    for sublist in lsts:
        total_ent += calculate_shannon_entropy(sublist)
    return total_ent



# 主程序入口（只有直接运行该文件时，才会执行以下代码）
if __name__ == "__main__":
    # 定义测试用例，覆盖不同混乱程度的场景
    test_cases = [
        [1, 1, 0, 0, 0],  # 混合数据，熵≈0.971（中等混乱）
        [1, 1, 0, 0],     # 均匀混合，熵=1.0（最混乱）
        [1, 1, 0],        # 混合数据，熵≈0.918（中等混乱）
        [1, 1, 1],        # 纯数据， 熵=0.0（完全不混乱）
    ]

    # 遍历每个测试用例，计算并打印熵值
    for case in test_cases:
        # 调用函数计算熵值
        ent = calculate_shannon_entropy(case)
        print(f"数据集 {case} 的熵: {ent:.4f}")
    # 计算数据集合的总熵
    print(f'总熵值: {calculate_shannon_entropys(test_cases):.6f}')
