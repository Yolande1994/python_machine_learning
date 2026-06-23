import numpy as np


class GaussianNB_Engineering_demo:  # 以doctor数据为例
    def __init__(self):
        self.classes = None  # 所有类别（如[0,1,2]）
        self.class_priors = None  # 每个类的先验概率
        self.means = None    # 每个类在每个特征上的均值（形状：n_classes × n_features）
        self.vars = None     # 每个类在每个特征上的方差（形状：n_classes × n_features）
    # 训练过程：计算均值、方差、先验概率
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]  # 取列
        # 初始化存储结构
        self.means = np.zeros((n_classes, n_features))  # 3×4的0数组
        self.vars = np.zeros((n_classes, n_features))   # 3×4的0数组
        self.class_priors = np.zeros(n_classes)         # 3×1的0数组
        # 对每个类别，计算均值、方差、先验概率
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # 按类分割X
            self.means[idx] = X_c.mean(axis=0)  # 每个特征的均值     index=0:按列计算，对每个特征单独算均值[4.98, 3.25, 1.42, 0.25]
            self.vars[idx] = X_c.var(axis=0)    # 每个特征的方差     index=0:按列计算，对每个特征单独算方差[0.18, 0.15, 0.08, 0.01]
            self.class_priors[idx] = len(X_c) / len(X)  # 先验概率  如: 30/100 = 0.3

    # 预测过程：用对数分数比较
    def predict(self, X):  # X测试特征（38×4），输出预测标签（38×1）
        if X.ndim == 1:
            X = X.reshape(1, -1)  # 如果是单个样本,转成二维
        log_scores = []  # 存每个类别的对数分数，最终形状3×38（3类×38个测试样本）
        for idx, c in enumerate(self.classes): # 遍历每个类别（0/1/2）
            # 1. 先验概率的对数
            log_prior = np.log(self.class_priors[idx])
            # 2. 每个特征的对数似然求和（工程核心！）（简化版：省略-0.5*log(2π)，不影响比较）
            # self.vars[idx]: 该类的4个特征方差；        self.means[idx]: 该类的4个特征均值      → 形状(4,)（一维数组）
            log_likelihood = -np.log(self.vars[idx]) - (X - self.means[idx]) ** 2 / (2 * self.vars[idx]) # log_likelihood形状: 因广播机制变成X的形状()
            log_likelihood_sum = log_likelihood.sum(axis=1)  # 多维特征按行求和: 该样本属于当前类的总似然度
            # 3. 该类的总对数分数
            total_score = log_prior + log_likelihood_sum
            log_scores.append(total_score) # 把该类的38个样本分数加入列表
        # 转置数组：从3×38 → 38×3（每个样本对应3个类别的分数）
        log_scores = np.array(log_scores).T
        predictions = self.classes[np.argmax(log_scores, axis=1)] # 按行取最大值的索引
        return predictions  # 输出38个样本的预测标签（38×1）

'''
变量	                 形状	    含义
训练 X	             100×4	    100 个训练样本，4 个特征
训练 y	             100×1      100 个训练标签
self.means	         3×4	    3 个类别，每个类别 4 个特征的均值
self.means[idx]	     4,         4 个特征的均值
self.vars	         3×4	    3 个类别，每个类别 4 个特征的方差
self.vars[idx]	     4,         4 个特征的方差
self.class_priors	 3×1	    3 个类别的先验概率（0.3/0.35/0.35）
测试 X	             38×4	    38 个测试样本，4 个特征
X - self.means[idx]  38×4       38 个样本的每个特征与类别均值的差值!（广播机制）
log_likelihood       38×4       最终对数似然值（38 样本 ×4 特征）
log_likelihood_sum   38,        单个类别下，38 个测试样本的「4 个特征对数似然求和值」
total_score          38,        单个类别下，38 个测试样本的「先验对数 + 似然和」
log_scores（转置前）	 3×38	    3 个类别，每个类别 38 个测试样本的分数
log_scores（转置后）	 38×3	    38 个测试样本，每个样本 3 个类别的分数
预测结果	             38×1	    38 个测试样本的预测标签
'''

#在所有主流机器学习框架（sklearn、TensorFlow、PyTorch）中，预测逻辑（对数似然计算、分数比较）都严格放在 predict 方法里，fit 只负责计算和保存模型参数。
# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
# 模拟数据：特征是二维（身高、体重），类别是0（小学生）、1（成年人）
    X = np.array([
        [120, 25], [125, 30], [130, 35],  # 小学生（0）
        [170, 60], [175, 65], [180, 70]  # 成年人（1）
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    # 训练+预测
    gnb = GaussianNB_Engineering_demo()
    gnb.fit(X, y)
    # 预测：身高135，体重40 → 应该是小学生（0）
    test_sample = np.array([[135, 40]])
    print("预测类别：", gnb.predict(test_sample))  # 输出：[0]