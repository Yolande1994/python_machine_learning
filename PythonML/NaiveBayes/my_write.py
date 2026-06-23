import math

import numpy as np
from pyexpat import features
'''
训练:
求出训练数组的类别数,
每个类别的先验概率,
每个特征在每个类别的均值,方差.

预测:
代入预测数组,
根据公式求出预测数组每个样本每个特征在每个类别的似然概率,
将每个特征的似然概率求积(log则求和),得到总似然概率,
总似然概率×(log则+)该类别先验概率==总概率分数,
每个样本取分数最高的类别为最终结果.
'''
X = np.array([[120, 25], [125, 30], [130, 35], [170, 60], [175, 65], [180, 70]]) # 身高  体重
y = np.array([0, 0, 0, 1, 1, 1])   # 小学生0  成年人1

class GNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.stds = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        for i,cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.priors[i] = len(X_cls)/len(X)
            self.means[i] = np.mean(X_cls, axis=0)
            self.vars[i] = np.var(X_cls, axis=0)
            self.stds[i] = np.sqrt(self.vars[i])
    # log
    def predict1(self, X):
        total_scores = []
        for i,cls in enumerate(self.classes):
            log_prior = np.log(self.priors[i])
            likelihood = -np.log(self.vars[i]) - (X - self.means[i])**2/(2*self.vars[i])
            likelihood_sum = likelihood.sum(axis=1)
            class_total_score = likelihood_sum + log_prior
            total_scores.append(class_total_score)
        total_scores = np.array(total_scores).T
        result = self.classes[np.argmax(total_scores, axis=1)]
        return result
    # 不log
    def predict2(self, X):
        total_scores = []
        for i,cls in enumerate(self.classes):
            # y = numpy.exp(-(x - mean) ** 2 / (2 * std** 2)) / (math.sqrt(2 * math.pi) * std)
            likelihood = np.exp(-(X - self.means[i]) ** 2 / (2 * self.stds[i]** 2)) / (math.sqrt(2 * math.pi) * self.stds[i])
            likelihood_sum = likelihood.prod(axis=1)  # 按行求积
            class_total_score = likelihood_sum * self.priors[i]
            total_scores.append(class_total_score)
        total_scores = np.array(total_scores).T
        result = self.classes[np.argmax(total_scores, axis=1)]
        return result

gnb = GNB()
gnb.fit(X, y)
gnb.predict1(X)