class NaiveBayesClassifier:
    """朴素贝叶斯二分类器（精简版）仅支持二分类和单样本"""
    def __init__(self):
        self.p_y1 = 0.0  # 先验概率 P(y=1)
        self.p_y0 = 0.0  # 先验概率 P(y=0)
        self.X1 = []  # 标签1对应的特征集
        self.X0 = []  # 标签0对应的特征集

    def fit(self, X, y):
        # 拆分特征集（按标签分组）
        self.X1 = [X[i] for i in range(len(X)) if y[i] == 1]
        self.X0 = [X[i] for i in range(len(X)) if y[i] == 0]
        # 计算先验概率
        self.p_y1 = len(self.X1) / max(len(y),1)
        self.p_y0 = len(self.X0) / max(len(y),1)
        return self

    def predict(self, x):
        # 计算特征在各标签下的似然概率乘积
        def _likelihood_product(X_group, x):
            if not X_group:  # 该标签无样本时概率为0
                return 0.0
            prob = 1.0
            for i, val in enumerate(x):
                print(i, val)
                # 计算 P(fi=val | y)
                count = sum(1 for sample in X_group if sample[i] == val)
                prob *= count / len(X_group)
            return prob
        # 联合概率 = 先验概率 × 似然概率乘积
        joint_y1 = self.p_y1 * _likelihood_product(self.X1, x)
        joint_y0 = self.p_y0 * _likelihood_product(self.X0, x)
        # 返回概率大的标签
        return 1 if joint_y1 > joint_y0 else 0


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # 1. 准备分离的特征和标签（符合sklearn风格）
    X = [[1, 0],[1, 0],[1, 1],[0, 0],[0, 0],[1, 1]]  # 特征部分
    y = [1, 1, 1, 1, 0, 0]  # 标签部分

    # 2. 初始化+训练+预测
    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    # 预测单个样本：[1,0]
    result = clf.predict([1,0])
    print(clf.X1)
    print(clf.X0)
    print(clf.p_y1)
    print(clf.p_y0)

    print(f"预测结果：{result}")  # 输出 1