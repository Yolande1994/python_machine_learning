import numpy as np

# 调整为伯努利朴素贝叶斯（按特征是否出现计数，而非值总和）
class BernoulliNB:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """
        初始化伯努利朴素贝叶斯分类器（适配二元特征：0=不出现，1=出现）
        :param alpha: 拉普拉斯平滑系数（默认1.0）
        :param fit_prior: 是否从数据中学习先验概率（默认True）
        :param class_prior: 手动指定的先验概率（默认None）
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.feature_log_prob_ = None  # 特征的对数似然概率（P(x=1|y)）
        self.class_log_prior_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        # ========== 核心改动1：伯努利NB的计数逻辑 ==========
        # 1. class_count：统计每个类别的样本个数（而非特征值总和）
        class_count = np.zeros(n_classes, dtype=np.float64)
        # 2. feature_count：统计每个类别下，特征=1的样本个数（而非特征值总和）
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        # 遍历每个类别，重新统计
        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]  # 取出当前类别的样本
            class_count[i] = len(X_cls)  # 关键：统计样本个数（不是sum）
            # 关键：统计特征=1的样本数（先转布尔型，再求和）
            feature_count[i] = np.sum(X_cls > 0, axis=0)  # >0表示特征出现（适配非0/1的计数特征）

        # ========== 先验概率计算（调整：按样本数算，而非特征值总和） ==========
        if self.fit_prior:
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)
            else:
                # 关键：先验概率=类别样本数/总样本数（伯努利NB标准逻辑）
                self.class_log_prior_ = np.log(class_count / n_samples)
        else:
            self.class_log_prior_ = np.log(np.ones(n_classes) / n_classes)

        # ========== 似然概率计算（调整：按样本数算，带平滑） ==========
        # 平滑：特征=1的样本数 + alpha
        smoothed_count = feature_count + self.alpha
        # 分母：类别样本数 + 2*alpha（因为特征只有0/1两种可能）
        smoothed_total = class_count[:, np.newaxis] + 2 * self.alpha
        # 计算P(x=1|y)的对数概率
        self.feature_log_prob_ = np.log(smoothed_count / smoothed_total)
        # 补充P(x=0|y)的对数概率（伯努利NB必须包含）
        self.feature_log_prob_0 = np.log(1 - np.exp(self.feature_log_prob_))

    def predict_proba(self, X):
        """
        预测概率（适配伯努利NB逻辑：同时计算x=1和x=0的概率）
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        # 核心：将输入特征转为二元值（>0=1，否则=0）
        X_binary = (X > 0).astype(np.float64)

        # 计算对数概率：log(P(y)) + sum(log(P(x_i|y)))
        # x=1时用feature_log_prob_，x=0时用feature_log_prob_0
        log_prob = self.class_log_prior_.reshape(1, -1) + \
                   np.dot(X_binary, self.feature_log_prob_.T) + \
                   np.dot(1 - X_binary, self.feature_log_prob_0.T)

        # 对数转原始概率并归一化
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)

    def predict(self, X):
        """保持和原代码一致的预测逻辑"""
        prob = self.predict_proba(X)
        max_idx = np.argmax(prob, axis=1)
        pred_result = self.classes[max_idx]
        if pred_result.shape[0] == 1:
            return pred_result[0]
        return pred_result


# ===================== 测试对比（和原代码用同一套数据） =====================
if __name__ == "__main__":
    # 1. 测试数据（和原代码一致）
    X_train = [
        [1, 0],  # 体育新闻
        [1, 1],  # 体育新闻
        [0, 0],  # 科技新闻
        [2, 1],  # 科技新闻
        [2, 0],  # 娱乐新闻
        [0, 1],  # 娱乐新闻
    ]
    y_train = [0, 0, 1, 1, 2, 2]

    X_test_single = [1, 0]
    X_test_batch = [[1, 0], [2, 1], [0, 1], [1, 1]]

    # 2. 训练伯努利NB模型
    print("===== 1. 训练伯努利NB模型（按特征出现次数计数） =====")
    clf_bern = BernoulliNB(alpha=1.0, fit_prior=True)
    clf_bern.fit(X_train, y_train)

    # 打印核心参数（对比原代码）
    print(f"识别到的类别：{clf_bern.classes}")
    print(f"类别对数先验概率：{np.round(clf_bern.class_log_prior_, 4)}")
    print(f"特征对数似然概率（P(x=1|y)）：\n{np.round(clf_bern.feature_log_prob_, 4)}")
    print("-" * 50)

    # 3. 单样本预测
    print("===== 2. 单样本预测 =====")
    pred_single = clf_bern.predict(X_test_single)
    prob_single = clf_bern.predict_proba(X_test_single)
    print(f"待预测单样本：{X_test_single}")
    print(f"预测类别：{pred_single}（0=体育，1=科技，2=娱乐）")
    print(f"类别概率分布：{np.round(prob_single, 4)}")
    print("-" * 50)

    # 4. 多样本预测
    print("===== 3. 多样本批量预测 =====")
    pred_batch = clf_bern.predict(X_test_batch)
    prob_batch = clf_bern.predict_proba(X_test_batch)
    print(f"待预测多样本：{X_test_batch}")
    print(f"批量预测结果：{pred_batch}")
    print(f"批量概率分布：\n{np.round(prob_batch, 4)}")