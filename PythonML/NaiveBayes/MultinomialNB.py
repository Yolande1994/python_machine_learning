import numpy as np
#  工业级的多项式朴素贝叶斯完整实现, 用 X@self.feature_log_prob_.T 矩阵乘法替代三层循环，速度提升10~100倍
#  伯努利 NB 的特征是「布尔型」（0/1），只关心 “有没有”，数 “样本个数”；
#  多项式 NB 的特征是「计数型」（0/1/2/3...），关心 “有多少”，数 “值的总和”
class MultinomialNB:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """
        初始化多项式朴素贝叶斯分类器
        :param alpha: 拉普拉斯平滑（Laplace Smoothing）的核心作用是避免概率为 0，而它并不会改变不同类别之间的概率相对大小，因此对最终的分类结果几乎没有影响。 alpha越大平滑越强
        :param fit_prior: 是否从数据中学习先验概率（默认True）
                          True：用训练集各类别占比算先验；False：所有类别先验概率相等（均匀分布）
        :param class_prior: 手动指定的先验概率（默认None）
                            例如：二分类时传入[0.6, 0.4]，表示类别0先验0.6，类别1先验0.4
        """
        # 保存初始化参数
        self.alpha = alpha  # 拉普拉斯平滑系数
        self.fit_prior = fit_prior  # 是否自动学习先验概率
        self.class_prior = class_prior  # 手动指定的先验概率
        # 训练后会赋值的核心参数（初始化时为None）
        self.classes = None  # 存储所有类别（如[0,1,2]），由训练集y自动识别
        self.feature_log_prob_ = None  # 特征的对数特征概率，形状：[类别数, 特征数]
        self.class_log_prior_ = None  # 类别的对数先验概率，形状：[类别数,]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes = np.unique(y)    # 识别所有类别（适配多分类）
        n_classes = len(self.classes)  # 类别数
        n_samples = X.shape[0]  # 总训练样本数（行数）   等价写法: n_samples, n_features = X.shape
        n_features = X.shape[1]  # 总特征数（列数）     机器学习里永远是「行 = 样本，列 = 特征」，比如 X.shape = (1000, 5) 就是1000个样本、5个特征
        # 创建 [n_classes行,n_features列] 的数组, 统计每个类别下，所有/每个 特征的总出现次数（为算特征概率做准备）
        class_count = np.zeros(n_classes, dtype=np.float64)                  # 创建一维数组，用来统计每个类别下，所有特征的值的总和
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)  # 创建二维数组，用来统计每个类别下，每个特征的值的总和

        # 遍历每个类别，统计计数
        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]  # 按类别划分样本
            class_count[i] = X_cls.sum()          # 当前类别所有特征值的总和                     [3, 3, 3]
            feature_count[i] = X_cls.sum(axis=0)  # 当前类别每个特征值的总和  index=0:按列求和     [[2, 1],[2, 1],[2, 1]]
#                                                                        shape0行1列,axis0列1行
        # 计算先验概率 P(y)（转对数避免后续乘法下溢）
        if self.fit_prior:  # fit_prior:初始化模型时传入的参数（默认True） True：从训练数据学习先验概率（最常用） False：假设各类别先验概率相等（均匀分布,比如3分类时每个类别都是 1/3）
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)                # 若手动指定了先验概率，就用这个指定值取对数
            else:
                self.class_log_prior_ = np.log(class_count / class_count.sum()) # 如果没有手动指定，就用训练数据计算[3/9,3/9,3/9] log:[-1.0986, -1.0986, -1.0986]
        else:
            self.class_log_prior_ = np.log(np.ones(n_classes) / n_classes)      # 当fit_prior=False，模型会假设所有类别先验概率相等

        # 计算特征概率 P(x|y)（带拉普拉斯平滑，转对数避免下溢）
        smoothed_count = feature_count + self.alpha    # [[3, 2], [3, 2], [3, 2]]
        smoothed_total = smoothed_count.sum(axis=1, keepdims=True) + self.alpha * n_features  # index=1:按行求和     a * n_feat:平滑的补偿项，确保分母和分子的平滑逻辑一致
        # sum后[[5], [5], [5]] + 1 * 2 , smoothed_total = [[7], [7], [7]]                     # keepdims=True:保留原数组的维度结构
        self.feature_log_prob_ = np.log(smoothed_count / smoothed_total) # [[3/7, 2/7], [3/7, 2/7], [3/7, 2/7]]  log:[[-0.8473,-1.2528], [-0.8473,-1.2528], [-0.8473,-1.2528]]

    def predict_proba(self, X):
        """
        预测每个样本属于各类别的概率（核心：向量化计算，替代3层循环. log对数转换，把乘法变成加法）
        P(y∣x) = P(y) × P(x1∣y) × P(x2∣y) ×⋯× P(xn∣y)                           →   把 “先验概率 × 特征概率之积”
        log(P(y∣x))=log(P(y)) + log(P(x∣y)) + log(P(x2∣y)) +⋯+ log(P(xn∣y))     → 转成 “对数先验 + 对数特征概率之和”，避免下溢，加快计算（结果一致）
        :param X: 待预测特征（单样本[1,0] / 多样本[[1,0],[0,1]]）
        :return: 概率矩阵，形状：[样本数, 类别数]，每行和为1
        """
        # 统一转为numpy数组并确保二维结构
        X = np.asarray(X)
        if X.ndim == 1:  # 判断输入是否为单样本（一维数组，比如 [1,0]）.  ndim：查看数组的 “维度数”
            X = X.reshape(1, -1)  # 把单样本转成二维数组（比如 [[1,0]]），方便后续的矩阵乘法统一处理单样本和多样本.   reshape(): 在不改变数组元素的前提下，调整数组的维度和尺寸.  1:固定为1行,-1:自动计算列数（元素总数÷行数）
        # 核心公式（对数形式）：log(P(y|x)) = log(P(y)) + sum(log(P(x_i|y)))
        log_prob = self.class_log_prior_ + X @ self.feature_log_prob_.T  # [[a,b],[a,b],[a,b]] .T转置后:[[a,a,a],[b,b,b]]
        # X @ self.feature_log_prob_.T = [[1,0]] @ [[-0.8473,-1.2528], [-0.8473,-1.2528], [-0.8473,-1.2528]]
        #                              = [[1*-0.8473 + 0*-1.2528, 1*-0.8473 + 0*-1.2528, 1*-0.8473 + 0*-1.2528]] = [[-0.8473, -0.8473, -0.8473]]
        # 以上矩阵乘法@把[[1,0]]每个特征值和其对应的特征概率相乘，再相加，一次性算出「每个类别下，样本所有特征的对数似然概率之和」
        # log_prob = [-1.0986, -1.0986, -1.0986] + [[-0.8473,-0.8473,-0.8473]] = [[-1.9459, -1.9459, -1.9459]]
        # 对数联合概率转回原始联合概率并归一化
        prob = np.exp(log_prob)  # log:[[-1.9459, -1.9459, -1.9459]] → [[0.143, 0.143, 0.143]]
        return prob / prob.sum(axis=1, keepdims=True)  # [[0.143/0.429, 0.143/0.429, 0.143/0.429]] = [[0.333, 0.333, 0.333]]

    def predict(self, X):
        """
        预测最终类别（取概率最大的类别）
        :param X: 待预测特征（单样本/多样本）
        :return: 预测结果（单样本返回标量，多样本返回列表）
        """
        # 获取每个样本的类别概率分布
        prob = self.predict_proba(X)
        # 取概率最大的类别索引
        max_idx = np.argmax(prob, axis=1)
        # 索引映射到真实类别
        pred_result = self.classes[max_idx]

        # 单样本时返回标量，提升使用体验
        if pred_result.shape[0] == 1:
            return pred_result[0]
        return pred_result

# 对数函数的基本性质:  log (ab) = log (a) + log (b)
# 不用数学证明，用直觉理解：
# log 的本质是 “指数的反函数”
# 乘法在指数世界里就是加法
# 例如：10^a × 10^b = 10^(a+b)
# log 就是把 “指数的加法” 再翻回来，所以：
# log(10^a × 10^b) = log(10^(a+b)) = a + b
# 也就是 log (ab) = log (a) + log (b)

# 拉普拉斯平滑:
# 1.平滑的目的：避免零概率，让模型更鲁棒，而不是改变分类结果。
# 2.平滑的公平性：对所有类别和特征都加相同的 alpha，保持了概率的相对比例。
# 3.决策的依据：朴素贝叶斯只关心 “哪个类别的后验概率更大”，而不是概率的绝对数值。
# 因此，只要 alpha 取值合理（通常取 1），拉普拉斯平滑不会影响最终的后验概率和分类结果，只会让模型更稳定。

# ===================== 完整测试案例 =====================
if __name__ == "__main__":
    # 1. 准备测试数据（多分类场景：3分类）
    # 训练集：模拟"文本分类"场景（特征=词频，类别=新闻类型：0=体育，1=科技，2=娱乐）
    X_train = [
        [1, 0],  # 体育新闻
        [1, 1],  # 体育新闻
        [0, 0],  # 科技新闻
        [2, 1],  # 科技新闻
        [2, 0],  # 娱乐新闻
        [0, 1],  # 娱乐新闻
    ]
    y_train = [0, 0, 1, 1, 2, 2]

    # 测试集：包含单样本和多样本
    X_test_single = [1, 0]
    X_test_batch = [
        [1, 0],
        [2, 1],
        [0, 1],
        [1, 1],
    ]

    # 2. 初始化+训练模型
    print("===== 1. 初始化并训练模型 =====")
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, y_train)

    # 打印训练后的核心参数
    print(f"识别到的类别：{clf.classes}")
    print(f"类别对数先验概率：{np.round(clf.class_log_prior_, 4)}")
    print(f"特征对数特征概率（形状{clf.feature_log_prob_.shape}）：\n{np.round(clf.feature_log_prob_, 4)}")
    print("-" * 50)

    # 3. 单样本预测
    print("===== 2. 单样本预测 =====")
    pred_single = clf.predict(X_test_single)
    prob_single = clf.predict_proba(X_test_single)

    print(f"待预测单样本：{X_test_single}")
    print(f"预测类别：{pred_single}（0=体育，1=科技，2=娱乐）")
    print(f"类别概率分布：{np.round(prob_single, 4)}")
    print("-" * 50)

    # 4. 多样本批量预测
    print("===== 3. 多样本批量预测 =====")
    pred_batch = clf.predict(X_test_batch)
    prob_batch = clf.predict_proba(X_test_batch)

    print(f"待预测多样本：{X_test_batch}")
    print(f"批量预测结果：{pred_batch}")
    print(f"批量概率分布（每行对应一个样本）：\n{np.round(prob_batch, 4)}")
    print("-" * 50)

    # 5. 验证手动指定先验概率
    print("===== 4. 手动指定先验概率测试 =====")
    clf_prior = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=[0.5, 0.3, 0.2])
    clf_prior.fit(X_train, y_train)

    prob_single_prior = clf_prior.predict_proba(X_test_single)
    print(f"手动指定先验后，单样本概率分布：{np.round(prob_single_prior, 4)}")
    print("（注：体育类概率更高，因为手动指定了更高的先验）")