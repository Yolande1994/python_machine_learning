import numpy as np

class SimpleKMeans:  # 支持多维特征聚类（兼容单特征）
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter      # 最大迭代次数（防止死循环）
        self.random_state = random_state  # 固定随机种子，结果可复现
        self.centers = None  # 最终聚类中心（多维数组：[n_clusters, n_features]）
        self.labels = None   # 每个点的聚类标签（一维数组：[n_samples]）

    def _euclidean_distance(self, x1, x2):  # 支持单/多维欧氏距离计算
        # 转换为numpy数组，统一计算逻辑
        x1 = np.array(x1)
        x2 = np.array(x2)
        # 多维欧氏距离：根号下（各维度差值的平方和）
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X):  # 训练K-means（支持单/多维数据）
        X = np.array(X, dtype=float)  # 转换为numpy数组，支持布尔索引
        n_samples, n_features = X.shape if X.ndim > 1 else (len(X), 1)  # len(X) 等价于 X.shape[0]

        # 1. 初始化：随机选K个不重复样本作为初始中心（兼容单/多维）
        np.random.seed(self.random_state)
        if X.ndim == 1:  # 单特征场景（保持原逻辑）
            self.centers = np.random.choice(X, size=self.n_clusters, replace=False)
        else:  # 多维场景：先抽样本索引，再取对应样本作为初始中心
            sample_indexes = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            self.centers = X[sample_indexes]

        for _ in range(self.max_iter):
            # 2. 分配标签：每个点归到最近的中心
            self.labels = self.predict(X)
            '''
            labels = []
            for x in X:
                # 计算当前点到所有中心的距离（兼容单/多维）
                distances = [self._euclidean_distance(x, c) for c in self.centers]
                # 选距离最小的中心的索引作为标签
                labels.append(np.argmin(distances))
            self.labels = np.array(labels)
            '''

            # 3. 更新中心：每个簇的平均值作为新中心（兼容单/多维）
            new_centers = []
            for i in range(self.n_clusters):
                # 取当前簇的所有点
                cluster_points = X[self.labels == i]
                if len(cluster_points) == 0:  # 防止空簇（随机重选一个点）
                    cluster_points = X[np.random.choice(n_samples, size=1)] if X.ndim > 1 else np.random.choice(X, size=1)
                # 多维数据按列求均值（axis=0），单特征直接求均值
                cluster_mean = np.mean(cluster_points, axis=0) if X.ndim > 1 else np.mean(cluster_points)
                new_centers.append(cluster_mean)
            new_centers = np.array(new_centers)

            # 4. 收敛判断：中心不再变化则停止
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers


    def predict(self, X):

        pred_labels = []
        for x in X:
            distances = [self._euclidean_distance(x, c) for c in self.centers]  # 计算距离
            pred_labels.append(np.argmin(distances))  # 选距离最近的中心的索引为标签
        self.labels = np.array(pred_labels)
        return np.array(self.labels)


if __name__ == '__main__':
    '''
    # 单特征数据（原测试数据）
    X_1d = [200, 250, 300, 500, 550, 600]
    test_1d = SimpleKMeans(n_clusters=2)
    centers_1d = test_1d.fit(X_1d)
    print("单特征最终聚类中心：", centers_1d)
    print("单特征样本标签：", test_1d.labels)
    '''

    # 多特征数据（二维数据）
    X_2d = [[2000, 1], [2500, 2], [3000, 1.5], [5000, 5], [5500, 6], [6000, 5.5]]
    test_2d = SimpleKMeans(n_clusters=2)
    centers_2d = test_2d.fit(X_2d)
    print("\n二维特征最终聚类中心：")
    np.set_printoptions(suppress=True, precision=1) # 关闭科学计数法，保留1位小数
    print(centers_2d)
    print("二维特征样本标签：", test_2d.labels)