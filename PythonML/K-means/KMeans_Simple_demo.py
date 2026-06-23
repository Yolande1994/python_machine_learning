import numpy as np

class SimpleKMeans:  # 仅支持单特征聚类，适配连续特征离散化场景
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter      # 最大迭代次数（防止死循环）
        self.random_state = random_state  # 固定随机种子（无论运行多少次，初始中心和聚类标签都和第一次完全一样，结果可复现）
        self.centers = None  # 最终聚类中心
        self.centers_indexes = None   # 每个点的聚类标签

    def _euclidean_distance(self, x1, x2):    # 计算两个数的欧氏距离（仅单特征下,欧氏距离=绝对值差）
        return abs(x1 - x2)

    def fit(self, X):  # 训练K-means（仅支持单特征一维数组）  param X: 一维数组，如[8000, 15000, 12000]
        X = np.array(X)
        # 1. 初始化：设置随机种子，随机选K个初始中心
        np.random.seed(self.random_state)
        # 从输入的样本数据 X 中，随机挑选 self.n_clusters 个不重复的样本，作为 K-means 的初始聚类中心
        self.centers = np.random.choice(X, size=self.n_clusters, replace=False)
        for _ in range(self.max_iter):
            # 2. 分配标签：每个点归到最近的中心
            indexes = []
            for x in X:
                # 计算当前点到所有中心的距离
                distances = [self._euclidean_distance(x, c) for c in self.centers]
                # 选距离最小的中心的索引作为标签
                indexes.append(np.argmin(distances))  # np.argmin()返回索引  np.min()返回值
            self.centers_indexes = np.array(indexes)
            # 3. 更新中心：每个簇的平均值作为新中心
            new_centers = []
            for i in range(self.n_clusters):
                # 取当前簇的所有点
                cluster_points = X[self.centers_indexes == i]
                if len(cluster_points) == 0:  # 防止空簇（随机重选一个点）
                    cluster_points = np.random.choice(X, size=1)
                new_centers.append(np.mean(cluster_points))  # 计算当前簇的所有样本的平均值（新的聚类中心），添加到new_centers列表中
            new_centers = np.array(new_centers)
            # 4. 收敛判断：中心不再变化则停止
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
        return self.centers



X = [200,250,300,500,550,600]
test = SimpleKMeans(n_clusters=2)
print(test.fit(X))