import numpy as np

# K-means 流程:
# 1.随机选择K个点作为中心点
# 2.分类: 计算每个数据点到各中心点的距离,归类于距离最近的中心
# 3.挪位置: 用簇内数据点的平均值,更新为新的簇中心
# 4.重复 2-3 步直到数据收敛(中心点不再移动)

class KMeans():
    def __init__(self, n_clusters=3, cycles=100, random_state=42):
        self.cycles = cycles
        self.n_clusters = n_clusters
        self.random_state = random_state
        # 初始中心点
        self.centers = None

    def calculate_distance(self,a,b):
        return abs(a-b)

    def fit(self, X):
        X = np.array(X)
        # 固定随机初始值(方便调试对比)
        np.random.seed(self.random_state)
        # 1.随机选择K个点作为中心点
        self.centers = np.random.choice(X, size=self.n_clusters, replace=False)
        print(self.centers)

        for _ in range(self.cycles):
            # 2.分类: 计算每个数据点到各中心点的距离,归类于距离最近的中心
            min_distance_indexes = self.predict(X)
            # 3.挪位置: 用簇内数据点的平均值,更新为新的簇中心
            new_centers = []
            for i in range(self.n_clusters):
                new_centers.append(np.mean(X[min_distance_indexes == i]))
            new_centers = np.array(new_centers)
            print(new_centers)
            # 4.重复 2-3 步直到数据收敛(中心点不再移动)
            if np.allclose(new_centers,self.centers):
                break
            self.centers = new_centers
        return self.centers

    def predict(self, X):
        min_distance_indexes = []
        for x in X:
            # 计算每个点到每个中心点的距离
            distances = [self.calculate_distance(x, c) for c in self.centers]
            # 归类于距离最近的中心
            min_distance_indexes.append(np.argmin(distances))
        return np.array(min_distance_indexes)


if __name__ == '__main__':

    X = [100,150,200,300,350,400,500,550,600]
    test = KMeans(n_clusters=3)
    test.fit(X)

    fortestX = [1,2,3,4,5,6,7,8,9,700]
    y = test.predict(fortestX)
    print(y)