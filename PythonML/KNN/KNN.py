import math
# KNN核心3步: 算距离 → 找邻居 → 投票
class KNN:
    def __init__(self,k=5):
        self.k = k
        self.train_X = None  # 存训练特征
        self.train_y = None  # 存训练标签

    # 训练方法：只保存数据(KNN训练阶段不计算)
    def fit(self, train_X, train_y):
        # 把pandas的DataFrame/Series转成列表（适配pandas数据）
        self.train_X = train_X.values.tolist()  # 特征转列表
        self.train_y = train_y.values.ravel().tolist()  # 标签转一维列表

    # 步骤1:计算单个测试样本到单个训练样本的欧式距离
    # 欧式距离(最常用)公式:  开根号[ (x1-y1)**2 + (x2-y2)**2 +...+ (xn-yn)**2 ]
    def calculate_distance(self,train_sample,test_sample):
        distance = 0
        for i in range(len(train_sample)):
            distance += (train_sample[i] - test_sample[i])**2
        return math.sqrt(distance)

    # 步骤2：找出“最近的K个训练样本”(按距离从小到大排序后选K个)
    def sort_by_distance(self,test_X):  # 去掉train_X/train_y/k参数，用类的属性
        distance_list = []
        for i in range(len(self.train_X)):
            distance_list.append((self.calculate_distance(self.train_X[i],test_X), self.train_y[i]))  #存距离,对应的怀孕结果
        distance_list.sort()
        return [distance_list[i][1] for i in range(self.k)]

    # 步骤3：在K个邻居里“投票”选出现次数最多的那个
    def predict(self,test_X):
        # 把测试样本的DataFrame转成列表
        testX = test_X.values.tolist()
        results = []
        # 遍历所有测试样本
        for test in testX:
            neighbors = self.sort_by_distance(test)
            count = 0
            for i in neighbors:
                if neighbors.count(i)>count:
                    count = neighbors.count(i)
                    target = i
            results.append(target)
        return results