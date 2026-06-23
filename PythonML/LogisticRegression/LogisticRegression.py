import numpy as np
import matplotlib.pyplot as plt
from numpy import corrcoef, ones, shape, exp  # 新增exp（Sigmoid需要）

# 设置matplotlib支持中文显示（和原代码一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


class LogisticRegression:
    # 基于梯度下降实现的逻辑回归模型（分类任务）
    def __init__(self):
        self.w = None  # 权重参数（含截距）
        self.X_train = None
        self.y_train = None  # 目标值是分类标签（0/1）


    @staticmethod
    # 静态方法：加载文本数据
    def loadDataSet(fileName):
        fr = open(fileName)
        numFeat = -1
        Xlist = []
        ylist = []
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            if numFeat == -1:
                numFeat = len(curLine) - 1
            lineArr = []
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            Xlist.append(lineArr)
            ylist.append(float(curLine[-1]))
        fr.close()
        return Xlist, ylist


    # Sigmoid激活函数（逻辑回归核心，把线性输出转成概率）
    @staticmethod
    def sigmoid(z):
        # Sigmoid函数，z是线性输出（X*w）
        return 1.0 / (1 + exp(-z))


    # 训练模型：批量梯度下降法
    def fit_batch(self, X, y, alpha=0.01, maxCycles=500):
        # alpha=学习率，maxCycles=迭代次数
        self.X_train = np.asmatrix(X)    # 主流用np.array代替matrix
        self.y_train = np.asmatrix(y).T  # 分类标签是0/1
        m, n = shape(self.X_train)  # m=样本数，n=特征数（含截距）
        self.w = ones((n, 1))  # 初始化权重为全1

        for k in range(maxCycles):
            # 步骤1：计算线性输出z = X*w
            z = self.X_train * self.w # 顺序不能颠倒
            # 步骤2：Sigmoid转概率
            y_p = self.sigmoid(z)
            # 步骤3：计算误差（预测-真实）和平均梯度（批量梯度下降要除以样本数m）
            error = y_p - self.y_train
            gradient = (self.X_train.T * error) / m
            # 步骤4：梯度下降更新权重(逻辑回归的梯度公式,对数损失的'梯度': XT(y^p-y)（无MSE相关项、无2倍系数）)
            self.w = self.w - alpha * gradient  # w = w - alpha * (1/n) * X.T * (y_p - y)

        # 打印权重信息
        print(f'截距b: {self.w[0, 0]}')
        if self.w.shape[0] > 1:
            print(f'权重w: {self.w[1, 0]}')
        return self


    # 训练模型：随机梯度下降法
    def fit_stoc(self, X, y, alpha=0.01, maxCycles=500):
        self.X_train = np.asmatrix(X)
        self.y_train = np.asmatrix(y).T
        m, n = shape(self.X_train)
        self.w = ones((n, 1))  # 初始化权重

        for k in range(maxCycles):
            # 随机梯度下降：逐个样本更新
            for i in range(m):
                # 取第i个样本
                xi = self.X_train[i, :]
                yi = self.y_train[i, :]
                # 计算线性输出+概率
                z = xi * self.w
                y_prob = self.sigmoid(z)
                # 计算误差
                error = yi - y_prob
                # 更新权重（用单个样本的梯度）
                self.w = self.w + alpha * xi.T * error
                # error = y_prob - yi
                # 梯度 = xi.T * error
                # self.w = self.w - alpha * xi.T * error

        print(f'截距b: {self.w[0, 0]}')
        if self.w.shape[0] > 1:
            print(f'权重w: {self.w[1, 0]}')
        return self


    # 模型预测：输出类别（0/1）和概率
    def predict(self, X, threshold=0.5):
        if self.w is None:
            raise ValueError("模型尚未训练！请先调用fit_batch/fit_stoc方法训练模型。")
        X_mat = np.asmatrix(X)
        z = X_mat * self.w  # 线性输出
        y_prob = self.sigmoid(z)  # 转概率
        # 概率≥threshold→类别1，否则→0
        y_pred = np.where(y_prob >= threshold, 1.0, 0.0)
        return y_pred, y_prob  # 返回类别+概率


    # 评估模型：分类准确率（评估指标改为分类准确率）
    def score(self, X=None, y=None, threshold=0.5):
        if self.w is None:
            raise ValueError("模型尚未训练！请先调用fit方法训练模型。")
        # 优先用测试数据，无则用训练数据
        if X is None or y is None:
            X_mat = self.X_train
            y_mat = self.y_train
        else:
            X_mat = np.asmatrix(X)
            y_mat = np.asmatrix(y).T
        # 预测类别
        y_pred, _ = self.predict(X_mat, threshold)
        # 计算准确率（预测正确的样本数/总样本数）
        accuracy = np.mean(y_pred == y_mat)
        print(f"分类准确率：{accuracy:.4f}")
        return accuracy


    # 可视化训练数据和分类边界（仅适用于单特征+截距）
    def plot_fit(self, threshold=0.5):
        if self.w is None:
            raise ValueError("模型尚未训练！请先调用fit方法训练模型。")
        if self.X_train.shape[1] != 2:
            raise ValueError("仅支持单特征（含截距列）的可视化！")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 将矩阵转成一维数组（flatten() + A[0]）
        y_train_arr = self.y_train.flatten().A[0]  # 转成一维数组 [0,1,0,...]
        X_train_arr = self.X_train[:, 1].flatten().A[0]  # 特征列转一维数组

        # 绘制散点图（区分类别0和1）
        # 绘制类别0的散点
        idx0 = y_train_arr == 0  # 一维布尔数组，维度匹配
        ax.scatter(
            X_train_arr[idx0],  # 仅取类别0的特征值
            y_train_arr[idx0],  # 仅取类别0的标签
            label='类别0', c='blue'
        )
        # 绘制类别1的散点
        idx1 = y_train_arr == 1
        ax.scatter(
            X_train_arr[idx1],
            y_train_arr[idx1],
            label='类别1', c='red'
        )
        # 绘制分类边界（z=0 → X*w=0 → x1 = -w0/w1）
        # 逻辑回归中，z=0是分类边界（概率=0.5）
        x_bound = -self.w[0, 0] / self.w[1, 0]
        ax.axvline(x=x_bound, color='green', linestyle='--', label='分类边界')

        # 添加标签
        ax.set_xlabel('特征值')
        ax.set_ylabel('类别（0/1）')
        ax.set_title('逻辑回归分类结果')
        ax.legend()
        plt.show()


# ---------------------- 测试逻辑回归模型 ----------------------
if __name__ == '__main__':
    # 1. 创建模型实例
    lr_model = LogisticRegression()
    # 2. 加载数据（复用原代码的loadDataSet）
    # 注意：你的ex0.txt是回归数据（目标值是连续值），逻辑回归需要分类数据！
    # 这里先用示例分类数据（如果用ex0.txt，需要先把目标值转成0/1）
    # 示例分类数据（替换成你的分类数据即可）
    Xlist = [[1, 0.677], [1, 0.427], [1, 0.995], [1, 0.738], [1, 0.981], [1, 0.526]]
    ylist = [1, 0, 1, 1, 1, 0]  # 目标值是0/1分类标签

    # 3. 训练模型（可选批量/随机梯度下降）
    # 批量梯度下降
    lr_model.fit_batch(Xlist, ylist, alpha=0.01, maxCycles=10000)
    # 或随机梯度下降
    #lr_model.fit_stoc(Xlist, ylist, alpha=0.001, maxCycles=500)

    # 4. 预测
    y_pred, y_prob = lr_model.predict(Xlist)
    print(f"\n前5个预测类别：\n{y_pred[:5]}")
    print(f"前5个预测概率：\n{y_prob[:5]}")

    # 5. 评估模型效果
    lr_model.score()

    # 6. 可视化分类结果
    lr_model.plot_fit()