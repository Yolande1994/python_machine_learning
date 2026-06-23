import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import shape, exp

# ======================== 环境配置 =========================
# 设置matplotlib支持中文显示，避免图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


class LogisticRegression:
    # 多分类逻辑回归模型（基于梯度下降实现）核心原理：通过Sigmoid函数将线性输出转换为概率，梯度下降优化权重. 适用场景：0/1/2三类分类任务

    def __init__(self):
        self.weights = None  # 权重矩阵（形状：类别数 × 特征数，含截距项）
        self.classes = None  # 数据集中的类别列表（如[0,1,2]）
        self.X_train = None  # 训练特征矩阵（已添加截距列）
        self.y_train = None  # 训练标签（一维数组）
        self.feature_names = ['气色', '气味', '脉相', '体温']  # 特征名称（方便可视化理解）

    # ===================== 数据加载模块 =====================
    @staticmethod
    def load_csv(file_path):
        df = pd.read_csv(file_path)  # 参数：file_path: CSV文件路径（如'../tree/train_X.csv'）
        return df.values             # 返回：numpy数组（自动去除CSV表头，适配模型计算）

    # ===================== 核心函数：Sigmoid激活 =====================
    @staticmethod
    # Sigmoid激活函数：将线性输出转换为0~1的概率
    def sigmoid(z):  # 参数z: 线性输出（X * w）
        z = np.clip(z, -100, 100)  # 限制z的取值在[-100,100]之间
        return 1.0 / (1 + exp(-z))

    # ===================== 训练函数1：批量梯度下降 =====================
    def fit1(self, X, y, alpha=0.0001, maxCycles=1000):
        """
        核心逻辑：
            1. 给特征加截距列（X0=1，对应权重中的截距项）
            2. 对每个类别训练二分类器（当前类为1，其他类为0）
            3. 批量计算所有样本的梯度，统一更新权重
        """
        # 步骤1：数据预处理（添加截距列）
        self.X_train = np.asmatrix(X)
        # 截距列：把全1的列向量拼接在特征矩阵最左侧（X0=1）
        self.X_train = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train]) # np.hstack([A, B]):把两个矩阵左右拼接，要求行数一致   np.ones((行数, 列数))矩阵
        self.y_train = np.asarray(y).ravel()     # 转为一维数组，方便后续处理
        self.classes = np.unique(self.y_train)   # 获取所有类别（unique()遍历数组，去掉重复类，保留唯一值0/1/2）
        n_classes = len(self.classes)
        n_samples, n_features = self.X_train.shape  # n_samples:样本数（112）  n_features:特征数（含截距列5）

        # 步骤2：创建形状为“类别数×特征数”的全1矩阵，作为模型权重初始值，后续梯度下降更新时，保证每个类别、每个特征都有初始的权重值参与计算
        self.weights = np.ones((n_classes, n_features)) # np.ones((a, b)):创建一个a行b列的矩阵，所有元素值都是 1
        print(f"开始批量梯度下降训练（共{n_classes}个分类）...")

        # 步骤3：一对多（One-vs-Rest）多分类训练
        for idx, cls in enumerate(self.classes):  # idx循环索引  cls循环值
            print(f"\n训练类别{cls}的二分类器（将类别{cls}标记为1，其他为0）")
            # 构造当前类的二分类标签
            y_binary = np.where(self.y_train == cls, 1, 0)  # np.where(条件,满足的值标1,不满足的值标0)
            y_binary = np.asmatrix(y_binary).T  # 把普通数组转换成矩阵,转置为列向量（样本数 × 1）
            # 批量梯度下降核心循环
            for k in range(maxCycles):
                weight_vec = self.weights[idx:idx + 1, :].T  # 把权重矩阵中 “当前类的权重行” 转换成 “列向量”（特征数 × 1）
                z = self.X_train * weight_vec   # 线性输出：z = X * w
                y_prob = self.sigmoid(z)        # 转换为概率（样本数 × 1）
                error = y_prob - y_binary       # 误差：预测概率 - 真实值
                # 梯度计算与权重更新
                gradient = self.X_train.T * error  # 梯度（特征数 × 1）
                self.weights[idx:idx + 1, :] -= alpha * gradient.T  # 权重行向量更新
                # 每200轮打印训练进度（监控收敛情况）
                if k % 200 == 0:
                    y_pred = np.where(y_prob >= 0.5, 1, 0)  # 概率≥0.5预测为1,否则为0
                    acc = np.mean(y_pred == y_binary)  # 训练准确率 =  “预测结果和真实标签一致的样本数” 占总样本数的百分比
                    #NumPy里True会被当作1，False会被当作0，np.mean()就是计算这些 1 和 0 的平均值
                    print(f"类别{cls}-迭代{k}轮，训练准确率：{acc:.4f}")
        print("\n批量梯度下降训练完成！")
        return self

    # ===================== 训练函数2：随机梯度下降 =====================
    def fit2(self, X, y, alpha=0.001, maxCycles=500):
        # 步骤1：数据预处理（同fit1）
        self.X_train = np.asmatrix(X)
        self.X_train = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        self.y_train = np.asarray(y).ravel()
        self.classes = np.unique(self.y_train)
        n_classes = len(self.classes)
        n_samples, n_features = self.X_train.shape

        # 步骤2：初始化权重矩阵
        self.weights = np.ones((n_classes, n_features))
        print(f"\n开始随机梯度下降训练（共{n_classes}个分类）...")

        # 步骤3：一对多多分类训练
        for idx, cls in enumerate(self.classes):
            print(f"\n训练类别{cls}的二分类器")
            y_binary = np.where(self.y_train == cls, 1, 0)
            y_binary = np.asmatrix(y_binary).T
            # 随机梯度下降核心循环
            for k in range(maxCycles):
                # 逐个样本更新权重
                for i in range(n_samples):
                    xi = self.X_train[i:i + 1, :]  # 单个样本（行向量：1 × 特征数）
                    yi = y_binary[i:i + 1, :]      # 单个样本标签（1 × 1）
                    # 线性输出与概率计算
                    weight_vec = self.weights[idx:idx + 1, :].T
                    z = xi * weight_vec
                    y_prob = self.sigmoid(z)
                    error = y_prob - yi
                    # 单样本梯度更新
                    gradient = xi.T * error
                    self.weights[idx:idx + 1, :] -= alpha * gradient.T
                # 每100轮打印进度
                if k % 100 == 0:
                    # 计算当前类的整体训练准确率
                    weight_vec = self.weights[idx:idx + 1, :].T
                    z_all = self.X_train * weight_vec
                    y_prob_all = self.sigmoid(z_all)
                    y_pred_all = np.where(y_prob_all >= 0.5, 1, 0)
                    acc = np.mean(y_pred_all == y_binary)
                    print(f"类别{cls}-迭代{k}轮，训练准确率：{acc:.4f}")
        print("\n随机梯度下降训练完成！")
        return self

    # ===================== 预测函数 =====================
    def predict(self, X):  #模型预测：输入特征，输出每个样本的分类结果
        """
        参数：     待预测特征矩阵（无截距列，形状：样本数 × 特征数）
        返回：     一维数组：每个样本的预测类别（0/1/2）
        核心逻辑：  对每个类别计算概率，选择概率最大的类别作为预测结果
        """
        if self.weights is None:
            raise ValueError("模型尚未训练！请先调用fit1（批量）或fit2（随机）训练")
        # 预处理：添加截距列（与训练数据保持一致）
        X_mat = np.asmatrix(X)
        X_mat = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
        # 计算每个类别的概率
        probs = []
        for idx in range(len(self.classes)):
            weight_vec = self.weights[idx:idx + 1, :].T  # 权重矩阵索引+1行的列向量
            z = X_mat * weight_vec
            prob = self.sigmoid(z)
            probs.append(prob.A.ravel())  # 把“概率矩阵”（如112行1列）转换成“一维数组”（112个元素），方便后续计算
        # 把每个样本在 “所有类别上的概率” 整理好，然后对每个样本，选概率最大的那个类别作为最终答案
        probs = np.array(probs).T  # 形状：样本数 × 类别数
        y_pred = self.classes[np.argmax(probs, axis=1)]  # np.argmax(数组, axis=1):沿“行”的方向找最大值的索引（axis=1 代表按行）
        return y_pred

    # ===================== 评估函数 =====================
    def score(self, X, y): #模型评估： 计算分类准确率
        #参数：  X: 特征矩阵（无截距列）   y: 真实标签
        #返回：  准确率（0~1之间，越高越好）
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y.ravel())
        print(f"\n模型整体分类准确率：{accuracy:.4f}")
        return accuracy

    # ===================== 可视化函数 =====================
    def plot_weights(self):  # 特征权重可视化：展示每个特征对不同类别的影响程度
        """
            权重为正：特征值越大，越倾向于该类别
            权重为负：特征值越大，越不倾向于该类别
            权重绝对值越大：特征对分类的影响越强
        """
        if self.weights is None:
            raise ValueError("模型尚未训练！无法可视化权重")
        # 创建子图（每个类别一个子图）
        fig, ax = plt.subplots(1, len(self.classes), figsize=(12, 4))
        for idx, cls in enumerate(self.classes):
            # 提取当前类的特征权重（去掉截距项，只展示实际特征）
            weights = self.weights[idx, 1:]
            # 绘制柱状图
            ax[idx].bar(self.feature_names, weights)
            ax[idx].set_title(f"类别{cls}的特征权重")
            ax[idx].set_ylabel("权重值")
            # 标注权重数值，方便读取
            for i, v in enumerate(weights):
                ax[idx].text(i, v, f"{v:.2f}", ha='center', va='bottom')
        plt.tight_layout()  # 自动调整子图间距
        plt.show()


# ===================== 教学演示：完整流程 =====================
if __name__ == '__main__':
    # 1. 创建逻辑回归模型实例
    doctor = LogisticRegression()

    # 2. 加载训练数据（特征+标签）
    X_train = doctor.load_csv('../doctor/train_X.csv')  # 特征数据：气色/气味/脉相/体温
    y_train = doctor.load_csv('../doctor/train_y.csv')  # 标签数据：0=女娃 1=男孩 2=未怀孕

    # 3. 训练模型（二选一：fit1=批量梯度，fit2=随机梯度）
    #doctor.fit1(X_train, y_train, alpha=0.001, maxCycles=40000)  # 批量梯度下降
    doctor.fit2(X_train, y_train, alpha=0.04, maxCycles=500)  # 随机梯度下降

    # 4. 加载测试数据并预测
    X_test = doctor.load_csv('../doctor/test_X.csv')
    y_test = doctor.load_csv('../doctor/test_y.csv')
    y_pred = doctor.predict(X_test)

    doctor.score(X_test, y_test)

    # 5. 打印预测结果（易读格式）
    labels = ['女娃', '男孩', '没有怀孕']  # 类别标签映射
    correct_num = 0  # 正确预测数
    print("\n===== 逻辑回归诊断结果（教学演示） =====")
    print("编号,预测结果,实际结果,诊断是否准确")
    for i in range(y_test.shape[0]):
        # 比较预测值与真实值
        if y_pred[i] == y_test[i, 0]:
            correct_num += 1
            result = "准确"
        else:
            result = "错误"
        # 打印每条测试结果
        print(f"{i + 1},{labels[int(y_pred[i])]},{labels[int(y_test[i, 0])]},{result}")
    # 计算并打印整体正确率
    accuracy = correct_num / y_test.shape[0]
    print(f"\n本次诊断整体正确率: {accuracy:.4f}")

    # 6. 特征权重可视化（理解特征对分类的影响）
    doctor.plot_weights()