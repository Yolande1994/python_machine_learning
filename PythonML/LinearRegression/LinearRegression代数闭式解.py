import numpy as np
import matplotlib.pyplot as plt
from numpy import corrcoef
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先黑体，备用默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'

class Closed_form_Solution_of_OLS:
#基于代数闭式解(单特征)实现的线性回归模型. 具备经典的fit/predict/score等机器学习模型接口

    def __init__(self):
        self.w = None  # 初始化特征权重
        self.b = None  # 截距
        self.X_train = None
        self.y_train = None

    @staticmethod
    # 静态方法：加载文本数据，分离特征和目标值
    # 参数: fileName: 数据文件路径
    # 返回: Xlist: 特征列表  ylist: 目标值列表
    def loadDataSet(fileName):
        fr = open(fileName)
        Xlist = []
        ylist = []
        for line in fr.readlines(): # 逐行读ex0.txt里的内容
            curLine = line.strip().split('\t') # line.strip()：去掉行首尾的空格/换行；split('\t')：按Tab把一行拆成3个字符串
            Xlist.append(float(curLine[1]))
            ylist.append(float(curLine[-1]))
        fr.close()  # 增加文件关闭，避免资源泄漏
        return Xlist, ylist


    # 训练模型（代数闭式解（OLS）计算最优权重wb）
    # 参数: X:特征矩阵   y:目标值
    def fit(self, X, y):
        # 统一转换为numpy数组(一维)，确保计算兼容性
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n = len(self.X_train)
        # ---------------------- OLS代数闭式解 ----------------------
        sum_x = np.sum(self.X_train)  # 所有x的和
        sum_y = np.sum(self.y_train)  # 所有y的和
        sum_xy = np.sum(self.X_train * self.y_train)  # 所有x*y的和
        sum_x_squared = np.sum(self.X_train ** 2)     # 所有x²的和
        # 计算权重w（斜率）
        numerator_w = n * sum_xy - sum_x * sum_y        # w的分子
        denominator_w = n * sum_x_squared - sum_x ** 2  # w的分母
        self.w = numerator_w / denominator_w
        # 计算偏置b（截距）
        self.b = (sum_y - self.w * sum_x) / n
        # 打印权重信息
        print(f'最优斜率 w = {self.w}')
        print(f'最优截距 b = {self.b}')


    # 模型预测
    # 参数: X: 待预测的特征矩阵
    # 返回: y_pred: 预测值矩阵
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练！请先调用fit方法训练模型。")
        X_mat = np.array(X)
        y_pred = X_mat * self.w + self.b  # 一维np数组 * 数字斜率w + 数字截距b (广播运算:数组每个数分别计算)
        return y_pred


    #计算预测值与真实值的皮尔逊相关系数，评估模型拟合效果
    #参数: X:可选，特征矩阵（若不传则使用训练集）  y:可选，目标值（若不传则使用训练集）
    #返回: corr_coef: 核心相关系数（越接近1拟合效果越好）
    def score(self, X=None, y=None):
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练！请先调用fit方法训练模型。")
        # 优先使用传入的测试数据，无则使用训练数据
        if X is None or y is None:
            X_mat = self.X_train
            y_mat = self.y_train
        else:
            X_mat = np.array(X)
            y_mat = np.array(y)
        # 计算预测值
        y_pred = X_mat * self.w + self.b
        # 计算相关系数矩阵
        corr_matrix = corrcoef(y_pred, y_mat)
        print(f"\n预测值与真实值的相关系数矩阵：\n{corr_matrix}")
        # 返回核心相关系数
        core_corr = corr_matrix[0, 1]
        print(f"核心相关系数：{core_corr}")
        return core_corr


    #可视化训练数据和拟合直线（仅适用于单特征+截距的情况）
    def plot_fit(self):
        if self.w is None or self.b is None:
            raise ValueError("请先调用fit()方法训练模型，再绘制拟合图！")
        # 检查是否是一维数组（单特征），而非二维矩阵
        if len(self.X_train.shape) != 1:
            raise ValueError("仅支持单特征的可视化！")
        # 绘制散点图（原始数据）
        fig = plt.figure()  # 新建一个空白的画图窗口
        ax = fig.add_subplot(111)  # 在窗口里加一个“子图”（可以画图的区域）
        ax.scatter(  # 画 “原始数据的散点”
            self.X_train.flatten(),
            self.y_train.flatten(),
            label='原始数据'
        )
        # 绘制拟合直线
        X_copy = self.X_train.copy()
        X_copy.sort()  # 按特征列升序排列，让直线更美观(“从左到右连续的”，而非杂乱的折线)
        y_predict = X_copy * self.w + self.b
        ax.plot(  # 画“直线”——表示“模型预测的结果”
            X_copy,         # x轴数据
            y_predict,      # y轴数据
            'r-',           # 线条样式(r：颜色代码 -：线条样式)
            label='拟合直线'  # 图例标签
        )
        # 添加图例、标签
        ax.set_xlabel('特征值')
        ax.set_ylabel('目标值')
        ax.set_title('线性回归拟合结果')
        ax.legend()  # 显示图例(对应设置的label参数)
        plt.show()


# ---------------------- 测试封装后的模型 ----------------------
if __name__ == '__main__':  # 只有当当前文件被「直接运行」时，才执行以下缩进里的代码；如果被导入为模块，缩进里的代码会被跳过。
    # 1. 创建模型实例
    lr_model = Closed_form_Solution_of_OLS()
    # 2. 加载数据（使用类的静态方法）
    Xlist, ylist = lr_model.loadDataSet('ex0.txt')
    # 3. 训练模型（fit方法）
    lr_model.fit(Xlist, ylist)
    # 4. 预测（可选，这里用训练集演示）
    y_pred = lr_model.predict(Xlist)
    print(f"\n前5个预测值：\n{y_pred[:5]}")
    # 5. 评估模型效果
    lr_model.score()
    # 6. 可视化拟合结果
    lr_model.plot_fit()
