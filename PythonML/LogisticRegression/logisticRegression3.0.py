import numpy as np
from numpy import *

# 加载文本文件中数据(已知最后一列是y)
def loadDataSet(fileName):
    fr = open(fileName)  # 打开文件
    numFeat = -1  # 特征数（先设为-1，后面再确定）
    Xlist = []   # 存特征的列表
    ylist = []   # 存标签的列表（每一行的最后一列是y）
    for line in fr.readlines():  # 逐行读取文件内容
        curLine = line.strip().split('\t')  # 处理当前行：去掉前后空格，按制表符（\t）分割成多个字符串
        if (numFeat == -1):
            numFeat = len(curLine) - 1
        lineArr = []  # 临时存当前行的特征
        # 将特征转换为浮点类型,存到lineArr（特征）
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        # 添加到列表中
        Xlist.append(lineArr)
        ylist.append(float(curLine[-1]))
    # 返回列表
    return Xlist, ylist

# 定义了一个全局变量weights（后面逻辑回归预测会用到），暂时设为None（空）
weights = None

# 线性回归的预测方法
def predict(X, weights):
    # 计算X和weights的矩阵点积（即线性回归的预测公式：y = w1*x1 + w2*x2 + ... + wn*xn）
    return np.dot(X, weights)

'''
该代码不是 “工业界的逻辑回归”，只是 “用线性回归的方式硬套二分类”，结果是不可靠的:
1.不是真正的逻辑回归（没有 sigmoid + 对数损失）；
2.用 MSE + 线性预测做二分类，会出现梯度消失、分类边界错误、概率无意义等致命问题；
3.能运行只是小数据集下的巧合，换个数据集就会失效
'''
# 用 “批量梯度下降法” 训练线性回归模型，求出最优的权重（系数）
def fit(X, y):
    m, n = shape(X)  # m是样本数（200）,n是特征数（2）,X是特征矩阵
    print(shape(X))
    print(shape(y))
    alpha = 0.001  # 步长（学习率）：控制每次更新权重的幅度
    maxCycles = 500  # 迭代次数：模型要训练多少轮
    # 初始化权重（所有特征的权重先设为 1）
    weights = ones((n, 1))  # ones((n,1))：numpy 的函数，生成n行1列的矩阵，所有元素都是 1；
    # 循环训练
    for i in range(maxCycles):
        # 用当前权重预测y（调用predict函数，计算X和weights的矩阵乘法: yp = w1*x1 + w2*x2 + ... + wn*xn ）
        yp = predict(X, weights)
        #yp = np.dot(X, weights)
        # 预测值和真实值的误差：y - yp
        error = yp - y  # 如果误差是y - yp，下面就用 “加”
        # 梯度下降更新权重：w = w + α * X.T * error（数学公式，核心是用误差调整权重）
        weights = weights - alpha * X.T * error
        '''
        gradient = (1/m) * X.T * error          # 批量梯度下降必须除以样本数m
        weights = weights - alpha * gradient  
        '''
    return weights


# 逻辑回归的预测方法
def predictLogic(X):
    # 先计算线性部分（X和weights的点积）
    y = np.dot(X, weights)
    # 用sigmoid函数把结果映射到0~1之间（逻辑回归的核心：把线性输出转成概率）
    y = 1.0 / (1 + exp(-y))
    ''''
    # 注释掉的部分是“把概率转成0/1分类”：概率<0.5输出0，否则输出1
    if y<0.5:
        return 0
    else:
        return 1;
    '''


# 从文件中取得数据(数组形式)
Xlist, ylist = loadDataSet('ex0.txt')
# 转换为矩阵
X, y = np.asmatrix(Xlist), np.asmatrix(ylist).T
# 训练
w = fit(X, y)


# 随机梯度下降法(和fit的“批量梯度下降”区别是：批量是用所有样本算误差，随机是用单个样本算误差（训练更快）)
def stocGradAscent0(Xarray, ylist):
    m, n = shape(Xarray)  # m是样本数,n是特征数,X是特征数组
    alpha = 0.01  # 步长
    weights = ones(n)  # 初始系数全部为1（这里是一维数组）
    maxCycles = 500  # 迭代次数
    for k in range(maxCycles):  # 循环训练maxCycles次
        # 逐个样本更新权重
        for i in range(m):
            # 用第i个样本预测y
            y0 = predict(Xarray[i],weights)  # 当前值，就是点积，w1*x1+w2*x2+...+wn*xn
            # 第i个样本的误差：真实值-预测值
            error = ylist[i] - y0
            # 更新权重：w = w + α * error * 第i个样本的特征
            weights = weights + alpha * error * Xarray[i]
    print(weights)
    return weights

# 把Xlist转成numpy数组，传入随机梯度下降函数训练
w = stocGradAscent0(array(Xlist), ylist)


# 如果结果是二值型，则可以用一个映射函数将结果映射到 0和1之间，比如sigmoid函数，这样梯度下降法就可以用来处理分类的问题了，
# 即回归分类法
def sigmoid(x):
    # sigmoid函数公式：把任意实数转成0~1之间的数（用于逻辑回归的分类）
    return 1.0 / (1 + exp(-x))

# 这份代码是线性回归 + 逻辑回归的基础实现：
# 1.从文本文件读数据，分成特征和标签；
# 2.用 “批量梯度下降” 和 “随机梯度下降” 两种方法训练线性回归模型；
# 3.提供了逻辑回归的预测（基于 sigmoid 函数）。