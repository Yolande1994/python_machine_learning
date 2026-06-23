import numpy as np
from numpy import *
# 图形化显示
import matplotlib.pyplot as plt
from numpy import corrcoef


# 定义函数(输入: 文本文件  输出: 两个列表)   作用: 把ex0.txt里的“特征”和“目标值”分开装到两个列表里
def loadDataSet(fileName):  # fileName是要读的文件名（这里是'ex0.txt'）
    fr = open(fileName)  # 打开ex0.txt文件，fr是“文件的把手”
    numFeat = -1  # 先随便设个“特征数量”的初始值（后面会自动算）
    Xlist = []  # 装“特征”的列表
    ylist = []  # 装“目标值”的列表
    for line in fr.readlines():  # 逐行读ex0.txt里的内容
        # 把一行内容“清理+拆分”：
        # strip()：去掉行首尾的'空格/换行'; split('\t')：按Tab把一行拆成3个字符串
        curLine = line.strip().split('\t')  # 比如第一行拆成["1.000000","0.067732","3.176513"
        # 第一次读行时，算“特征数量”：
        if (numFeat == -1):
            numFeat = len(curLine) - 1  # 一行有3个元素，特征是前2个（3-1=2）
        lineArr = []  # 临时装当前行的特征
        # 把前numFeat个元素（前2个）转成数字，装到lineArr里：
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))  # 比如第一行转成[1.0, 0.067732]
        Xlist.append(lineArr)  # 把这一行的特征加到Xlist里（Xlist越来越长）
        ylist.append(float(curLine[-1]))  # 把最后一个元素（目标值）加到ylist里（比如3.176513）
    # 把装好的“特征列表”和“目标值列表”返回
    return Xlist, ylist


# 调用函数读数据
Xlist, ylist = loadDataSet('ex0.txt')
# 把列表转成 “矩阵”
X = np.asmatrix(Xlist)     # 把 Xlist 转成 “矩阵”（数学里的矩阵，能做乘法、转置）
y = np.asmatrix(ylist).T   # 把 ylist 转成 “列矩阵”（因为数学上要和 X 对应，比如 ylist 是一行，转成一列 , 对应 X 的两列）
print(X.shape,'\n',y.shape)
# .T 是numpy矩阵/数组的转置属性，作用是把数据的「行和列互换」—— 比如原是 3 行 2 列 的数据，转置后变成 2 行 3 列
# 形状: X = [[1.0, 0.5],  y = [[3.0]
#           [1.0, 1.5]]       [4.0]]

# 测试:自定义特征矩阵X和目标值矩阵y
#X=np.asmatrix([[1,0.5],[1,1.5]])
#y=np.asmatrix([[3],[4]])

# 核心公式：算线性回归的“最优权重w(能让预测误差最小的参数)”
w = (X.T * X).I * X.T * y         # 区别于手写代码：sklearn 自动加截距列、用 SVD 优化求逆，结果和手写 OLS 一致
print('第一行是截距b:',w[0],'（对应线性回归公式y = wx + b里的 b）','\n第二行是斜率w:',w[1],'（对应公式里的 w）')
# w 的形状不是固定的! w 是公式算出来的, X 有几列，w 就有几行
# w = [[b],
#      [w1],
#      [w2],
#      ...]

# plt.figure().add_subplot(111).scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0])
fig = plt.figure()        # 新建一个空白的画图窗口
ax = fig.add_subplot(111) # 在窗口里加一个“子图”（理解为“可以画图的区域”）
ax.scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0]) # 画 “原始数据的散点”
# X[:, 1]：取 X 矩阵的 “第 2 列”（因为第1列是固定的1.0，第2列才是真正的特征值）；
# flatten().A[0]：把矩阵转成普通列表（方便画图工具识别）；
# scatter(...)：画 “散点图”—— 用蓝色点表示 “原始数据”。

# 绘制模型图(“模型拟合的直线”)
xCopy = X.copy() # 复制一份X矩阵（避免修改原数据）
# 将点按照升序排列
xCopy.sort(0)    # 按第1列升序排列（让画出来的线是“从左到右连起来的”，不是乱的）
# 用模型算“预测的目标值”
yPredict = xCopy * w  # [[1, 0.067] * [[3]    =   1*3 + 0.067*1.7  =  [[y1]
#                        [1, 0.427]]   [1.7]]     1*3 + 0.427*1.7      [y2]]
# 画“直线”——用红色线表示“模型预测的结果”
ax.plot(xCopy[:, 1], yPredict)
# 显示图表
plt.show() # 执行弹出一个窗口：里面是 “蓝色散点（原始数据）+ 红色直线（模型拟合）”，能直观看到模型和数据的贴合度。

# 实际与预测之间的相关性
#corr = corrcoef(yPredict.T, y.T)  # 对比“预测值”与“真实值”
corr = corrcoef((X * w).T, y.T)
print(corr)
print(f"\n核心相关系数：{corr[0,1]}")
# X * w：用模型算所有样本的 “预测值”；
# corrcoef(...)：算 “预测值” 和 “真实值” 的相关性 —— 结果越接近 1，说明模型越准。
# corrcoef()是计算 “皮尔逊相关系数” 的函数，作用是衡量两组数据之间的 “线性相关程度”，是数据分析中判断变量关系的常用工具。



# 利用梯度下降法实现 见下一节课



# 核心公式：算线性回归的“最优权重w(能让预测误差最小的参数)”
# w = (X.T * X).I * X.T * y
# 矩阵相乘有个硬性规定：只有「前一个矩阵的列数」等于「后一个矩阵的行数」，才能相乘。否则代码会报错！
# 1. 先看 y 为啥要 .T？
# 原始 ylist 转成矩阵后是「1 行 n 列」的行向量（比如ex0.txt里有150个样本，就是1 行 150 列）；
# X 矩阵是「n 行 2 列」（150 行 2 列）；
# 如果 y 不转置，X * y 就会报错（X 的列数是 2，y 的行数是 1，2≠1）；
# 转置后 y 变成「n 行 1 列」（150 行 1 列），X 的列数（2）≠ y 的行数（150）？别急，看第二步～
# 2. 再看 X.T 为啥要加？
# 核心公式里的 X.T * X:
# X 是「n 行 2 列」，X.T 转置后变成「2 行 n 列」；
# X.T * X 就是「2 行 n 列」×「n 行 2 列」→ 结果是方阵「2 行 2 列」（符合矩阵相乘规则）；
# (X.T * X).I 是「2 行 2 列」的逆矩阵（方阵才能求逆）；
# 乘以 X.T（2 行 n 列）→ 结果是「2 行 n 列」；
# 再乘以 y.T（n 行 1 列）→ 最终得到「2 行 1 列」的 w（刚好对应 2 个参数：截距 b + 权重 w）。
