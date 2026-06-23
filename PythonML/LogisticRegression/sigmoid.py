# 导入numpy库（用来计算e的次方）
import numpy as np
from numpy import exp

# 定义Sigmoid函数
def sigmoid(score):
    # e：是数学里的一个常数，叫 “自然常数”，数值约等于2.71828（就像π一样，是固定值）
    # 计算 e^(-分数)：np.exp()就是计算自然常数e的次方
    exp_part = np.exp(-score)
    # 计算 1/(1 + e^(-分数))
    probability = 1 / (1 + exp_part)
    # 返回概率，同时打印中间步骤（方便看运算过程）
    print(f"===== 分数 = {score} =====")
    print(f"第一步：-分数 = {-score}")
    print(f"第二步：e^(-分数) = e^({-score}) ≈ {exp_part:.5f}")
    print(f"第三步：1 + e^(-分数) ≈ {1 + exp_part:.5f}")
    print(f"第四步：概率 = 1/(1 + e^(-分数)) ≈ {probability:.5f}\n")
    return probability

def sigmoid2(z):  # 参数z: 线性输出（X * w）
    return 1.0 / (1 + exp(-z))

# 验证3个例子
sigmoid(5)    # 分数=5
sigmoid(0)    # 分数=0
sigmoid(-5)   # 分数=-5

print(sigmoid2(-5))