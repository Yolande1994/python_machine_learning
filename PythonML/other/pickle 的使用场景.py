# pickle 本质是 Python 对象的「持久化工具」，只要需要把内存里的 Python 对象（不是纯文本/数值）保存到磁盘、或在程序间传输，都能用它。下面说几个最常用的场景：
# (以下代码只作演示)

# 一、机器学习 / 数据科学（最核心场景）
# 1.保存训练好的模型
# 训练完 kNN/SVM/ 简单的 sklearn 模型后，不用每次运行都重新训练（耗时），可以把模型对象直接存成 pickle 文件，下次用的时候直接加载：
import pickle
from sklearn.linear_model import LogisticRegression
# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)
# 保存模型到磁盘
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
# 后续使用：直接加载模型（无需重新训练）
with open("trained_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
loaded_model.predict(X_test)  # 直接预测
# 2.保存预处理后的数据集
# 比如对 CIFAR-10 做了归一化、特征提取后，把处理好的 numpy 数组（X_train/y_train）存成 pickle，下次直接加载就能用，不用重复预处理。


# 二、缓存耗时计算的结果
# 如果你的代码里有耗时的计算（比如复杂的数值运算、爬虫数据解析），可以把计算结果存成 pickle，避免重复计算：
import pickle
import time
# 模拟耗时计算
def heavy_calculation():
    time.sleep(3)  # 模拟3秒耗时
    return [i*2 for i in range(10000)]
# 先检查是否有缓存文件，有就直接加载，没有就计算+保存
if os.path.exists("calc_result.pkl"):
    with open("calc_result.pkl", "rb") as f:
        result = pickle.load(f)
else:
    result = heavy_calculation()
    with open("calc_result.pkl", "wb") as f:
        pickle.dump(result, f)


# 三、保存复杂的 Python 对象
# JSON 只能存列表、字典、字符串等基础类型，没法存 numpy 数组、自定义类实例、函数等，但 pickle 可以：
import pickle
import numpy as np
# 自定义类
class MyData:
    def __init__(self, name, arr):
        self.name = name
        self.arr = arr
# 复杂对象（包含自定义类+numpy数组）
obj = MyData("测试数据", np.array([1,2,3]))
# 序列化保存
with open("complex_obj.pkl", "wb") as f:
    pickle.dump(obj, f)
# 反序列化还原（对象属性、类型都不变）
with open("complex_obj.pkl", "rb") as f:
    loaded_obj = pickle.load(f)
print(loaded_obj.name)  # 输出：测试数据
print(loaded_obj.arr)   # 输出：[1 2 3]


# 四、程序间 / 进程间传输 Python 对象
# 比如用多进程处理数据时，主进程可以把复杂对象（如配置字典、数据集）序列化后传给子进程，子进程反序列化就能用（比手动拆分成基础类型更方便）。


# 注意（新手避坑）
# pickle 只能在 Python 里用，不能跨语言（比如 Java 读不了 Python 的 pickle 文件）；
# 不要读取来源不明的 pickle 文件（可能有安全风险）；
# 不同 Python 版本的 pickle 可能不兼容（就像代码里处理 CIFAR-10 时要适配 Python2/3）。

# 总结
# pickle 最常用在机器学习保存模型 / 缓存数据，避免重复训练 / 计算；
# 适合保存复杂 Python 对象（numpy 数组、自定义类等），弥补 JSON 的不足；
# 核心价值是「原样保存 / 还原 Python 对象」，但仅限 Python 生态内使用。

# 简单说：只要你想把 Python 里的 “东西”（不是纯文本）存起来下次用，又不想手动转格式，就可以用 pickle。