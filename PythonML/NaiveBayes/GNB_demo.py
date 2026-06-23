import numpy as np
from scipy.stats import norm  # 用scipy的正态分布函数计算似然概率（大白话：算“像不像”的概率）


class GNB_demo:
    def __init__(self):
        self.classes = None  # 存储所有类别（0=健康，1=感冒）
        self.prior = {}  # 存储先验概率 P(y)
        self.mean = {}  # 存储每个类别下特征的均值
        self.var = {}  # 存储每个类别下特征的方差

    def fit(self, X, y):
        """训练模型：计算先验概率、均值、方差"""
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)  # 获取所有类别：[0, 1]

        # 遍历每个类别，计算均值、方差、先验概率
        for cls in self.classes:
            # 1. 取出该类别的所有特征数据（这里特征只有“体温”1列）
            X_cls = X[y == cls]
            # 2. 计算先验概率：该类别样本数 / 总样本数
            self.prior[cls] = len(X_cls) / len(X)
            # 3. 计算该类别下特征的均值（index=0：按列算）
            self.mean[cls] = np.mean(X_cls, axis=0)
            # 4. 计算该类别下特征的方差（index=0：按列算，ddof=1：无偏方差）
            self.var[cls] = np.var(X_cls, axis=0, ddof=1)

            # 打印训练结果（方便你看计算过程）
            print(f"===== 类别 {cls}（{'健康' if cls == 0 else '感冒'}）=====")
            print(f"先验概率 P(y={cls}) = {self.prior[cls]}")
            print(f"体温均值 = {self.mean[cls][0]:.2f}°C")
            print(f"体温方差 = {self.var[cls][0]:.4f}\n")

    def predict(self, X_test):
        """预测新样本：计算每个类别的联合概率，选最大的"""
        X_test = np.array(X_test)
        predictions = []

        for sample in X_test:
            cls_probs = {}  # 存储每个类别的联合概率

            for cls in self.classes:
                # 1. 先验概率 P(y)
                prior = self.prior[cls]
                # 2. 似然概率 P(x|y)：用正态分布计算“该体温在这个类别下出现的概率”
                # norm.pdf(样本值, 均值, 标准差) → 标准差=方差开根号
                likelihood = norm.pdf(sample, self.mean[cls], np.sqrt(self.var[cls]))
                # 3. 联合概率 = 先验概率 × 似然概率（因为只有1个特征，直接乘）
                joint_prob = prior * likelihood[0]  # [0]是因为特征只有1列

                cls_probs[cls] = joint_prob
                # 打印预测过程（方便理解）
                print(f"类别 {cls}（{'健康' if cls == 0 else '感冒'}）：")
                print(f"  先验概率 = {prior}, 似然概率 = {likelihood[0]:.6f}, 联合概率 = {joint_prob:.6f}\n")

            # 选联合概率最大的类别作为预测结果
            pred_cls = max(cls_probs, key=cls_probs.get)
            predictions.append(pred_cls)
            print(f"新样本体温 {sample[0]}°C → 预测结果：{'健康' if pred_cls == 0 else '感冒'}\n")

        return predictions


# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 1. 准备数据（和例子里完全一致）
    # X：体温（特征只有1列），y：标签（0=健康，1=感冒）
    X = [[36.5], [36.6], [36.4], [37.5], [37.6], [37.4]]
    y = [0, 0, 0, 1, 1, 1]

    # 2. 创建模型并训练
    gnb = GNB_demo()
    gnb.fit(X, y)

    # 3. 预测新样本（测试两个案例）
    print("===== 预测新样本 =====")
    X_test = [[37.0], [37.4]]  # 两个新样本：37.0°C、37.4°C
    pred = gnb.predict(X_test)