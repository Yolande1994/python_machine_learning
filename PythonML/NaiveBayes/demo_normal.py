import numpy as np
"""朴素贝叶斯多分类器,支持多分类多样本"""
X = [[1,0],
     [1,0],
     [1,1],
     [0,0],
     [0,0],
     [1,1],]
y = [1,1,1,1,0,0]

class MultinomialNB:
    def __init__(self):
        self.classes = None
        self.prior = {}           # 存先验概率 {类别: 概率}
        self.feature_probs = {}   # 存特征概率 {类别: {特征索引: {特征值: 概率}}}

    def fit(self, X , y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)         # 获取类别
        n_samples, n_features = X.shape     # n_samples:总样本数(6)  n_features:总特征数(2)
        # 计算先验概率 P(y)
        for cls in self.classes:
            count = np.sum(y == cls)          # 计算每个类别的样本数
            self.prior[cls] = count / len(y)  # 类别样本数/总样本数

        # 计算每个类别下各特征的特征概率 P(x|y)
        for cls in self.classes:
            X_cls = X[y == cls]  # 取出各类别的所有特征数据(X和y靠样本索引一一对应)
            self.feature_probs[cls] = {}
            print(f'\n类别y为{cls}的数组:\n{X_cls}\n')

            # 遍历每个特征
            for idx in range(n_features):
                print(f'正在遍历特征x{idx+1}:')
                # 取出各特征所在列
                feat_values = X_cls[:, idx]    # 例 X_cls[:, 0]  :取所有行, 0取第一列
                print(f'{X_cls}里的第{idx+1}列,即x{idx+1}是:{feat_values}')
                # 全局特征的唯一值
                feat_unique = np.unique(X)  # [0,1]

                # 计算该特征各取值的特征概率
                self.feature_probs[cls][idx] = {}
                for val in feat_unique:
                    # 该特征取val的样本数
                    val_count = np.sum(feat_values == val)
                    print(f'其中x{idx+1}为{val}的个数:{val_count}')
                    # 特征概率（特征取值次数/该类别总样本数）
                    self.feature_probs[cls][idx][val] = val_count / len(X_cls)
                    print(f'其中x{idx+1}为{val}的概率:{self.feature_probs[cls][idx][val]}')
                print()


    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        # 遍历每个待预测样本
        for sample in X_test:
            cls_probs = {}
            # 算每个类别的联合概率
            for cls in self.classes:
                prior = self.prior[cls]  # 每个类别的先验概率
                print(f'\n类别为{cls}的先验概率      :{prior}')
                for idx, val in enumerate(sample):
                    print(f'特征为x{idx+1},值为{val}的特征概率:{self.feature_probs[cls][idx].get(val, 0)}')
                    prior = prior * self.feature_probs[cls][idx].get(val, 0)  # 乘每个特征的特征概率
                    #self.feature_probs[cls][idx]：取出字典类别cls下，特征索引idx的特征概率
                    #get(val, 0)：如果该特征值val在训练时出现过，就取对应的特征概率；如果没出现过（无平滑），就返回 0
                cls_probs[cls] = prior # 存每个类别的联合概率
                print(f'判断y为{cls}时的联合概率   :{cls_probs[cls]}')

            # 选择概率最大的类别作为预测结果
            predictions.append(max(cls_probs, key=cls_probs.get))

        return predictions



# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # X = [是否发烧, 是否血压高]，y = [诊断结果(1=亚健康, 0=抑郁症)]
    X = [[1, 0],
         [1, 0],
         [1, 1],
         [0, 0],
         [0, 0],
         [1, 1]]
    y = [1, 1, 1, 1, 0, 0]
    # 创建分类器实例（无平滑，无需传alpha）
    nb = MultinomialNB()
    # 训练模型
    nb.fit(X, y)
    # 预测新样本：
    X_test = [[1, 0]]
    pred = nb.predict(X_test)
    # 输出结果
    print(f"\n\n\n新样本特征：发烧=1，血压正常=0")
    print(f"预测结果：{'亚健康' if pred[0] == 1 else '抑郁症'}")
    print(f"先验概率（亚健康）：{nb.prior[1]:.4f}")
    print(f"先验概率（抑郁症）：{nb.prior[0]:.4f}")
    # 额外输出特征概率
    print("===== 特征概率验证 =====")
    print(f"亚健康(y=1)下，发烧(x1=1)的概率：{nb.feature_probs[1][0][1]:.4f}")
    print(f"亚健康(y=1)下，血压正常(x2=0)的概率：{nb.feature_probs[1][1][0]:.4f}")


'''
# 类别cls=1（亚健康）、cls=0（抑郁症）
self.feature_probs = {
    1: {  # 类别1（亚健康）
        0: {  # 特征索引0（是否发烧）
            1: 0.75,  # 特征值1（发烧）的特征概率：3/4
            0: 0.25   # 特征值0（不发烧）的特征概率：1/4
        },
        1: {  # 特征索引1（是否血压高）
            0: 0.75,  # 特征值0（血压正常）的特征概率：3/4
            1: 0.25   # 特征值1（血压高）的特征概率：1/4
        }
    },
    0: {  # 类别0（抑郁症）
        0: {  # 特征索引0（是否发烧）
            1: 0.5,   # 特征值1（发烧）的特征概率：1/2
            0: 0.5    # 特征值0（不发烧）的特征概率：1/2
        },
        1: {  # 特征索引1（是否血压高）
            0: 0.5,   # 特征值0（血压正常）的特征概率：1/2
            1: 0.5    # 特征值1（血压高）的特征概率：1/2
        }
    }
}
层级	    代码片段	                            含义	                         例子（cls=1, idx=0, val=1）
第一层	self.feature_probs[cls]	            取某个类别的所有特征概率	         self.feature_probs[1] → 亚健康的所有特征特征概率
第二层	self.feature_probs[cls][idx]	    取该类别下某个特征的特征概率	     self.feature_probs[1][0] → 亚健康下 “是否发烧” 的特征概率
第三层	self.feature_probs[cls][idx][val]	取该类别下该特征取某个值的特征概率	 self.feature_probs[1][0][1] → 亚健康下 “发烧（1）” 的概率 0.75
'''