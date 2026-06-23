#训练数据
train=[ '这 不是 我的问题 本来 就 不该 我负责',
        '这个问题 解决 不了，谁都 搞不定',
        '你看，不行了 吧，我就说过 不行',
        '没有问题，我处理',
        '好的，马上 处理',
        '太好 了' ]  # CountVectorizer默认按空格来分词，所以文本必须提前分好词，否则会把整句话当成一个词
target=[0,0,0,1,1,1]
target_names=['负面','积极']

#提取文本特征：从“原始文本”到“词频稀疏矩阵”（CountVectorizer的工作流程:「先扫描所有文本，收集所有唯一词汇 → 对全局词汇按拼音排序 → 分配唯一列号 → 再逐行统计词频」）
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计器: 把文本集合转换为 “词频稀疏矩阵”，即统计每个词语在每篇文本中出现的次数。
count_vect = CountVectorizer()   # 初始化词频统计器
X_train_counts = count_vect.fit_transform(train)  # fit:先“学习”训练文本中的所有词汇（构建词汇表）| transform:再将每一条文本转换为对应的词频向量。
#打印稀疏矩阵，只显示非零值的位置和对应词频(行:文本 | 列:词在词汇表的索引 | 值:词频)
print('词频稀疏矩阵: \nshape(6,20): 6行对应train中的6条文本, 20列对应词汇表大小（词汇表有20个唯一词汇，每个词汇占1列）  Values: 词频\n',X_train_counts)
print('词汇表:\n',count_vect.vocabulary_)  # 打印构建的词汇表（字典格式，key:词语， value:词语在矩阵中的列索引）

#词频和逆向文件频率
'''
TF（Term Frequency）表示某个关键词在某文档中出现的频率
    "词频" = 某个词在文档中出现的次数/该文档的总词组数
    假设现在有一篇文章《贵州的大数据分析》，这篇文章包含了10000个词组，其中“贵州”、“大数据”、“分析”各出现100次，“的”出现500次（假设没有去除停用词），
    则通过前面TF词频计算公式，可以计算得到三个单词的词频，即：
    TF(“贵州”)= 100/10000 = 0.01
    TF(“大数据”)= 100/10000 = 0.01
    TF(“分析”)= 100/10000 = 0.01
    TF(“的”)= 500/10000 = 0.05
IDF（InversDocument Frequency）表示逆文档频率
    逆文档频率 = log(语料库文档总数/(包含该词的文档个数+1))    → (以10为底 分母+1 半平滑简化版)
    基础理论版公式: IDF(t) = log(训练集总样本数N / 包含词汇t的文档数df(t))     → (默认以自然对数ln计算,sklearn中也是如此)
    sklearn工程版: IDF(t) = log(训练集总样本数N+1 / 包含词汇t的文档数df(t)+1) + 1
    现在语料库中共存在1000篇文章，其中包含“贵州”的共99篇，包含“大数据”的共19篇，包含“分析”的共“59”篇，包含“的”共“899”篇。则它们的IDF计算如下：
    IDF(“贵州”) = 1000/(99+1) = 1.000
    IDF(“大数据”) = 1000/(19+1) = 1.700
    IDF(“分析”) = 1000/(59+1) = 1.221
    IDF(“的”) = 1000/(899+1) = 0.046
    可以发现，当某个词在语料库中各个文档出现的次数越多，它的IDF值越低，当它在所有文档中都出现时，其IDF计算结果为0，
    而通常这些出现次数非常多的词或字为“的”、“我”、“吗”等，它对文章的权重计算没有作用。
    某个词的IDF越高，说明这个词的分类作用越重要
TF-IDF = 词频×逆文档频率
    "最终权重": 只在本句频繁出现，又在全局很少见的词，权重最高。
    通过TF-IDF计算，“大数据”在某篇文章中出现频率很高，这就能反应这篇文章的主题就是关于“大数据”方向的。如果只选择一个词，“大数据”就是这篇文章的关键词。
    所以，可以通过TF-IDF方法统计文章的关键词。同时，如果同时计算“贵州”、“大数据”、“分析”的TF-IDF，将这些词的TF-IDF相加，可以得到整篇文档的值，用于信息检索。
超简短终极记忆版:
    TF:  这个词在这句话里多不多
    IDF: 这个词在所有话里稀不稀有
    TF-IDF: 这个词对这句话的重要程度
'''

#词频转TF-IDF: 单纯词频有缺陷：比如“的”这类常用词词频高，但对情感分类没意义；TF-IDF通过“词频×逆文档频率”，给“分类价值高”的词语更高的权重，给“无意义常用词”更低的权重
from sklearn.feature_extraction.text import TfidfTransformer  # TfidfTransformer: 将单纯的词频矩阵转换为更有价值的矩阵 TF → TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # fit_transform()：先学习词频矩阵的统计规律（比如文档总数、各词语出现的文档数），再转换为 TF-IDF 矩阵
print('\nTF-IDF稀疏矩阵:\n',X_train_tfidf)  # 打印TF-IDF稀疏矩阵，值不再是整数词频，而是 0~1 之间的浮点数（权重值），权重越高，该词语对这篇文本的情感分类越重要

#训练分类器
from sklearn.naive_bayes import MultinomialNB    # 使用多项式朴素贝叶斯（MultinomialNB）算法，训练一个基于 TF-IDF 特征的情感分类模型
clf = MultinomialNB().fit(X_train_tfidf, target) # 用 “TF-IDF特征矩阵” 和 “对应的情感标签” 训练模型，让模型学习 “哪些TF-IDF特征对应负面情感，哪些对应积极情感”
#训练完成后，clf就是一个可以用于预测的成熟模型

#新的待预测数据也转换为tfidf
docs_new = ['不行', '我 做不到','好的','不是 我的问题']       # 注意: 以下用的是transform()，不是fit_transform()，目的是避免新文本改变训练好的特征规则
X_new_counts = count_vect.transform(docs_new)           # 用训练阶段构建的词汇表（不是重新构建）统计新文本的词频，生成词频矩阵（保证词语索引和训练数据一致）
X_new_tfidf = tfidf_transformer.transform(X_new_counts) # 用训练阶段学习的 TF-IDF 规律，将新文本的词频矩阵转换为 TF-IDF 矩阵
print('\n预测样本的TF:\n',X_new_counts)   # "我 做不到" 不存在于词汇表里,所以词频为0,打印不显示
print('\n预测样本的TF-IDF:\n',X_new_tfidf)

#预测(用训练好的模型clf预测新文本的情感标签，并格式化输出结果)
predicted = clf.predict(X_new_tfidf)  # clf.predict(X_new_tfidf)：输入新文本的TF-IDF矩阵，返回对应的情感标签（0 或 1）
print('\n预测结果:',predicted)
for doc, category in zip(docs_new, predicted):  # 循环遍历：将“新文本”和“预测结果”一一对应，转换为易读的“负面/积极”并打印
    print ('%r => %s'%(doc, target_names[category]))



#关于稀疏矩阵 https://docs.scipy.org/doc/scipy/reference/sparse.html
from scipy.sparse import csr_matrix  # CSR: Compressed Sparse Row（按行压缩）的稀疏矩阵. 优势是高效支持按行切片、矩阵乘法等操作，非常适合文本分类这类场景
# csr_matrix((M, N),[dtype]).  (M, N)是必填参数，用来指定矩阵的形状, [dtype]是可选参数，用来指定矩阵元素的数据类型，默认是'd'（即 float64，64位浮点数）
a = csr_matrix((100000,100))  # 创建了形状为 100000行 × 100列 的空的稀疏矩阵
a[0,1] = 1    # 给第0行、第1列的元素赋值为 1
a[0,2:4] = 2  # 给第0行、列索引在[2,4)范围内的所有元素赋值为 2
a[10,0] = 10
print('\n稀疏矩阵演示: Coords:坐标(行,列)  Values:对应的值\n',a)


#分类特征编码 http://sklearn.lzjqsdd.com/modules/preprocessing.html#preprocessing-categorical-features
import numpy as np
from sklearn import preprocessing  # 包含了各种数据预处理工具，LabelEncoder,OneHotEncoder 就是其中之一
train_feature = np.array([["male"  , "from US"   , "uses Internet Explorer"],
                          ["female", "from Asia" , "uses Chrome"],
                          ["female", "from CHina", "uses Chrome"]]) # 这是一个3行×3列的特征矩阵，每一行代表一个样本，每一列代表一个分类特征
#文本转换为整数 (这些文本数据无法直接输入机器学习模型，必须先转换成数值)
train_featureT = train_feature.T
# 用来保存每个特征列对应的LabelEncoder（标签编码）实例
integer_encoded = [] # 用来保存每个特征列编码后的整数数组
label_encoders = []  # 保存编码器的意义：label_encoders列表保存了每个特征列的编码器，后续遇到新数据时，可以用同一个编码器来做相同的编码，保证编码规则一致。
for v in train_featureT:   # 循环对每一列特征做标签编码
    label_encoder = preprocessing.LabelEncoder() # 标签编码（Label Encoding）：适合处理「有序分类特征」（比如 低→中→高） | 但对「无序分类特征」（比如 US→Asia→China）可能会引入错误的顺序关系，这种场景更推荐用独热编码（One-Hot Encoding）
    integer_encoded.append(label_encoder.fit_transform(v)) # fit:先学习当前特征列的所有唯一类别（比如性别列的male、female），给每个类别分配一个唯一整数ID  |  transform:把当前列的所有文本值替换成对应的整数ID
    label_encoders.append(label_encoder)
integer_encoded = np.array(integer_encoded).T  # 再次转置，恢复原始样本行的结构
#integer_encoded = np.array([label_encoder.fit_transform(v) for v in train_feature.T]).T
print(label_encoders)
print('\n打印标签编码后的数组:\n',integer_encoded)
'''
array([[1, 2, 1],
       [0, 0, 0],
       [0, 1, 0]], dtype=int64)
'''
#用训练阶段保存好的 label_encoders 对新的测试特征数据做完全一致的标签编码，保证训练和测试数据的编码规则完全统一
test_feature = np.array([["female", "from US", "uses Chrome"],
                         ["female", "from CHina", "uses Chrome"]])
test_featureT = test_feature.T
test_integer_encoded = []  # 初始化测试数据的编码结果列表
for i in range(test_featureT.shape[0]):
    v = test_featureT[i]
    test_integer_encoded.append(label_encoders[i].transform(v))
test_integer_encoded = np.array(test_integer_encoded).T
#test_feature_integer_encoded = np.array([label_encoder.transform(v) for v in test_feature.T]).T
print('打印新测试特征:\n',test_integer_encoded)
'''
array([[0, 2, 0],
       [0, 1, 0]], dtype=int32)
'''


#转换为One-Hot稀疏矩阵(独热编码)
oneHotEncoder = preprocessing.OneHotEncoder() # OneHotEncoder是sklearn中专门用来做独热编码的工具,核心作用是解决「标签编码会引入错误顺序关系」的问题，更适合处理无序分类特征
a = oneHotEncoder.fit_transform(integer_encoded) # fit_transform(): 先学习训练数据的整数编码规则，再把它转换成独热编码矩阵
print('\n训练数据转独热编码:\n',a) # 存储元素数 = 样本数 × 特征列数（每个特征独热编码后贡献 1 个「1」）
'''  独热编码后矩阵形状为 (样本数, 7)，7 是各特征唯一类别数之和（2+3+2），所有非零元素都符合预先定义的编码规则
  (0, 1)	1.0
  (0, 4)	1.0
  (0, 6)	1.0
  (1, 0)	1.0
  (1, 2)	1.0
  (1, 5)	1.0
  (2, 0)	1.0
  (2, 3)	1.0
  (2, 5)	1.0
'''
#对测试数据的整数编码结果做独热编码
b = oneHotEncoder.transform(test_integer_encoded) # 这里只用了 transform() 而不是 fit_transform()，和标签编码的逻辑一致：必须复用训练阶段学习到的编码规则
print('测试数据转独热编码:\n',b)
'''
  (0, 0)	1.0
  (0, 4)	1.0
  (0, 5)	1.0
  (1, 0)	1.0
  (1, 3)	1.0
  (1, 5)	1.0
'''
#把稀疏矩阵转成稠密矩阵（可选）
array = a.toarray() # .toarray()可以把稀疏矩阵转换成普通的 numpy 稠密数组，方便查看完整的编码结果
print('普通numpy稠密数组:\n',array)

'''
核心知识点
1.独热编码 vs 标签编码
标签编码：把分类特征映射成整数，适合有序分类（如 低→中→高）。
独热编码：把每个分类值扩展成一个二进制特征，适合无序分类（如 US→Asia→China），避免引入错误的顺序关系。
2.稀疏矩阵的必要性
独热编码后特征维度会大幅增加（比如 3 个原始特征扩展成了 7 列），稀疏矩阵可以只存储非零元素，极大节省内存。
3.训练与测试的编码一致性
和标签编码一样，独热编码器也必须用fit_transform()处理训练数据，用transform()处理测试数据，否则会导致编码维度不匹配。
'''