#导入相关工具包
import pandas
from sklearn.tree import DecisionTreeClassifier

#读取训练数据 特征部分(机器人传感器收集的112个顾客气色,气味,脉相,体温等特征)
X=pandas.read_csv('train_X.csv')
#读取训练数据 实际结果部分(上述112位顾客真实的怀孕状态，0表示女娃，1表示男孩，2表示没有怀孕)
y=pandas.read_csv('train_y.csv')

#创建一个机器人
doctor = DecisionTreeClassifier()
#训练机器人
doctor.fit(X, y)

#下面使用38位顾客的数据测试机器人诊断效果
#读取38位顾客的气色,气味,脉相,体温等特征数据
test_X=pandas.read_csv('test_X.csv')
#诊断！结果存放到result数组中
result=doctor.predict(test_X)

#打印输出诊断结果，与实际的结果比较
#读取38位顾客怀孕状态的实际值(0表示女娃，1表示男孩，2表示没有怀孕)
test_y=pandas.read_csv('test_y.csv')
labels=['女娃','男孩','没有怀孕']
i=0
#正确的诊断数
predictOKNum=0
print("编号,诊断值,实际值,")
while i<test_y.shape[0]:
    #第i个诊断结果与实际的第i个结果比较，相等表示诊断正确
    if result[i]==test_y.values[i,0]:
        predictOKNum=predictOKNum+1
        okOrNo="准确"
    else:
        okOrNo="错误"
    print("%s,%s,%s,%s" % (i+1,labels[result[i]],labels[test_y.values[i,0]],okOrNo))
    i=i+1
print("诊断正确率:%s" % (predictOKNum/i))