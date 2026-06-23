#导入相关工具包
from LogisticRegression_doctor import LogisticRegression

#创建模型
doctor = LogisticRegression()

#读取训练数据 特征部分(顾客气色,气味,脉相,体温等)
X = doctor.load_csv('../doctor/train_X.csv')
#读取训练数据 实际结果部分(上述顾客真实的怀孕状态，0表示女娃，1表示男孩，2表示没有怀孕)
y = doctor.load_csv('../doctor/train_y.csv')

#训练模型
doctor.fit1(X, y, alpha=0.001, maxCycles=40000)
#doctor.fit2(X, y, alpha=0.04, maxCycles=500)

#读取测试数据
test_X = doctor.load_csv('../doctor/test_X.csv')
test_y = doctor.load_csv('../doctor/test_y.csv')

#诊断（预测）
result = doctor.predict(test_X)

#打印输出诊断结果，与实际的结果比较
#读取顾客怀孕状态的实际值(0表示女娃，1表示男孩，2表示没有怀孕)
labels=['女娃','男孩','没有怀孕']
#正确的诊断数
predictOKNum=0
i=0
print("\n编号,诊断值,实际值,")
while i < test_y.shape[0]:
    #第i个诊断结果与实际的第i个结果比较，相等表示诊断正确
    if result[i] == test_y[i, 0]:
        predictOKNum=predictOKNum+1
        okOrNo="准确"
    else:
        okOrNo="错误"
    # 打印结果
    print("%s,%s,%s,%s" % (i + 1, labels[int(result[i])], labels[int(test_y[i, 0])], okOrNo))
    i=i+1
print("诊断正确率:%s" % (predictOKNum/i))