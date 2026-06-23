import math
# 准备数据格式
train_X = [[5.9,3,4.2,1.5],
           [5.8,2.6,4,1.2],
           [6.8,3,5.5,2.1],
           [4.7,3.2,1.3,0.2],
           [6.9,3.1,5.1,2.3]]
train_y = [1,1,2,0,2]
test_X = [5.8,2.8,5.1,2.4]
k = 3
#KNN核心3步: 算距离 → 找邻居 → 投票
# 步骤1:计算测试样本到所有训练样本的 “距离”
# 欧式距离(最常用)公式:  开根号[ (x1-y1)**2 + (x2-y2)**2 +...+ (xn-yn)**2 ]
def calculate_distance(train_sample,test_sample):
    distance = 0
    for i in range(len(train_sample)):
        distance += (train_sample[i] - test_sample[i])**2
    return math.sqrt(distance)
# 打印测试
for i in train_X:
    print(calculate_distance(i, test_X))

# 步骤2：找出“最近的K个训练样本”(按距离从小到大排序)
def sort_by_distance(train_X,test_X,train_y,k):
    distance_list = []
    for i in range(len(train_X)):
        distance_list.append((calculate_distance(train_X[i],test_X), train_y[i]))  #存距离,对应的怀孕结果
    print(distance_list)
    distance_list.sort()
    print(distance_list)
    return [distance_list[i][1] for i in range(k)]

# 步骤3：“投票”选出现次数最多的邻居
neighbors = sort_by_distance(train_X,test_X,train_y,k)
print(neighbors)
count = 0
for i in neighbors:
    if neighbors.count(i)>count:
        count = neighbors.count(i)
        target = i
print(f"预测结果（0:女娃/1:男孩/2:没怀孕）：{target}")