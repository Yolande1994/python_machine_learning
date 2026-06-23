# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# ===================== 设置Matplotlib支持中文 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===================== 1. 准备数据 =====================
# 身高体重数据：8个人，格式为 [身高(cm), 体重(kg)]
# 对应：A(160,50)、B(165,55)、C(170,65)、D(175,70)、E(180,80)、F(185,85)、G(160,65)、H(165,70)
data = np.array([
    [160, 50], [165, 55], [170, 65], [175, 70],
    [180, 80], [185, 85], [160, 65], [165, 70]
])

# 给每个人命名（方便后续看结果）
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# ===================== 2. 肘部法则：找候选K =====================
# 尝试K=1到6，计算每个K的总SSE
sse_list = []  # 存储每个K对应的总SSE
k_range = range(1, 7)  # 尝试K=1到6

for k in k_range:
    # 训练K-means模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # random_state固定，结果可复现    n_init=10：让KMeans运行10次不同的初始中心，选效果最好的一次（默认是n_init=1，容易陷入局部最优）。
    kmeans.fit(data)
    # 记录总SSE（inertia_就是总组内平方和）
    sse_list.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse_list, 'o-', color='blue')
plt.title('肘部法则：K值 vs 总SSE')
plt.xlabel('K值（分几组）')
plt.ylabel('总SSE（组内平方和）')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.show()

# ===================== 3. 轮廓系数：选最优K =====================
# 轮廓系数要求K≥2（至少分2组），所以尝试K=2到5
sil_score_list = []
k_sil_range = range(2, 6)

for k in k_sil_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)  # labels是每个点的聚类标签（属于哪一组）
    # 计算平均轮廓系数
    sil_score = silhouette_score(data, labels)
    sil_score_list.append(sil_score)
    print(f'K={k} 时，平均轮廓系数：{sil_score:.4f}')

# 绘制轮廓系数图
plt.figure(figsize=(8, 4))
plt.plot(k_sil_range, sil_score_list, 'o-', color='orange')
plt.title('轮廓系数：K值 vs 平均轮廓系数')
plt.xlabel('K值（分几组）')
plt.ylabel('平均轮廓系数（越接近1越好）')
plt.xticks(k_sil_range)
plt.grid(True, alpha=0.3)
plt.show()

# 找最优K（轮廓系数最大的K）
best_k = k_sil_range[np.argmax(sil_score_list)]
print(f'\n最优K值为：{best_k}')

# ===================== 4. 用最优K训练模型，看最终结果 =====================
# 用最优K=3训练最终模型
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(data)

# 打印每个人的聚类结果
print('\n===== 最终聚类结果 =====')
for name, label, (height, weight) in zip(names, final_labels, data):
    print(f'{name}({height}cm, {weight}kg) → 属于第{label+1}类（体型）')

# 可视化最终聚类结果
plt.figure(figsize=(8, 6))
# 定义3种颜色，对应3类体型
colors = ['red', 'green', 'purple']
# 绘制每个点，按聚类标签上色
for i in range(len(data)):
    plt.scatter(data[i, 0], data[i, 1], color=colors[final_labels[i]], s=100, alpha=0.8)
    # 标注人名
    plt.text(data[i, 0]+0.5, data[i, 1]+0.5, names[i], fontsize=10)

# 绘制聚类中心（每类体型的代表）
centers = final_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='*', s=200, label='聚类中心')

plt.title(f'K={best_k} 时的聚类结果（身高体重分体型）')
plt.xlabel('身高(cm)')
plt.ylabel('体重(kg)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()