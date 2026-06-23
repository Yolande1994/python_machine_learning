"""
EDA可视化探索模块
功能：   读取原始数据，绘制核心分析图表，输出关键统计结论
学习重点：如何通过可视化快速发现数据规律、验证业务假设
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 全局配置：先设置seaborn样式，再设置中文字体（解决乱码）
sns.set_style(
    "whitegrid",
    rc={
        "font.sans-serif": ["SimHei", "Microsoft YaHei"],
        "axes.unicode_minus": False
    }
)
# 绘制核心EDA图表，输出关键统计信息    data_path:训练集数据路径
def draw_eda_analysis(data_path: str = "data/train.csv"):
    # 1. 读取原始数据
    print("="*30 + " 开始EDA可视化分析 " + "="*30)
    train_data = pd.read_csv(data_path)
    print(f"训练集数据规模：{train_data.shape[0]}行，{train_data.shape[1]}列")

    # 2. 创建画布，2行2列，4张核心图
    plt.figure(figsize=(16, 12))

    # 子图1：性别 vs 生还率（核心强特征）
    plt.subplot(2, 2, 1)
    sns.countplot(x='Sex', hue='Survived', data=train_data, palette='Set2')
    plt.title('1. 性别与生还率关系', fontsize=14)
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('人数', fontsize=12)
    plt.legend(['遇难', '生还'], fontsize=12)
    # 输出统计结论
    sex_survived = train_data.groupby('Sex')['Survived'].mean() # .groupby('Sex'):按Sex分组  ['Survived']:只看分组后的数据中「Survived 列」
    print(f"男性生还率：{sex_survived['male']:.2%} | 女性生还率：{sex_survived['female']:.2%}")

    # 子图2：船舱等级 vs 生还率（社会地位特征）
    plt.subplot(2, 2, 2)
    sns.countplot(x='Pclass', hue='Survived', data=train_data, palette='Set2')
    plt.title('2. 船舱等级与生还率关系', fontsize=14)
    plt.xlabel('船舱等级 (1=头等, 2=二等, 3=三等)', fontsize=12)
    plt.ylabel('人数', fontsize=12)
    plt.legend(['遇难', '生还'], fontsize=12)
    # 输出统计结论
    pclass_survived = train_data.groupby('Pclass')['Survived'].mean()
    print(f"头等舱生还率：{pclass_survived[1]:.2%} | 二等舱：{pclass_survived[2]:.2%} | 三等舱：{pclass_survived[3]:.2%}")

    # 子图3：家庭人数 vs 生还率（衍生特征灵感）
    plt.subplot(2, 2, 3)
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1  # +1是自己
    sns.countplot(x='FamilySize', hue='Survived', data=train_data, palette='Set2')
    plt.title('3. 家庭总人数与生还率关系', fontsize=14)
    plt.xlabel('家庭总人数（含自己）', fontsize=12)
    plt.ylabel('人数', fontsize=12)
    plt.legend(['遇难', '生还'], fontsize=12)

    # 子图4：年龄分布 vs 生还率（分箱特征灵感）
    plt.subplot(2, 2, 4)                           #  multiple='stack':堆叠  bins=20：将年龄划分为20个区间
    sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack', bins=20, palette='Set2')
    plt.title('4. 年龄分布与生还率关系', fontsize=14)
    plt.xlabel('年龄', fontsize=12)
    plt.ylabel('人数', fontsize=12)
    plt.legend(['遇难', '生还'], fontsize=12)

    # 3. 保存图片+显示
    plt.tight_layout() # 自动调整子图间距，避免重叠
    plt.savefig('data/eda_analysis.png', dpi=300, bbox_inches='tight') # bbox_inches='tight' 是保存的时候让图片更紧凑，没有多余白边
    plt.show()
    print("EDA分析完成！图表已保存为 data/eda_analysis.png")
    print("="*70 + "\n")

# 单独运行这个文件，会执行EDA分析
if __name__ == "__main__":
    draw_eda_analysis()