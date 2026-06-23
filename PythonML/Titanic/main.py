# 项目主程序: 串联全流程，一键运行整个项目
from eda import draw_eda_analysis
from preprocess import preprocess_data
from model import train_and_evaluate_model, predict_test_data

# 项目完整流程
def main():
    print("泰坦尼克号生还预测项目 - 全流程启动")
    print("="*80 + "\n")

    # 步骤1：执行EDA可视化分析
    draw_eda_analysis()

    # 步骤2：数据预处理+特征工程
    train_features, train_label, test_features = preprocess_data()

    # 步骤3：模型训练与评估
    trained_model, accuracy = train_and_evaluate_model(train_features, train_label)

    # 步骤4：测试集预测，生成提交文件
    predict_test_data(trained_model, test_features)

    print("项目全流程执行完成！")
    print(f"最终模型验证集准确率：{accuracy:.4f}")

# 运行主程序
if __name__ == "__main__":
    main()