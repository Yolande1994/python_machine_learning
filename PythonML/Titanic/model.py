"""
模型训练与评估模块
功能：训练逻辑回归模型、评估效果、生成测试集预测结果
学习重点：模型训练流程、效果评估、特征重要性解读
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def train_and_evaluate_model(features, label):
    """
    训练模型+评估效果
    :param features: 训练集特征(X)
    :param label:    训练集标签(y)
    :return: 训练好的模型、验证集准确率
    """
    print("="*30 + " 开始模型训练与评估 " + "="*30)

    # 1. 拆分训练集和验证集（80%训练，20%验证，固定随机种子保证结果可复现）
    X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=0.2, random_state=41)
    print(f"训练集规模：{X_train.shape[0]}行 | 验证集规模：{X_val.shape[0]}行")

    # 2. 初始化并训练逻辑回归模型
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, y_train)
    print("模型训练完成！")

    # 3. 模型评估（计算准确率）
    y_pred = lr_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"验证集准确率：{val_accuracy:.4f}")

    # 4. 特征重要性分析（模型最看重哪个特征）
    feature_importance = pd.DataFrame({'特征名称': features.columns,'特征权重': lr_model.coef_[0]}).sort_values(by='特征权重', ascending=False)
    print("\n===== 特征重要性排名（权重越大，对生还预测的正向影响越强） =====")
    print(feature_importance)

    # 可选:用随机森林查看特征重要性
    clf = RandomForestClassifier()  # 训练一个随机森林模型
    clf.fit(X_train, y_train)
    # 获取特征重要性
    #importances = clf.feature_importances_
    feature_importance_clf = pd.DataFrame({'特征名称': features.columns,'特征重要性': clf.feature_importances_}).sort_values(by='特征重要性', ascending=False)
    print(feature_importance_clf)


    print("模型训练与评估完成！")
    print("="*70 + "\n")

    return lr_model, val_accuracy


def predict_test_data(model, test_features, test_path: str = "data/test.csv", save_path: str = "data/submission.csv"):
    """
    用训练好的模型预测测试集，生成Kaggle可提交的文件
    :param model: 训练好的模型
    :param test_features: 测试集特征
    :param test_path: 原始测试集路径（用来获取PassengerId）
    :param save_path: 提交文件保存路径
    """
    print("="*30 + " 开始测试集预测 " + "="*30)
    # 1. 预测测试集生还结果
    test_pred = model.predict(test_features)
    # 2. 读取原始测试集的PassengerId（Kaggle提交必须有）
    test_raw = pd.read_csv(test_path)
    # 3. 生成提交文件
    submission = pd.DataFrame({'PassengerId': test_raw['PassengerId'],'Survived': test_pred.astype(int)})
    # 4. 保存文件
    submission.to_csv(save_path, index=False) # index=False：不把DataFrame的行索引（index）作为一列写入文件
    print(f"测试集预测完成！提交文件已保存为 {save_path}，可直接上传Kaggle")
    print("="*70 + "\n")



# ---------------- 新增：XGBoost 训练与评估函数 ----------------
def train_and_evaluate_xgboost(features, label):
    """
    训练 XGBoost 模型 + 评估效果（极简版，保证运行）
    :param features: 训练集特征(X)
    :param label:    训练集标签(y)
    :return: 训练好的XGB模型、验证集准确率
    """
    print("="*30 + " 开始 XGBoost 模型训练与评估 " + "="*30)

    # 1. 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=0.2, random_state=45)
    print(f"训练集规模：{X_train.shape[0]}行 | 验证集规模：{X_val.shape[0]}行")

    # 2. 初始化 XGBoost 模型（仅保留核心参数）
    xgb_model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        eval_metric="logloss"
    )

    # 3. 纯基础训练（去掉所有早停相关参数，避免版本兼容问题）
    xgb_model.fit(X_train, y_train)
    print("XGBoost 模型训练完成！")

    # 4. 模型评估
    y_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"XGBoost 验证集准确率：{val_accuracy:.4f}")

    # 5. 特征重要性
    feature_importance_xgb = pd.DataFrame({
        '特征名称': features.columns,
        'XGBoost重要性': xgb_model.feature_importances_
    }).sort_values(by='XGBoost重要性', ascending=False)
    print("\n===== XGBoost 特征重要性排名 =====")
    print(feature_importance_xgb)

    print("XGBoost 模型训练与评估完成！")
    print("="*70 + "\n")

    return xgb_model, val_accuracy



# 单独运行这个文件时，做简单测试
if __name__ == "__main__":
    from preprocess import preprocess_data
    X, y, test_X = preprocess_data()

    # 原有逻辑回归
    #model, acc = train_and_evaluate_model(X, y)
    #predict_test_data(model, test_X)

    # 新增 XGBoost
    xgb_model, xgb_acc = train_and_evaluate_xgboost(X, y)
    predict_test_data(xgb_model, test_X, save_path="data/submission_xgb.csv")