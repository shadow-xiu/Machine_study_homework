from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA  # 新增PCA降维

# 设置随机种子
np.random.seed(42)

# 加载CIFAR-10数据集
def load_cifar10_data(data_dir):
    def unpickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    train_data = []
    train_labels = []
    for batch_num in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{batch_num}")
        batch = unpickle(batch_path)
        train_data.append(batch[b"data"])
        train_labels.append(batch[b"labels"])

    X_train = np.concatenate(train_data, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    X_test = test_batch[b"data"]
    y_test = test_batch[b"labels"]

    return X_train, y_train, X_test, y_test

# 预处理数据，标准化
def preprocess_data(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

# 优化后的随机森林分类器
def optimized_random_forest_classifier(X_train, y_train, X_test, y_test, sample_sizes):
    # 准备记录准确率
    accuracies = []

    # 降维和特征选择
    pca = PCA(n_components=96, random_state=42)  # 降到100维
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 随机森林超参数调优：通过网格搜索找最优超参数
    param_grid = {
        'n_estimators': [100, 150],  # 选择不同的树数量
        'max_depth': [None, 10, 20],  # 树的最大深度
        'min_samples_split': [2, 5],  # 最小样本分裂数
        'max_features': ['sqrt', 'log2'],  # 特征选择的策略
    }
#默认的参数组合是：
    # n_estimators：100
    # max_depth：无限制（None）
    # min_samples_split：2
    # max_features：auto，

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)  # n_jobs=-1：使用所有可用的CPU核心
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=1, verbose=2)  # 将n_jobs设为1以避免并行问题

    # 训练并找到最佳超参数
    grid_search.fit(X_train_pca, y_train)
    best_rf = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")

    # 评估随机森林分类器
    for size in sample_sizes:
        best_rf.fit(X_train_pca[:size], y_train[:size])
        y_pred = best_rf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    final_accuracy = accuracies[-1]
    print(f"\nOptimized Random Forest Final Accuracy: {final_accuracy * 100:.2f}%")
    plot_confusion_matrix(y_test, best_rf.predict(X_test_pca), class_names=[str(i) for i in range(10)])

    return accuracies

# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 主程序
def main():
    # 确保路径没有中文字符，并且路径正确
    data_dir = "E:\cifar-10-batches-py"  # 确保路径没有中文字符
    if not os.path.exists(data_dir):
        print(f"Error: The specified directory {data_dir} does not exist.")
        return

    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

    # 预处理数据，标准化并降维
    X_train, X_test = preprocess_data(X_train, X_test)

    # 定义训练样本数目列表
    sample_sizes = [5000 * i for i in range(1, 11)]  # 每次增加5000个样本直到50000个

    # 使用优化后的随机森林分类器
    print("\n== Optimized Random Forest Classifier ==")
    accuracies = optimized_random_forest_classifier(X_train, y_train, X_test, y_test, sample_sizes)

    # 绘制准确度随样本规模变化的曲线
    plt.figure(figsize=(12, 8))
    plt.plot(sample_sizes, accuracies, label="Optimized Random Forest", color='b')
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.title("Optimized Random Forest Accuracy vs. Training Sample Size")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
