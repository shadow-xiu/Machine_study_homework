import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, neighbors, ensemble
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA  # 新增PCA降维
from sklearn.metrics import normalized_mutual_info_score

# 设置随机种子
np.random.seed(42)

# 加载CIFAR-10数据集
def load_cifar10_data(data_dir):
    def unpickle(file):#加载二进制数据
        with open(file, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    train_data = []
    train_labels = []
    for batch_num in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{batch_num}")
        batch = unpickle(batch_path)
        train_data.append(batch[b"data"])
        train_labels.append(batch[b"labels"])

    X_train = np.concatenate(train_data, axis=0)#合并五次的数据
    y_train = np.concatenate(train_labels, axis=0)

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    X_test = test_batch[b"data"]
    y_test = test_batch[b"labels"]

    return X_train, y_train, X_test, y_test

# 预处理数据，标准化
def preprocess_data(X_train, X_test, use_pca=True, n_components=96):#后续可分开处理数据，不同算法的维度要求不一样
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)#3D数组展开成为2D数组1024维

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)#数据标准化

    if use_pca:
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test

# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))#10项，7个等级
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)#绘制热图
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# KNN分类器
def knn_classifier(X_train, y_train, X_test, y_test, sample_sizes):
    accuracies = []
    for size in sample_sizes:
        clf = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        clf.fit(X_train[:size], y_train[:size])
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    final_accuracy = accuracies[-1]
    print(f"\nKNN Final Accuracy: {final_accuracy * 100:.2f}%")
    plot_confusion_matrix(y_test, clf.predict(X_test), class_names=[str(i) for i in range(10)])
    return accuracies

# 随机森林分类器
def random_forest_classifier(X_train, y_train, X_test, y_test, sample_sizes):
    accuracies = []
    for size in sample_sizes:
        clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)#100棵树
        clf.fit(X_train[:size], y_train[:size])
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    final_accuracy = accuracies[-1]
    print(f"\nRandom Forest Final Accuracy: {final_accuracy * 100:.2f}%")
    plot_confusion_matrix(y_test, clf.predict(X_test), class_names=[str(i) for i in range(10)])

    return accuracies

# SVM分类器
def svm_classifier(X_train, y_train, X_test, y_test, sample_sizes):
    accuracies = []
    for size in sample_sizes:
        clf = svm.SVC(kernel='rbf', C=1, gamma='scale', random_state=42, max_iter=500)#径向基核函数，正则化系数C，自动选gamma
        clf.fit(X_train[:size], y_train[:size])
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    final_accuracy = accuracies[-1]
    print(f"\nSVM Final Accuracy: {final_accuracy * 100:.2f}%")
    plot_confusion_matrix(y_test, clf.predict(X_test), class_names=[str(i) for i in range(10)])
    return accuracies

#kmeans
def kmeans_classifier(X_train, y_train, X_test, y_test, sample_sizes):
    nmi_scores=[]
    for size in sample_sizes:
        kmeans = KMeans(n_clusters=10, random_state=32)
        train_labels_pred = kmeans.fit_predict(X_train[:size])
        nmi = normalized_mutual_info_score(y_train[:size], train_labels_pred[:size])
        nmi_scores.append(nmi)
    print(f'\nthe last NMI Score: {nmi_scores}')
    best_nmi = max(nmi_scores)
    best_size = sample_sizes[nmi_scores.index(best_nmi)]
    print(f'Best NMI: {best_nmi} for sample size: {best_size}')
    return nmi_scores

def main():
    data_dir = "D:\桌面\pythonProject2\data\cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

    X_train, X_test = preprocess_data(X_train, X_test, use_pca=True, n_components=300)

    # 定义训练样本数目列表
    sample_sizes = [5000 * i for i in range(1, 11)]  # 每次增加5000个样本直到50000个

    # 初始化模型字典
    models = {
        "Kmeans":kmeans_classifier,
        "Random_Forest":random_forest_classifier,
        "SVM":svm_classifier,
        "KNN": knn_classifier
    }
    i=0
    # 遍历每个模型
    for model_name, classifier in models.items():
        print(f"\n== {model_name} Classifier ==")
        accuracies = classifier(X_train, y_train, X_test, y_test, sample_sizes)
        if i==0:#2
            a_Kmeans=accuracies
        if i==1:
            a_Random_Forest=accuracies
        if i==2:
            a_SVM=accuracies
        if i==3:
            a_KNN=accuracies
        i=i+1

    i=0
    plt.figure(figsize=(12, 8))
    for model_name, classifier in models.items():
    # 绘制每个模型的准确度曲线
        if i==0:#3
            accuracies=a_Kmeans
        if i==1:
            accuracies=a_Random_Forest
        if i==2:
            accuracies=a_SVM
        if i==3:
            accuracies=a_KNN
        plt.plot(sample_sizes, accuracies, label=model_name)
        i=i+1
    plt.xlabel("Training Sample Size")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs. Training Sample Size")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
