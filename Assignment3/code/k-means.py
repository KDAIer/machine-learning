import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.linalg import det, inv
import time
from scipy.optimize import linear_sum_assignment  # 使用Scipy的匈牙利算法实现
# 通过K-means和GMM两种方法对数据进行聚类，分出10个类别（0-9手写体的分类）
# 不调用现成软件包（如sklearn）情况下，使用np，scipy等科学计算包探索 K-Means 和 GMM 这两种聚类算法的性能。
"""
读取mnist_test和mnist_train数据集
mnist_train.csv文件中包含了60000个训练样本 10个类别0~9。
mnist_test.csv文件包含10000个测试样本 10个类别。文件每一行有785个值 第一列是类别标签0~9 其余784列是手写字体的像素值。
"""

def read_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, 1:].values
    Y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    Y_test = test_data.iloc[:, 0].values

    return X_train, Y_train, X_test, Y_test

# 数据预处理
def preprocess(X_train, Y_train, X_test, Y_test):
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    Y_train = Y_train.astype(np.int32)
    Y_test = Y_test.astype(np.int32)

    return X_train, Y_train, X_test, Y_test

# K-means算法
class KMeans:
    def __init__(self, K, epoch = 50, init_method = "random"): 
        self.K = K
        self.epoch = epoch
        self.init_method = init_method
        self.train_acc = []
        self.test_acc = []
        self.center = []
        self.epoch_times = []
    def init_centers(self, X): # 不同的初始化方法，随机选择K个样本点作为初始中心或者使用kmeans++方法
        if self.init_method == "random": # 随机选择K个样本点作为初始中心
            return X[np.random.choice(X.shape[0], self.K, replace=False)]
        elif self.init_method == "kmeans++": # 使用kmeans++方法，具体做法是先随机选择一个样本点作为第一个中心，然后计算每个样本点到最近中心的距离的平方，以此为权重随机选择下一个中心
            center1 = X[np.random.choice(X.shape[0])]
            for k in range(1, self.K):
                dist = np.min(np.linalg.norm(X[:, np.newaxis] - center1[:k], axis=2)**2, axis=1) # 计算每个样本点到最近中心的距离的平方，np.linalg.norm()计算范数
                prob = dist / np.sum(dist) # 以此为权重随机选择下一个中心
                new_center = np.random.choice(X.shape[0], p=prob) 
                center1 = np.vstack([center1, X[new_center]])  
            return center1
    def fit(self, X_train, Y_train, X_test, Y_test): 
        self.centers = self.init_centers(X_train)
        for _ in range(self.epoch):
            start_time = time.time()
            dist = np.linalg.norm(X_train[:, np.newaxis] - self.centers, axis=2) # 计算每个样本点到中心的距离
            labels = np.argmin(dist, axis=1) # np.argmin()返回最小值的索引
            new_centers = np.array([X_train[labels == k].mean(axis=0) for k in range(self.K)])
            if np.all(new_centers == self.centers):
                break;
            self.centers = new_centers
            # 记录每个epoch的训练时间
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)
            train_acc = eva_acc(labels, Y_train, self.K)
            self.train_acc.append(train_acc)
            test_labels = test(X_test, self.centers)
            test_acc = eva_acc(test_labels, Y_test, self.K)
            self.test_acc.append(test_acc)
            self.center.append(np.linalg.norm(new_centers - self.centers))
            print(f"Iteration {_ + 1}, Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}")
        return labels, new_centers
    
    def print_epoch_times(self):
        total_time = sum(self.epoch_times)
        print(f"Total training time: {total_time:.2f} seconds")
        for epoch, epoch_time in enumerate(self.epoch_times, 1):
            print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")    
    
def test(X_test, centers):
    dist = np.linalg.norm(X_test[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(dist, axis=1)
    return labels    

def hungarian_algorithm(labels, Y, K):
    """
    使用匈牙利算法对聚类标签进行对齐
    """
    # 构造代价矩阵
    cost_matrix = np.zeros((K, K))
    
    # 填充代价矩阵
    for i in range(K):
        for j in range(K):
            # 对每个聚类标签 i 和真实标签 j，计算误差（例如不匹配数量）
            cost_matrix[i, j] = np.sum((labels == i) & (Y == j))
    
    # 使用匈牙利算法（线性求解最优化）来找到最小化代价的标签映射
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 使用负的代价矩阵来最大化匹配
    
    # 根据匈牙利算法找到的映射，重新映射聚类标签
    new_labels = np.zeros_like(labels)
    for i in range(K):
        new_labels[labels == i] = col_ind[i]
    
    return new_labels

def eva_acc(labels, Y, K):
    """
    计算聚类准确率
    """
    labels = np.array(labels)
    Y = np.array(Y)
    
    # 使用匈牙利算法对齐标签
    aligned_labels = hungarian_algorithm(labels, Y, K)
    
    # 计算对齐后的准确率
    accuracy = np.mean(aligned_labels == Y)
    return accuracy

def plot(train_acc, test_acc):
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(train_acc)), train_acc, label="Train accuracy")
    plt.plot(range(len(test_acc)), test_acc, label="Test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Test/Train Accuracy")
    plt.grid(True)
    plt.show()


# def plot_center(center):
#     plt.figure(figsize=(12, 4))
#     plt.plot(range(len(center)), center, label="Center")
#     plt.xlabel("Epochs")
#     plt.ylabel("Center")
#     plt.legend()
#     plt.title("Center")
#     plt.grid(True)
#     plt.show()
# 对MNIST训练集进行K-Means聚类
def main():
    X_train, Y_train, X_test, Y_test = read_data(r"D:\vs_codes\大三上\机器学习\Assignment3\material\mnist_train.csv", r"D:\vs_codes\大三上\机器学习\Assignment3\material\mnist_test.csv")
    X_train, Y_train, X_test, Y_test = preprocess(X_train, Y_train, X_test, Y_test)
    K = 10
    # 同时训练random和kmeans++两种初始化方法
    # for i in ["random", "kmeans++"]:
    print(f"Initialization method: random")
        # print(f"Initialization method: {i}")
    kmeans = KMeans(K, epoch=100, init_method = "random")
    labels, centers = kmeans.fit(X_train, Y_train, X_test, Y_test)
    kmeans.print_epoch_times()
        
    plot(kmeans.train_acc, kmeans.test_acc)
    # plot_center(kmeans.center)

        
if __name__ == "__main__":
    main()

