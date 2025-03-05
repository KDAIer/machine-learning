import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment  # 直接导入
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import time

# 数据读取和预处理
def read_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, 1:].values
    Y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    Y_test = test_data.iloc[:, 0].values

    return X_train, Y_train, X_test, Y_test

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

# GMM 类定义
class GMM:
    def __init__(self, K, epoch=10, init_method="random", cov_type = "full", pca_dim=None):
        self.K = K  # 高斯分布的数目
        self.epoch = epoch  # 最大迭代次数
        self.init_method = init_method  # 初始化方法
        self.pca_dim = pca_dim
        self.mu = None  # 均值
        self.sigma = None  # 协方差矩阵
        self.pi = None  # 混合权重
        self.cov_type = cov_type  # 协方差类型
        self.train_acc = []
        self.test_acc = []
        self.epoch_times = []  # 用于存储每个epoch的训练时间
    # 对⾓且元素值都相等：self.sigma = np.array([np.diag(np.ones(n_features) * sigma_value)] * self.K)
    # 对⾓但对元素值不要求相等：self.sigma = np.array([np.diag(np.random.rand(n_features) * sigma_max_value)] * self.K)
    # 普通：self.sigma = np.array([np.cov(X_train.T)] * self.K)
    def fit(self, X_train, Y_train, X_test, Y_test):
        # 初始化参数
        n_samples, n_features = X_train.shape
        if self.init_method == "random":
            # 随机初始化均值、协方差和权重
            self.mu = X_train[np.random.choice(n_samples, self.K, replace=False)]
            if self.cov_type == "full":
                self.sigma = np.array([np.cov(X_train.T)] * self.K)
            elif self.cov_type == "diag":
                self.sigma = np.array([np.diag(np.diag(np.cov(X_train.T))) for _ in range(self.K)])
            elif self.cov_type == "diag_same":
                self.sigma = np.array([np.diag([np.diag(np.cov(X_train.T)).mean()] * n_features) for _ in range(self.K)])
            self.pi = np.ones(self.K) / self.K
        elif self.init_method == "kmeans++":
            # 使用kmeans++初始化参数
            center1 = X_train[np.random.choice(n_samples)]
            for k in range(1, self.K):
                dist = np.min(np.linalg.norm(X_train[:, np.newaxis] - center1[:k], axis=2)**2, axis=1)
                prob = dist / np.sum(dist)
                new_center = X_train[np.random.choice(n_samples, p=prob)]
                center1 = np.vstack([center1, new_center])
            self.mu = center1
            if self.cov_type == "full":
                self.sigma = np.array([np.cov(X_train.T)] * self.K)
            elif self.cov_type == "diag":
                self.sigma = np.array([np.diag(np.diag(np.cov(X_train.T))) for _ in range(self.K)])
            elif self.cov_type == "diag_same":
                self.sigma = np.array([np.diag([np.diag(np.cov(X_train.T)).mean()] * n_features) for _ in range(self.K)])
            self.pi = np.ones(self.K) / self.K

        # EM算法
        for _ in range(self.epoch):
            start_time = time.time()  # 记录每个epoch开始时间
            # E步骤：计算每个点属于每个簇的概率
            gamma = self.e_step(X_train)

            # M步骤：更新参数
            self.mu, self.sigma, self.pi = self.m_step(X_train, gamma)
            print(f"Epoch {_ + 1} completed")
            
                        # 记录每个epoch的训练时间
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)

            # 保存每轮的准确率
            train_pred = self.predict(X_train)
            test_pred = self.predict(X_test)
            self.train_acc.append(eva_acc(train_pred, Y_train, self.K))
            self.test_acc.append(eva_acc(test_pred, Y_test, self.K))

    def print_epoch_times(self):
        total_time = sum(self.epoch_times)
        print(f"Total training time: {total_time:.2f} seconds")
        for epoch, epoch_time in enumerate(self.epoch_times, 1):
            print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")

    def guassian(self,data,mean,cov):
        """
        计算高维高斯分布的概率密度
        :param data:用于采样的数据
        :param mean: 均值
        :param cov: 协方差
        """
        N=multivariate_normal(mean=mean,cov=cov)
        return N.pdf(data)

    # def guassian(self, X, mu, sigma):
    #     return np.exp(-0.5 * np.sum(np.dot(X - mu, np.linalg.inv(sigma)) * (X - mu), axis=1)) / (np.sqrt(np.linalg.det(sigma)) * np.power(2 * np.pi, X.shape[1]/ 2))
    
    def e_step(self, X):
        n_samples, _ = X.shape
        gamma = np.zeros((n_samples, self.K))
        for i in range(self.K):
            gamma[:, i] = self.pi[i] * self.guassian(X, self.mu[i], self.sigma[i])
        # 归一化
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma


    def m_step(self, X_train, gamma): # 更新模型参数
        n_samples, _ = X_train.shape
        N_k = np.sum(gamma, axis=0) # 分母

        for k in range(self.K):
            # 更新均值
            self.mu[k] = np.dot(gamma[:, k], X_train) / N_k[k]
            # 更新协方差
            diff = X_train - self.mu[k]
            self.sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / N_k[k]
            # 更新权重
            self.pi[k] = N_k[k] / n_samples
        return self.mu, self.sigma, self.pi

    def predict(self, X):
        # 预测每个点属于哪个簇
        gamma = self.e_step(X)
        return np.argmax(gamma, axis=1)

# # 评估聚类精度
# def clustering_accuracy(y_true, y_pred):
#     # 找到标签之间的最佳映射 构造混淆矩阵
#     contingency = np.zeros((10, 10))
#     for i in range(len(y_true)):
#         contingency[y_true[i], y_pred[i]] += 1
#     # 通过匈牙利算法找到最佳匹配
#     rows, cols = linear_sum_assignment(-contingency)
#     return contingency[rows, cols].sum() / len(y_true)

def hungarian_algorithm(labels, Y, K): # 使用匈牙利算法对聚类标签进行对齐
    # 构造代价矩阵
    cost_matrix = np.zeros((K, K)) 
    for i in range(K):
        for j in range(K):
            # 对每个聚类标签 i 和真实标签 j，计算误差（不匹配数量）
            cost_matrix[i, j] = np.sum((labels == i) & (Y == j))
    
    # 使用匈牙利算法（线性求解最优化）来找到最小化代价的标签映射
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 使用负的代价矩阵来最大化匹配
    
    new_labels = np.zeros_like(labels) # 根据匈牙利算法找到的映射，重新映射聚类标签
    for i in range(K):
        new_labels[labels == i] = col_ind[i]
    
    return new_labels

def eva_acc(labels, Y, K): # 计算准确率
    labels = np.array(labels)
    Y = np.array(Y)
    aligned_labels = hungarian_algorithm(labels, Y, K) # 使用匈牙利算法对齐标签
    accuracy = np.mean(aligned_labels == Y) # 计算对齐后的准确率
    return accuracy


def plot(train_acc, test_acc):
    plt.plot(train_acc, label="Train Clustering Accuracy")
    plt.plot(test_acc, label="Test Clustering Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Clustering Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制数据集的分类图
def plot_classification(X, Y, K):
    plt.figure(figsize=(8, 6))
    # 使用PCA降维到二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 绘制散点图，不同的类别用不同的颜色表示
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='tab10', s=20)
    plt.colorbar()  # 显示颜色条
    plt.title(f"Classification Plot (K={K})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def main():
    X_train, Y_train, X_test, Y_test = read_data(r"D:\vs_codes\大三上\机器学习\Assignment3\material\mnist_train.csv", r"D:\vs_codes\大三上\机器学习\Assignment3\material\mnist_test.csv")
    X_train, Y_train, X_test, Y_test = preprocess(X_train, Y_train, X_test, Y_test)
    
    K = 10  # 簇的数目
    for i in ["random", "kmeans++"]:
        gmm = GMM(K=K, epoch=100, init_method=i, cov_type="full", pca_dim=50)
        pca = PCA(n_components=50)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        gmm.fit(X_train, Y_train, X_test, Y_test)
        gmm.print_epoch_times()
        y_pred_train = gmm.predict(X_train)
        y_pred_test = gmm.predict(X_test)
        train_acc = eva_acc(Y_train, y_pred_train, K)
        test_acc = eva_acc(Y_test, y_pred_test, K)
        print(f"Train Clustering Accuracy: {train_acc}")
        print(f"Test Clustering Accuracy: {test_acc}")
        plot(gmm.train_acc, gmm.test_acc)
        plot_classification(X_train, Y_train, K)

    # for cov_type in ["full", "diag", "diag_same"]:
    #     print(f"Testing with covariance type: {cov_type}")
    #     gmm = GMM(K=K, epoch=60, init_method="random", cov_type=cov_type, pca_dim=50)
    #     pca = PCA(n_components=50)
    #     X_train = pca.fit_transform(X_train)
    #     X_test = pca.transform(X_test)
    #     gmm.fit(X_train, Y_train, X_test, Y_test)
    #     gmm.print_epoch_times()
        
    #     y_pred_train = gmm.predict(X_train)
    #     y_pred_test = gmm.predict(X_test)
        
    #     train_acc = eva_acc(Y_train, y_pred_train, K)
    #     test_acc = eva_acc(Y_test, y_pred_test, K)
    #     print(f"Train Clustering Accuracy: {train_acc}")
    #     print(f"Test Clustering Accuracy: {test_acc}")
        
    #     plot(gmm.train_acc, gmm.test_acc)
    #     plot_classification(X_train, Y_train, K)
    # gmm = GMM(K=K, epoch=20, init_method="random", pca_dim=50)
    # # PCA降维
    # pca = PCA(n_components=50)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # # 训练
    # gmm.fit(X_train, Y_train, X_test, Y_test)
    # # # gmm.fit(X_test)
    #     # 打印训练时间
    # gmm.print_epoch_times()
    # # # 预测
    # y_pred_train = gmm.predict(X_train)
    # y_pred_test = gmm.predict(X_test)
    
    # # # 评估
    # # # train_acc = clustering_accuracy(Y_train, y_pred_train)
    # # # test_acc = clustering_accuracy(Y_test, y_pred_test)
    # train_acc = eva_acc(Y_train, y_pred_train, K)
    # test_acc = eva_acc(Y_test, y_pred_test, K)

    # print(f"Train Clustering Accuracy: {train_acc}")
    # print(f"Test Clustering Accuracy: {test_acc}")
    # # 绘制准确率曲线
    # plot(gmm.train_acc, gmm.test_acc)

    # # 绘制分类图
    # plot_classification(X_train, Y_train, K)

if __name__ == "__main__":
    main()

