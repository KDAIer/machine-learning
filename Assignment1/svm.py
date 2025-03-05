# 考虑两种不同的核函数：i) 线性核函数; ii) 高斯核函数
# 可以直接调用现成 SVM 软件包来实现
# 手动实现采用 hinge loss 和 cross-entropy loss 的线性分类模型，并比较它们的优劣

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 读取数据
# def read_data(filename):
#     data = np.genfromtx_testt(filename, delimiter=',') # 使用numpy的genfromtx_testt函数读取数据
#     x_test = data[:,1:] # 提取特征，去掉第一列标签列
#     y = data[:,0] # 第一列为标签
#     return x_test, y

# x_train, y_train = read_data('mnist_01_train.csv')
# x_test, y_test = read_data('mnist_01_test.csv')

# # 检测数据能否正常读取
# print(f"train_data: {x_train.shape}, test_data: {x_test.shape}")  # 经测试显示正常{train_data: (12666, 784), test_data: (2116, 784)}

# 读取数据
train_data = pd.read_csv('mnist_01_train.csv')
test_data = pd.read_csv('mnist_01_test.csv')

# 提取特征和标签
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# 使用线性核函数的SVM
def linear_svm(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel='linear') # 使用线性核函数
    clf.fit(x_train, y_train) # 训练模型
    clf.train_acc = clf.score(x_train, y_train) # 训练集准确率
    clf.test_acc = clf.score(x_test, y_test) # 测试集准确率
    return clf.test_acc, clf.train_acc
test_acc, train_acc = linear_svm(x_train, y_train, x_test, y_test)
print(f"train_acc_linear: {train_acc}, test_acc_linear: {test_acc}")

# 高斯核 SVM
def rbf_svm(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel='rbf') # 使用高斯核函数    
    clf.fit(x_train, y_train) # 训练模型
    clf.train_acc = clf.score(x_train, y_train) # 训练集准确率
    clf.test_acc = clf.score(x_test, y_test) # 测试集准确率
    return clf.test_acc, clf.train_acc
test_acc1, train_acc1 = rbf_svm(x_train, y_train, x_test, y_test)
print(f"train_acc_rbf: {train_acc1}, test_acc_rbf: {test_acc1}")

# Hinge loss 线性分类模型
def hinge_loss(w, x, y, alpha=0.01): # alpha为正则化系数，防止过拟合
    margin = y * (np.dot(x, w[1:]) + w[0]) # 计算间隔，y为标签，x为特征，w为权重
    loss = np.maximum(0, 1 - margin) # 计算损失
    temp = alpha * np.sum(w[1:] ** 2) # 正则化项
    return np.mean(loss) + temp # 返回平均损失

def hinge_loss_gd(w, x, y, alpha=0.01): # 梯度下降, alpha为正则化系数
    margin = y * (np.dot(x, w[1:]) + w[0])
    temp = (margin < 1).astype(int)
    # 当间隔小于1时为1，否则为0，即max(0, 1 - margin)
    b = -np.sum(y * temp) / len(y) # 计算偏置项的梯度
    w = -np.dot(x.T, y * temp) / len(y) + 2 * alpha * w[1:] # 计算权重的梯度
    return np.concatenate([[b], w]) # 返回梯度

def pred_svm(w, x):
    sc = np.dot(x, w[1:]) + w[0] # 计算得分
    return (sc > 0).astype(int) # 0和1分类

def train_svm_hinge(x_train, y_train, lr=0.001, alpha=0.01, epochs=1000):
    losses = []
    acc = []
    w = np.zeros(x_train.shape[1] + 1)  # 784+1维度的权重（包括偏置项）
    y_train = 2 * y_train - 1 # 标签转换为1和-1

    for epoch in range(epochs): # 迭代训练
        gd = hinge_loss_gd(w, x_train, y_train, alpha)
        w -= lr * gd
        # 计算损失和准确率
        loss = hinge_loss(w, x_train, y_train, alpha)
        losses.append(loss)

        y_pred = pred_svm(w, x_train) # 预测, 0和1分类, 1和-1标签
        accuracy = np.mean(y_pred == (y_train == 1)) # 计算准确率
        acc.append(accuracy)

    return w, losses, acc

# 训练SVM模型
svm_hinge, loss_hinge, acc_hinge = train_svm_hinge(x_train, y_train, lr=0.001, alpha=0.01, epochs=1000)

# 测试SVM模型
y_pred_svm_hinge = pred_svm(svm_hinge, x_test)

print(classification_report(y_test, y_pred_svm_hinge))
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm_hinge)}')

def sigmoid(x): # Sigmoid函数，注意防止溢出
    x = np.clip(x, -500, 500)  # 防止溢出
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(w, x, y): # 交叉熵损失
    sc = np.dot(x, w[1:]) + w[0] # 计算得分
    pd = sigmoid(sc) 
    pd = np.clip(pd, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(pd) + (1 - y) * np.log(1 - pd)) # 计算损失, y为标签, pd为预测值, 交叉熵损失
    return loss

def cross_entropy_gd(w, x, y): # 梯度下降
    sc = np.dot(x, w[1:]) + w[0]  
    pd = sigmoid(sc)
    b = np.mean(pd - y)
    w = np.dot(x.T, pd - y) /len(y)
    return np.concatenate([[b], w])

def pred_entropy(w, x):
    sc = np.dot(x, w[1:]) + w[0]
    temp = sigmoid(sc)
    return (temp >= 0.5).astype(int)

def train_entropy(x_train, y_train, lr = 0.01, epochs = 1000):
    losses = []
    acc = []
    w = np.zeros(x_train.shape[1] + 1) # 784+1维度的权重（包括偏置项）

    for epoch in range(epochs):
        gd = cross_entropy_gd(w, x_train, y_train)
        w -= lr * gd
        # 计算损失和准确率
        loss = cross_entropy_loss(w, x_train, y_train)
        losses.append(loss)

        y_pred = pred_entropy(w, x_train)
        accuracy = np.mean(y_pred == y_train)
        acc.append(accuracy)

    return w, losses, acc

# 训练逻辑回归模型
entropy, loss_entropy, acc_entropy = train_entropy(x_train, y_train, lr = 0.01, epochs = 1000)
# 测试逻辑回归模型
test_entropy = pred_entropy(entropy, x_test)

# 性能评估
print(classification_report(y_test, test_entropy))
print(f"Accuracy: {accuracy_score(y_test, test_entropy)}")

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_hinge, label='SVM (Hinge Loss)')
plt.plot(loss_entropy, label='Logistic Regression (Cross-Entropy Loss)')
plt.title('Loss Function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(acc_hinge, label='SVM (Hinge Loss)')
plt.plot(acc_entropy, label='Logistic Regression (Cross-Entropy Loss)')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()