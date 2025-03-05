import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据处理，加载数据
def load_data(dir):
    X_train = []
    Y_train = []
    # 读取训练数据
    for i in range(1, 6):
        file_path = os.path.join(dir, f'data_batch_{i}')
        print(f"Loading file: {file_path}") # 打印加载的文件路径
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes') # pickle.load()函数可以将文件中的数据解析为一个python对象
            X_train.append(dict[b'data']) 
            Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0) # 沿着指定轴连接数组, concatenate函数的第一个参数是一个列表，表示要连接的数组，第二个参数axis表示连接的轴
    
    """
    在 CIFAR-10 数据集中 dict[b'data'] 通常是一个已经预处理成 NumPy 数组格式的数据。
    因此 直接赋值 X_test = dict[b'data'] 会使得 X_test 成为 NumPy 数组
    """

    # 读取测试数据
    test_file_path = os.path.join(dir, 'test_batch')
    print(f"Loading file: {test_file_path}")
    with open(test_file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X_test = dict[b'data']
        Y_test = dict[b'labels']
    
    return X_train, Y_train, X_test, Y_test

# 数据预处理，进行归一化

def preprocess(X_train, Y_train, X_test, Y_test):
    X_train = X_train.reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0 # 将数据转换为3通道的图片格式，并归一化，astype转换数据类型
    X_test = X_test.reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0 # reshape参数-1表示自动计算该维度的大小，3表示通道数，32表示图片的长宽
    Y_train = np.array(Y_train) # 将list转换为numpy数组
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

# 定义数据集类
class CIFAR10Dataset(Dataset): # pytorch的数据集类需要继承torch.utils.data.Dataset
    def __init__(self, X, Y):
        self.data = torch.from_numpy(X).float() # 将数据转换为 PyT
        self.label = torch.from_numpy(Y).long() # 将标签转换为 PyTorch 的 LongTensor 类型

    def __len__(self):
        return self.data.shape[0] # 返回数据集的大小

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] # 返回数据和标签
    
# softmax分类器
# class Softmax(nn.Module):
#     def __init__(self):
#         super(Softmax, self).__init__() # 调用父类的构造函数
#         self.fc = nn.Linear(3 * 32 * 32, 10) # 全连接层，输入维度是 3 * 32 * 32，输出维度是 10，对应于 10 个类别
#     def forward(self, x):
#         x = x.view(x.size(0), -1) # 将输入数据展平成一维向量
#         x = self.fc(x) # 全连接层
#         x = F.log_softmax(x, dim=1) 
#         return x

# MLP原始模型
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(3 * 32 * 32, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# #MLP对比训练
# class MLP(nn.Module):
#     def __init__(self, layer_sizes):
#         super(MLP, self).__init__()
#         layers = []
#         input_size = 3 * 32 * 32
#         for size in layer_sizes:
#             layers.append(nn.Linear(input_size, size))
#             layers.append(nn.ReLU())
#             input_size = size
#         layers.append(nn.Linear(input_size, 10))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)

# CNN模型
# LeNet with max_pooling
class max_pooling(nn.Module):
    def __init__(self):
        super(max_pooling, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# LeNet with avg_pooling
class avg_pooling(nn.Module):
    def __init__(self):
        super(avg_pooling, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# LeNet with 4 convolution layers (adjusted)
# class LeNet4Conv(nn.Module):
#     def __init__(self):
#         super(LeNet4Conv, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5, padding=2)  # 添加padding，保持图像大小
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5, padding=2)  # 添加padding，保持图像大小
#         self.conv3 = nn.Conv2d(16, 32, 5, padding=2)  # 添加padding，保持图像大小
#         self.conv4 = nn.Conv2d(32, 64, 5, padding=2)  # 添加padding，保持图像大小
#         self.fc1 = nn.Linear(64 * 2 * 2, 120)  # 调整输入到全连接层的大小
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # 卷积层1 -> 激活函数 -> 池化层
#         x = self.pool(F.relu(self.conv2(x)))  # 卷积层2 -> 激活函数 -> 池化层
#         x = self.pool(F.relu(self.conv3(x)))  # 卷积层3 -> 激活函数 -> 池化层
#         x = self.pool(F.relu(self.conv4(x)))  # 卷积层4 -> 激活函数 -> 池化层
#         x = x.view(-1, 64 * 2 * 2)  # 展平，调整大小以适应全连接层
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# # 定义不同滤波器数量的LeNet模型
# class LeNet3Conv_Filters(nn.Module):
#     def __init__(self, filters):
#         super(LeNet3Conv_Filters, self).__init__()
#         self.conv1 = nn.Conv2d(3, filters[0], 5, padding=2)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(filters[0], filters[1], 5, padding=2)
#         self.conv3 = nn.Conv2d(filters[1], filters[2], 5, padding=2)
#         self.fc1 = nn.Linear(filters[2] * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, self.conv3.out_channels * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# 训练模型函数
def train_model(model, device, train_loader, optimizer, epoch, loss_history, accuracy_history):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算训练集准确率
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(f'Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tAccuracy: {accuracy * 100:.2f}%')

# 测试模型函数
def test_model(model, device, test_loader, accuracy_history, test_loss_history):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum').item()
            test_loss += loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    accuracy_history.append(accuracy)
    test_loss_history.append(test_loss)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)')


# 训练模型并绘制结果
def plot_results(models, model_names, train_loaders, test_loaders, device, epochs=100):
    # 定义图形
    plt.figure(figsize=(12, 5))
    
    accuracy_train_all = []  # 用来存储所有模型的训练准确率
    accuracy_test_all = []  # 用来存储所有模型的测试准确率
    loss_train_all = []  # 用来存储所有模型的训练损失
    loss_test_all = []  # 用来存储所有模型的测试损失

    for model, model_name, train_loader, test_loader in zip(models, model_names, train_loaders, test_loaders):
        optimizer = optim.Adam(model.parameters())

        accuracy_history_train = []  # 训练准确率
        accuracy_history_test = []  # 测试准确率
        loss_history_train = []  # 训练损失
        loss_history_test = []  # 测试损失

        for epoch in range(1, epochs + 1):
            # 训练模型
            train_model(model, device, train_loader, optimizer, epoch, loss_history_train, accuracy_history_train)
            # 测试模型
            test_model(model, device, test_loader, accuracy_history_test, loss_history_test)

        accuracy_train_all.append(accuracy_history_train)
        accuracy_test_all.append(accuracy_history_test)
        loss_train_all.append(loss_history_train)
        loss_test_all.append(loss_history_test)

        # 绘制训练和测试的准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(accuracy_history_train) + 1), accuracy_history_train, label=f'{model_name} Train')
        plt.plot(range(1, len(accuracy_history_test) + 1), accuracy_history_test, label=f'{model_name} Test')

        # 绘制训练损失和测试损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(loss_history_train) + 1), loss_history_train, label=f'{model_name} Train')
        plt.plot(range(1, len(loss_history_test) + 1), loss_history_test, label=f'{model_name} Test')

    # 设置图表属性
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

# 主函数（滤波器数量对比）
# def main():
#     data_dir = r'D:\vs_codes\大三上\机器学习\Assignment2\material\data'
#     X_train, Y_train, X_test, Y_test = load_data(data_dir)
#     X_train, Y_train, X_test, Y_test = preprocess(X_train, Y_train, X_test, Y_test)

#     trainDataset = CIFAR10Dataset(X_train, Y_train)
#     testDataset = CIFAR10Dataset(X_test, Y_test)

#     trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
#     testLoader = DataLoader(testDataset, batch_size=256, shuffle=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 定义不同滤波器数量的模型
#     filters_list = [
#         [6, 16, 32],  # 标准LeNet模型
#         [8, 16, 32],  # 改变第一层滤波器数量
#         [16, 32, 64],  # 增加滤波器数量
#         [32, 64, 128]  # 更高的滤波器数量
#     ]

#     # 定义模型
#     models = [LeNet3Conv_Filters(filters).to(device) for filters in filters_list]
#     model_names = [f'LeNet with Filters {filters}' for filters in filters_list]

#     # 训练并绘制结果
#     plot_results(models, model_names, [trainLoader]*len(models), [testLoader]*len(models), device, epochs=100)

# if __name__ == '__main__':
#     main()

# 主函数(pooling对比)
def main():
    data_dir = r'D:\vs_codes\大三上\机器学习\Assignment2\material\data'
    X_train, Y_train, X_test, Y_test = load_data(data_dir)
    X_train, Y_train, X_test, Y_test = preprocess(X_train, Y_train, X_test, Y_test)

    trainDataset = CIFAR10Dataset(X_train, Y_train)
    testDataset = CIFAR10Dataset(X_test, Y_test)

    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=256, shuffle=False, drop_last=True)


    device = torch.device("cuda")

    models = [max_pooling().to(device), avg_pooling().to(device)]
    model_names = ['max_pooling', 'avg_pooling']
    train_loaders = [trainLoader, trainLoader]
    test_loaders = [testLoader, testLoader]

    plot_results(models, model_names, train_loaders, test_loaders, device, epochs=100)

if __name__ == '__main__':
    main()