import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision

# 读取MNIST数据集
def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"{file_path} 不是有效的 MNIST 图片文件")
        
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f"{file_path} 不是有效的 MNIST 标签文件")
        
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class MNISTDataset(Dataset): # 数据集类
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# GAN模型
class Generator(nn.Module): # 生成器
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1 * 28 * 28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 1, 28, 28)  # 将输出转换为图像形状
        return self.tanh(x)

class Discriminator(nn.Module): # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)  # 将输入展平
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)

# 显示原始数据集中的一部分图片
def show_original_images(images, num_images=64, nrow=8):
    images = images[:num_images]  # 只取前 num_images 张图片
    grid = utils.make_grid(images, nrow=nrow, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Original MNIST Images")
    plt.show()

# 生成训练前的初始图像
def generate_initial_images(generator, num_images=64):
    noise = torch.randn(num_images, 100, device=device)
    with torch.no_grad():
        generated_images = generator(noise)
    return generated_images

if __name__ == '__main__':
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)  # 将生成器移动到GPU
    discriminator = Discriminator().to(device)  # 将判别器移动到GPU

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_images = read_mnist_images('D:/vs_codes/MNIST/train-images.idx3-ubyte')
    train_labels = read_mnist_labels('D:/vs_codes/MNIST/train-labels.idx1-ubyte')
    train_images = train_images / 255.0

    # 将图片数据转换为PyTorch张量
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    train_images = train_images * 2.0 - 1.0  # 将数据归一化到[-1, 1]区间
    train_dataset = MNISTDataset(train_images)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    show_original_images(train_images)


    # 生成并显示初始图像
    initial_fake_images = generate_initial_images(generator)
    grid = utils.make_grid(initial_fake_images, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Initial Fake Images (Before Training)")
    plt.show()

    # 训练过程
    num_epochs = 50
    real_label = 1
    fake_label = 0

    losses_G = []
    losses_D = []


    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            # 将数据移动到GPU
            real_data = data.to(device)
        
            # 训练判别器
            discriminator.zero_grad()
        
            # 真实数据
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float32, device=device)
        
            # 将label的形状从(batch_size,)转换为(batch_size, 1)
            label = label.view(-1, 1)

            output = discriminator(real_data)
            loss_D_real = criterion(output, label)
            loss_D_real.backward()

            # 生成数据
            noise = torch.randn(batch_size, 100, device=device)
            fake_data = generator(noise)
            label.fill_(fake_label)
        
            output = discriminator(fake_data.detach())  # 不计算梯度
            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()
        
            optimizer_D.step()

            # 训练生成器
            generator.zero_grad()
        
            label.fill_(real_label)
            output = discriminator(fake_data)
            loss_G = criterion(output, label)
            loss_G.backward()
        
            optimizer_G.step()

            losses_G.append(loss_G.item())
            losses_D.append(loss_D_real.item() + loss_D_fake.item())

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                    f"Loss D: {loss_D_real.item() + loss_D_fake.item()}, Loss G: {loss_G.item()}")

        # 每个epoch后生成一张图像进行可视化
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(torch.randn(64, 100, device=device)).detach()
                grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.show()

    # 绘制损失曲线
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.legend()
    plt.show()
