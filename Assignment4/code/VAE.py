import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision

# 手动加载MNIST数据集
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


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)   # 均值
        self.fc_logvar = nn.Linear(256, latent_dim)  # 对数方差

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# 计算VAE的损失函数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# 自定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # 归一化到 [0, 1]
        image = image.reshape(-1)  # 展平
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 20
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001

    # 手动加载MNIST数据
    images_path = 'D:/vs_codes/MNIST/train-images.idx3-ubyte'
    labels_path = 'D:/vs_codes/MNIST/train-labels.idx1-ubyte'
    images = read_mnist_images(images_path)
    labels = read_mnist_labels(labels_path)

    # 创建数据集和数据加载器
    dataset = MNISTDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # 训练VAE
    vae.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            recon_data, mu, logvar = vae(data)
            loss = vae_loss(recon_data, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 绘制损失曲线
    plt.plot(losses, label="VAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 可视化生成的图像
    vae.eval()
    with torch.no_grad():
        # 从标准正态分布中采样生成图像
        z = torch.randn(64, latent_dim).to(device)
        generated_images = vae.decode(z).view(-1, 1, 28, 28)

        # 显示生成的图像
        grid = torchvision.utils.make_grid(generated_images, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Generated Images by VAE")
        plt.show()
