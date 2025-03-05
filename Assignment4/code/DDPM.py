import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import struct
import os

# 读取MNIST图像文件
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

# 读取MNIST标签文件
def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f"{file_path} 不是有效的 MNIST 标签文件")
        
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# UNet模型定义
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_map_size=64):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_map_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_map_size * 2, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        dec = self.decoder(middle)
        return dec

# DDPM模型定义
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(DDPM, self).__init__()
        self.model = model
        self.timesteps = timesteps
        
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def forward(self, x_0):
        self.alpha_cumprod = self.alpha_cumprod.to(x_0.device)
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0)
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def denoise(self, x_t, t):
        predicted_noise = self.model(x_t)
        return predicted_noise

    def loss_fn(self, x_0, x_t, noise, t):
        predicted_noise = self.denoise(x_t, t)
        return F.mse_loss(predicted_noise, noise)

# 训练过程
def train_ddpm(train_images, model, timesteps=1000, epochs=10, batch_size=64, lr=1e-4, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0  # Normalize to [0, 1]
    train_images = train_images.to(device)
    
    all_losses = []  # 用于记录每个epoch的损失
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx in range(0, len(train_images), batch_size):
            batch = train_images[batch_idx:batch_idx+batch_size]
            
            # 前向过程
            x_t, noise = model(batch)

            # 随机选择t
            t = torch.randint(0, model.timesteps, (batch.size(0),), device=device)
            
            # 计算损失
            loss = model.loss_fn(batch, x_t, noise, t)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        # 记录每个epoch的损失
        all_losses.append(epoch_loss / len(train_images))

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss / len(train_images):.4f}")
        
        # 每个epoch生成一张图像
        if (epoch + 1) % 10 == 0:  # 每个epoch都生成图像
            model.eval()
            with torch.no_grad():
                # 使用一个批次的图像来生成图像
                sample_images, _ = model(train_images[:batch_size])  # 只用前batch_size个图片
                show_samples(sample_images, epoch)

    # 绘制损失曲线
    plt.plot(range(epochs), all_losses, label="Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

# 显示生成的样本
def show_samples(samples, epoch):
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2  # Normalize to [0, 1]
    grid_size = int(np.sqrt(samples.shape[0]))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i, j]
            ax.imshow(samples[i * grid_size + j, 0], cmap="gray")
            ax.axis('off')
    plt.suptitle(f"Epoch {epoch+10}")
    plt.show()

# 读取数据集
train_images = read_mnist_images('D:/vs_codes/DDPM-main/MNIST/train-images.idx3-ubyte')
train_labels = read_mnist_labels('D:/vs_codes/DDPM-main/MNIST/train-labels.idx1-ubyte')


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet(in_channels=1, out_channels=1).to(device)
ddpm_model = DDPM(model=unet_model).to(device)

# 训练模型
train_ddpm(train_images, ddpm_model, epochs=50, batch_size=64, device=device)
