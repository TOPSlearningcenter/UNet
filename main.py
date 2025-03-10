import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条
import cv2  # OpenCV库，用于图像处理


# 定义UNet网络结构
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 定义一个卷积块，包含两个3x3卷积层、BatchNorm和ReLU激活函数
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 编码器部分，包含4个卷积块
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        # 最大池化层，用于下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间层，包含一个卷积块
        self.middle = conv_block(512, 1024)

        # 解码器部分，包含4个上采样卷积块
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        # 最后的1x1卷积层，输出通道数为类别数
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    # 前向传播函数
    def forward(self, x):
        # 编码器，提取图像特征并逐层降低分辨率
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        middle = self.middle(self.pool(enc4))

        # 解码器，逐层恢复分辨率
        dec4 = self.upconv4(middle)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 跳跃连接
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接
        dec1 = self.decoder1(dec1)

        # 通过1x1卷积层输出预测结果
        out = self.final(dec1)

        return out


# 定义数据集类，用于加载图像和标签
class SegmentationDataset(Dataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 ignore_label=255,
                 num_classes=19,
                 size=(512, 512),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.image_dir = image_dir  # 图像目录
        self.label_dir = label_dir  # 标签目录
        self.mean = mean  # 图像归一化的均值
        self.std = std  # 图像归一化的标准差
        self.size = size  # 图像和标签的尺寸

        self.images = os.listdir(image_dir)  # 获取图像文件名列表
        self.labels = os.listdir(label_dir)  # 获取标签文件名列表
        self.labels = [label for label in self.labels if label.endswith('labelIds.png')]  # 过滤标签文件

        # 标签映射，将原始标签映射到0-18的类别，忽略某些标签
        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

    # 标签转换函数，将原始标签映射到0-18的类别
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    # 返回数据集的大小
    def __len__(self):
        return len(self.images)

    # 获取数据集中的单个样本
    def __getitem__(self, idx):
        # 获取图像和标签路径
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('leftImg8bit', 'gtFine_labelIds'))
        name = self.images[idx].split('leftImg8bit')[0][:-1]

        # 读取图像并调整大小
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = self.image_transform(image)
        image = image.transpose((2, 0, 1))  # 将图像从HWC转换为CHW格式

        # 读取标签并调整大小
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
        label = self.convert_label(label)  # 将原始标签映射到0-18的类别
        label = self.label_transform(label)  # 转换为numpy数组

        return image, label, name

    # 图像预处理函数，归一化图像
    def image_transform(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    # 标签预处理函数，转换为uint8类型
    def label_transform(self, label):
        return np.array(label).astype(np.uint8)


# 定义Runner类，用于训练和测试模型
class Runner:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model  # 模型
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.device = device  # 指定设备

    # 保存模型检查点
    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, filename)

    # 训练函数
    def train(self, dataloader, epochs):
        # 训练模式
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels, name in tqdm(dataloader):  # 使用tqdm显示进度条
                # 将图像和标签转移到指定设备
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images)  # 前向传播
                loss = self.criterion(outputs, labels)  # 损失计算

                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                self.optimizer.zero_grad()  # 梯度清零

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

        self.save_checkpoint(f"final.pth")  # 训练结束，保存模型权重

    # 计算损失函数
    def get_loss(self, outputs, targets):
        outputs = outputs.flatten(2)  # BxCxHxW -> BxCx(H*W)
        targets = targets.flatten(1)  # BxHxW -> Bx(H*W)
        return self.criterion(outputs, targets)  # 交叉熵损失

    # 测试函数
    def test(self, dataloader, weight_path, device, vis=True):
        self.model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])  # 从指定路径加载模型权重

        # 推理模式
        self.model.eval()
        # 关闭梯度计算
        with torch.no_grad():
            for idx, (images, labels, name) in enumerate(dataloader):
                # 将图像和标签转移到指定设备
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)  # 前向传播
                outputs = torch.argmax(outputs, dim=1)  # 获取预测类别

                self.visualize(images, labels, outputs, str(idx))  # 可视化结果

    # 可视化函数，显示输入图像、真值标签和预测结果
    def visualize(self, images, labels, outputs, iter):
        os.makedirs('vis', exist_ok=True)  # 创建可视化目录

        # 在batch中逐样本可视化
        for i in range(images.shape[0]):
            image = images[i].squeeze().cpu().permute(1, 2, 0)
            label = labels[i].squeeze().cpu()
            output = outputs[i].squeeze().cpu()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(self.invert_image_transform(image))
            axes[0].set_title("Input Image")
            axes[1].imshow(self.grey2color(label))
            axes[1].set_title("Ground Truth")
            axes[2].imshow(self.grey2color(output))
            axes[2].set_title("Prediction")
            fig.savefig(os.path.join('vis', f"{iter}.{i}.png"))  # 保存可视化结果
        plt.show()

    # 根据类别索引获取颜色，将灰度图像转换为彩色图像
    def grey2color(self, image):
        h, w = image.shape
        color_image = torch.zeros((h, w, 3), dtype=torch.uint8)
        for i in range(19):
            color_image[image == i] = torch.tensor(class_to_color[i], dtype=torch.uint8)
        return color_image

    # 反归一化图像，将图像从归一化状态转换回原始状态
    def invert_image_transform(self, image):
        image = image.cpu().numpy().astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image *= std
        image += mean
        image *= 255.0
        return image.astype(np.uint8)


# 可视化色彩映射，将类别映射到颜色
class_to_color = {
    0: (0, 0, 0),  # 黑色
    1: (255, 0, 0),  # 红色
    2: (0, 255, 0),  # 绿色
    3: (0, 0, 255),  # 蓝色
    4: (255, 255, 0),  # 黄色
    5: (255, 0, 255),  # 紫色
    6: (0, 255, 255),  # 青色
    7: (128, 0, 0),  # 深红色
    8: (0, 128, 0),  # 深绿色
    9: (0, 0, 128),  # 深蓝色
    10: (128, 128, 0),  # 深黄色
    11: (128, 0, 128),  # 深紫色
    12: (0, 128, 128),  # 深青色
    13: (192, 192, 192),  # 银色
    14: (128, 128, 128),  # 灰色
    15: (255, 165, 0),  # 橙色
    16: (255, 192, 203),  # 粉红色
    17: (255, 215, 0),  # 金色
    18: (0, 255, 128)  # 亮绿色
}

# 训练和测试数据路径
image_dir = 'images'
label_dir = 'labels'

# 训练设置
batch_size = 8
learning_rate = 0.0001
num_classes = 19
ignore_label = 255
size = (512, 512)

# 图像归一化参数
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建数据集和数据加载器
dataset = SegmentationDataset(image_dir, label_dir, num_classes=num_classes, size=size, mean=mean, std=std)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
model = UNet(in_channels=3, out_channels=num_classes).to(device)  # 初始化模型
criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)  # 设置损失函数，交叉熵损失，忽略标签255
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 设置优化器，Adam优化器

# 创建Runner实例
runner = Runner(model, criterion, optimizer, device)

runner.train(dataloader, epochs=20)

runner.test(dataloader, 'final.pth', device)
# runner.test(dataloader, 'pretrained_model.pth', device)
