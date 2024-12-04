import torch
from torch import nn
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


# class Embedding(nn.Module):
#     def __init__(self, num_patches, min_dim, dim):
#         super(Embedding, self).__init__()
#         #  第一层卷积，输入通道3，输出通道16，卷积核大小3x3，步长1，padding=1
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.num = num_patches
#         #  最大池化层，池化窗口大小2x2
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         #  第二层卷积，输入通道16，输出通道16，卷积核大小3x3，步长1，padding=1
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(16, num_patches, kernel_size=3, stride=1, padding=1)
#         #  全连接层，输入特征768  (16*28*28)，输出特征768
#         self.fc = nn.Linear(min_dim * min_dim, dim)
#
#     def forward(self, x):
#         # n, c, c = x.shape
#         x = self.relu(self.conv1(x))  # torch.Size([20, 16, 224, 224])
#         x = self.pool(x)  # torch.Size([20, 16, 112, 112])
#         x = self.relu(self.conv2(x))  # torch.Size([20, 32, 112, 112])
#         x = self.pool(x)  # torch.Size([20, 32, 56, 56])
#         x = self.relu(self.conv3(x))  # torch.Size([20, 16, 56, 56])
#         x = self.pool(x)  # torch.Size([20, 16, 28, 28])
#         # print('x', x.shape)
#         x = x.view(self.num, -1)  # torch.Size([20, 16, 784])
#         # print('x', x.shape)
#         x = self.fc(x)  # torch.Size([20, 16, 768])
#         return x


class DatasetProcessingChest(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)
        # self.embedding = Embedding(num_patches=9, min_dim=48, dim=768)  # 48/28/63

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        # img = self.embedding(img[np.newaxis, :, :])  # [num_patches=9,dim=768]
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingISIC2018(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingCIFAR(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)
        # self.embedding = Embedding(num_patches=9, min_dim=48, dim=768)  # 48/28/63

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        # img = self.embedding(img[np.newaxis, :, :])  # [num_patches=9,dim=768]
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

