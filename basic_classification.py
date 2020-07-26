import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def img_show(img):
    # torch.tensor [c, h, w]
    img = img / 2 + 0.5  # 返归一
    nping = img.numpy()
    nping = np.transpose(nping, (1, 2, 0))  # [h, w, c]
    plt.imshow(nping)
    plt.show()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ]
)

# 训练数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8)

# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)

if __name__ == '__main__':
    data_iter = iter(train_loader)  # 随机加载一个mini batch
    images, labels = data_iter.next()
    img_show(torchvision.utils.make_grid(images))
