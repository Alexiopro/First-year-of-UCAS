from tkinter.tix import Tree
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils, datasets
from torchvision.datasets import mnist 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch

torch.manual_seed(3047)
train_batch_size = 128
test_batch_size = 128
learning_rate = 0.1
num_epoches = 30
lr = learning_rate
# 动量参数
momentum = 0.5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
    ])

train_dataset = mnist.MNIST('.\data', train=True, transform=transform, download=False)
test_dataset = mnist.MNIST('.\data', train=False, transform=transform,download=False)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

examples = enumerate(test_loader)
batch_indevx, (examples_data, example_targets) = next(examples)

#调用nn工具箱建立神经网络
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        #self.layer = nn.Sequential(
        #nn.Linear(in_dim, n_hidden_1),
        #nn.ReLU(True),
        #nn.Linear(n_hidden_1, n_hidden_2),
        #nn.ReLU(True),
        #nn.Linear(n_hidden_2, out_dim))
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
 
    def forward(self, x):
        x1 = torch.relu(self.layer1(x))
        x2 = torch.relu(self.layer2(x1))
        x3 = torch.sigmoid(self.layer3(x2))
        return x3

#实例化网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#for n in range(1,11):
    #model = Net(28 * 28, 100*n, 200*n, 10)
model = Net(28 * 28, 300, 600, 10)
    
model.to(device)

    # 定义损失函数(交叉熵）和优化器（梯度下降）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
losses = []
acces = []
eval_losses = []
eval_acces = []


for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    #动态修改参数学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.9
    for img, label in train_loader:
        img=img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img=img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader)))

