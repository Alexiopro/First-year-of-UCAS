from multiprocessing.util import ForkAwareLocal
from pyexpat import model
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

torch.manual_seed(3047)
batch_size = 5

train_set = torchvision.datasets.CIFAR10(root = r"./data",train=True, transform=transform, 
                                         download=False)
train_loader = torch.utils.data.DataLoader(train_set,batch_size= batch_size,shuffle=True)
test_set  = torchvision.datasets.CIFAR10(root = r"./data",train=True, transform=transform,
                                         download=False)
test_loader  = torch.utils.data.DataLoader(test_set,batch_size= batch_size,shuffle=False)
classes   = ('plane', 'car', 'bird', 'cat','deer',
             'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CovNet(nn.Module):
    def __init__(self):
        super(CovNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.bn1   = nn.BatchNorm2d(16,momentum=0.98)

        self.conv2 = nn.Conv2d(16,36,3)
        self.bn2   = nn.BatchNorm2d(36,momentum=0.98)

        self.conv3 = nn.Conv2d(36,72,3)
        self.bn3   = nn.BatchNorm2d(72,momentum=0.98)

        self.fc1 = nn.Linear(72, 36)
        self.bn4 = nn.BatchNorm1d(36,momentum=0.98)

        self.fc3 = nn.Linear(36, 10)

        self.aap   = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.max_pool2d(x,2)

        x = self.aap(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc3(x)

        return x

model = CovNet()
model = model.to(device)

for m in model.modules():
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)#卷积层参数初始化
        nn.init.constant_(m.bias, 0)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight)

lr = 0.01
momentum=0.95
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
t = time.time()
model.train()
for epoch in range(10):
    t1 = time.time()
    train_loss = 0
    if epoch % 2 == 0:
        lr = lr * 0.9
        adjust_lr(optimizer, lr)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #正向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels) / batch_size
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #损失测算
        train_loss += loss.item()
        if i % 5000 == 4999:
            Loss = train_loss / 5000
            print('[%d, %5d] loss: %.3f use: %.2d s' %(epoch + 1, i+1, Loss , time.time() - t1))
            train_loss = 0

print('Finish Training ! Total used: {:.2f} s'.format(time.time() - t))


model.eval()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))








