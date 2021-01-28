from zipfile import ZipFile
import os
import gdown

if not os.path.isfile('fruits.zip'):
    url = 'https://drive.google.com/uc?id=16hvrP_4yUFPt-Tjv1srU_uxncAyiG7Bi'
    output = 'fruits.zip'
    gdown.download(url, output, quiet=False)

    with ZipFile(output, 'r') as zf:
        zf.extractall()

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = dset.ImageFolder(root='fruits-360/training/', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = dset.ImageFolder(root='fruits-360/test/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=131):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, bias=False)
        self.pool4 = nn.MaxPool2d(2, 2)
        # classifier
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


net = SimpleCNN()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def main():
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


def test():
    net.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for i, data in enumerate(testloader):
            total += 4
            inputs, labels = data
            outputs = torch.argmax(net(inputs), dim=1)
            correct += torch.sum(torch.eq(outputs, labels)).cpu().item()
    print(f'Test set accuracy: {correct * 100 / total:.4f}%')


if __name__ == '__main__':
    main()
    torch.save(net.state_dict(), 'models/model.pt')
    test()
