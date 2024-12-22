import torch
import torchvision
import torchvision.transforms as tsf
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

transform = tsf.Compose([tsf.ToTensor(), tsf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=r"D:\picture fenlei\cifar-10-batches-py", train=True, download=True,
                                        transform=transform)
testset = torchvision.datasets.CIFAR10(root=r"D:\picture fenlei\cifar-10-batches-py", train=False, download=True,
                                       transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

bz = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(), lr=0.001)


def train_model(model, trainloader, bz, optimizer, epochs=10):
    best = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = bz(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        print(f"Epoch_{epoch + 1},Loss: {running_loss / len(trainloader):.4f},Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print(f'Epoch_{epoch + 1}结果保存')
        if accuracy > best:
            best = accuracy
            name = f'{accuracy:.2f}%.pth'
            torch.save(model.state_dict(), name)
            print(f"当前Epoch_{epoch + 1}是最好的模型，best准确率保存为{accuracy:.2f}%")


train_model(model, trainloader, bz, optimizer, epochs=10)

model = CNN()
model.load_state_dict(torch.load('model_epoch_5.pth', map_location=device))
model.to(device)
train_model(model, trainloader, bz, optimizer, epochs=10)


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'测试准确度为: {100 * correct / total:.2f}%')


evaluate(model, testloader)


def show(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


for images, labels in testloader:
    show(images[0])
    print(f'真实结果: {classes[labels[0]]}')
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print(f'预测结果: {classes[predicted[0]]}')