import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.soft_max(out)
        return out


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 256, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.soft_max(out)
        return out


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 32, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.soft_max(out)
        return out


best_acc = -1
best_model = None
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
DATA_PATH = 'datasets/CNN/'
MODEL_STORE_PATH = 'model/'

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=trans, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for network in [ConvNet1, ConvNet2, ConvNet3]:
    model = network()
    print(network.__name__)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              correct / total))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print('Test Accuracy of the model', network.__name__, 'on the 10000 test images: {}'.format(accuracy))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

trans = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = datasets.FashionMNIST(root=DATA_PATH, train=False, transform=trans, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

similar = [0] * 10
similar_probabilities = [0] * 10
CM = [0] * 10
for i in range(10):
    CM[i] = [0] * 10
    similar[i] = [object] * 10
    similar_probabilities[i] = [0] * 10

with torch.no_grad():
    correct = 0
    total = 0
    k = 0
    for images, labels in test_loader:
        outputs = best_model(images)
        pred_prob, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            CM[labels[i]][predicted[i]] += 1
        for i in range(len(labels)):
            if pred_prob[i] > similar_probabilities[predicted[i]][labels[i]]:
                similar_probabilities[predicted[i]][labels[i]] = pred_prob[i]
                similar[predicted[i]][labels[i]] = images[i].shape[0] + 100 * k
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        k += 1
    accuracy = correct / total
    print('Accuracy of the best model on the FashionMNIST test images: {}'.format(accuracy))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fig, plots = plt.subplots(ncols=10, nrows=10)
    for i in range(10):
        for j in range(10):
            k = 0
            for image, label in test_loader:
                if k == similar[i][j]:
                    plots[i][j].imshow(image[0].reshape(28, 28), cmap='gray')
                    break
                k += 1
            plots[i][j].set_xticks([])
            plots[i][j].set_yticks([])
print(CM )
plt.show()
