import numpy as np
import torch
import torch.nn as nn # contains functions needed to defined layers
import torch.nn.functional as F # contains activation functions
import torch.optim as optim
from torchvision import datasets, transforms #torchvision
import yaml
from my_models import Net, BigCNN, SmolCNN
import matplotlib.pyplot as plt

with open('MNIST_param.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda")
batch_size = config['batch_size']
epochs = config['epochs']
lr = config['learning_rate']
print("batch_size: ", batch_size)
print("epochs: ", epochs)
print("learning_rate: ", lr)

# Returns the mean loss of the model on the training set for epoch  
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        losses.append(loss.item())  
    return np.mean(losses)      

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # at test time we don't need gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset) #added here
    return test_loss, test_acc 

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
# Dataset1: Train, Dataset2: Test
dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, num_workers=1, pin_memory=True,shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, num_workers=1, pin_memory=True,shuffle=True,batch_size=batch_size)

model = BigCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

train_loss = []
test_loss = []
test_acc = []
for epoch in range(1, epochs + 1):
    train_loss.append(train(model, device, train_loader, optimizer, epoch))
    test_loss, test_acc = test(model, device, test_loader)
    