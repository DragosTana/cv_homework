import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(1024, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.bn3(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = self.bn6(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.bn7(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class SmolCNN(nn.Module):
    def __init__(self):
        super(SmolCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(5408, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = nn.functional.relu(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        output = F.log_softmax(x, dim=1)
        return output