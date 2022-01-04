# imports
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# create leNet network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x

# set device
#device = torch.device('cuda' if torch.cude.is_available() else 'cpu')

# hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS = 1
LEARNING_RATE = 0.01

# load data
train = datasets.MNIST("", train=True, download=True, 
                            transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, 
                            transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True) 

# initialize network
net = LeNet()

# optimizer & loss function
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

# network training
def train_network(num_epochs, net, trainset):
    net.train()

    total_step = len(trainset)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainset):
            b_x = images
            b_y = labels

            optimizer.zero_grad()

            # forward
            output = net(b_x)[0]
            loss = loss_function(output, b_y)

            # backward
            loss.backward()

            # gradient descent
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"epoch [{epoch + 1}\{num_epochs}], step [{i+1}\{total_step}], loss: {loss.item():.4f}")

train_network(NUM_EPOCHS, net, trainset)

# check accuracy of model
def check_accuracy(testset, net):
    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testset:
            net_out = net(images)[0]
            _, pred_y = net_out.max(1)
            correct += (pred_y == labels).sum()
            total += pred_y.size(0)
        
        accuracy = float(correct) / float(total) * 100
        print(f"accuracy: {accuracy:.2f}%")

check_accuracy(testset, net)
