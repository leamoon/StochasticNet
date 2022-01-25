import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# super parameters
picture_feature = 16*16
picture_hidden1 = 32
picture_hidden2 = 32
picture_output = 10
learning_rate = 0.01
BATCH_SIZE = 1
EPOCHS = 100


# data loading
train_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(16),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=train_transformer),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=train_transformer),
    batch_size=BATCH_SIZE, shuffle=True)


# Network structure
class Net(nn.Module):

    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1, bias=False)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.out = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.out(x))
        return x


# create a network sample
net = Net(n_feature=picture_feature, n_hidden1=picture_hidden1, n_hidden2=picture_hidden2, n_output=picture_output)

# loss function and optimizer
Loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# tensor vision
writer = SummaryWriter('D:\\Python_projects\\StochasticNet\\FullConnect_Mnist')


# train function
def train(module, train_data, optimizer_func, epoch_num):
    for batch_idx, (t_data, target) in enumerate(train_data):
        t_data = t_data.view(t_data.size(0), -1)
        # t_data = t_data.view(picture_feature, -1)
        t_data, target = Variable(t_data), Variable(target)
        optimizer_func.zero_grad()
        output = module(t_data)
        loss = Loss_function(output, target)
        loss.backward()
        optimizer_func.step()
        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(t_data), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))
            # writer.add_scalar('train_loss{}'.format(epoch_num), loss.item(), batch_idx)


# test function
def test(model, test_data, epoch_num):
    correct = 0
    with torch.no_grad():
        for t_data, target in test_data:
            t_data = t_data.view(t_data.size(0), -1)
            t_data, target = Variable(t_data), Variable(target)
            output = model(t_data)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\nTest set: Epoch:{} Accuracy: {}/{} ({:.2f}%) \n".format(epoch_num, correct, len(test_data.dataset),
                                                                     100. * correct / len(test_data.dataset)))
    writer.add_scalar('Accuracy/test', 100. * correct / len(test_data.dataset), epoch_num)


t1 = time.time()
for epoch in range(1, EPOCHS + 1):
    train(module=net, train_data=train_loader, optimizer_func=optimizer, epoch_num=epoch)
    test(model=net, test_data=test_loader, epoch_num=epoch)
t2 = time.time()
print('All time consumed :{:.1f}s'.format(t2 - t1))