import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys


# train function
def train(module, train_data, optimizer_function, epoch_num):
    for batch_idx, (t_data, target) in enumerate(train_data):
        # data -> binary
        # t_data = t_data.view(t_data.size(0), -1)
        t_data_binary = np.ceil(t_data.numpy())
        t_data = torch.from_numpy(t_data_binary)
        t_data, target = Variable(t_data).to(device), Variable(target).to(device)
        optimizer_function.zero_grad()
        output = module(t_data)
        loss = Loss_function(output, target)
        loss.backward()
        optimizer_function.step()
        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(t_data), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))
        

# test function
def test(model, test_data, epoch_num, writer):
    correct = 0
    with torch.no_grad():
        for t_data, target in test_data:
            # t_data = t_data.view(t_data.size(0), -1)
            t_data_binary = np.ceil(t_data.numpy())
            t_data = torch.from_numpy(t_data_binary)
            t_data, target = Variable(t_data), Variable(target)
            output = model(t_data)
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\nTest set: Epoch:{} Accuracy: {}/{} ({:.2f}%) \n".format(epoch_num, correct, len(test_data.dataset),
                                                                     100. * correct / len(test_data.dataset)))
    # record data in tensorboard log
    writer.add_scalar('Accuracy', 100. * correct / len(test_data.dataset), epoch_num)

# Network structure
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7, 30, bias=False)
        self.fc2 = nn.Linear(30, 30, bias=False)
        self.fc3 = nn.Linear(30, 10, bias=False)

    # connect inputs and outputs size: 15x15 -> 7x7 -> 30 -> 30 -> 10 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)
        x = x.view(-1, 7*7)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# hyper parameters
# size_inputs = 16*16
# size_hidden1 = 32
# size_hidden2 = 32
# size_outputs = 10
learning_rate = 0.01
BATCH_SIZE = 1
EPOCHS = 100


if __name__ == '__main__':
    # data precoding
    train_transformer = transforms.Compose([
        transforms.Resize(15), # down sampling
        transforms.ToTensor()
    ])

    # data loading
    train_loader = data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transformer),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=train_transformer),
        batch_size=BATCH_SIZE, shuffle=True)
    
    # compare images (28x28 vs 15x15)
    raw_train_data = data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=False)
    transform_train_data = data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transformer),
        batch_size=BATCH_SIZE, shuffle=False)

    # check for data
    show_fig = False
    if show_fig:
        for batch_idx, (t_data, target) in enumerate(raw_train_data):
            t_data = t_data.view(28, 28)
            t_data_binary = np.ceil(t_data.numpy())
            t_data_binary = torch.from_numpy(t_data_binary)
            # print(t_data)
            if batch_idx < 3:
                plt.figure(f'raw data {batch_idx}')
                plt.imshow(t_data)
                
        target_number_list = []
        for batch_idx, (t_data, target) in enumerate(transform_train_data):
            t_data = t_data.view(15, 15)
            target_number_list.append(target)
            # save data as a .txt file for zehan's test
            if not os.path.exists('data_figures'):
                os.mkdir('data_figures')
            t_data_1 = t_data.numpy()
            # np.savetxt(f'data_figures/fig{batch_idx}.txt', t_data_1)
            # print(t_data_1)
            t_data_binary = np.ceil(t_data.numpy())
            t_data_binary = torch.from_numpy(t_data_binary)
            if batch_idx < 3:
                plt.figure(f'transformed data {batch_idx}')
                plt.imshow(t_data)
                plt.figure(f'data_binary {batch_idx}')
                plt.imshow(t_data_binary)
                print(t_data_binary)
        plt.show()
        # np.savetxt('value_list.txt', np.array(target_number_list))
        sys.exit(-1)

    # cuda acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'  # in MNIST recognition 'GPu' is slower than 'cpu'
    print(f"Using {device} device")
    
    # create a network sample, shape of network could be changed in definition of cnn network
    net = Net().to(device)
    print(net)
    print(net.state_dict().keys())

    # record the weight datas as .npy form
    conv1_max_list, conv1_min_list = [], []
    fc1_max_list, fc1_min_list = [], []
    fc2_max_list, fc2_min_list = [], []
    fc3_max_list, fc3_min_list = [], []

    # initial weight contribution
    nn.init.normal_(net.state_dict()['conv1.weight'], mean=0, std=0.1)
    nn.init.normal_(net.state_dict()['fc1.weight'], mean=0, std=0.1)
    nn.init.normal_(net.state_dict()['fc2.weight'], mean=0, std=0.1)
    nn.init.normal_(net.state_dict()['fc3.weight'], mean=0, std=0.1)

    conv1_weight = net.state_dict()['conv1.weight'].numpy()
    fc1_weight = net.state_dict()['fc1.weight'].numpy()
    fc2_weight = net.state_dict()['fc2.weight'].numpy()
    fc3_weight = net.state_dict()['fc3.weight'].numpy()

    print(f'conv1 max: {np.max(conv1_weight)} min: {np.min(conv1_weight)}')
    print(f'fc1 max: {np.max(fc1_weight)} min: {np.min(fc1_weight)}')
    print(f'fc2 max: {np.max(fc2_weight)} min: {np.min(fc2_weight)}')
    print(f'fc3 max: {np.max(fc3_weight)} min: {np.min(fc3_weight)}')

    conv1_max_list.append(np.max(conv1_weight))
    conv1_min_list.append(np.min(conv1_weight))
    fc1_max_list.append(np.max(fc1_weight))
    fc1_min_list.append(np.min(fc1_weight))
    fc2_max_list.append(np.max(fc2_weight))
    fc2_min_list.append(np.min(fc2_weight))
    fc3_max_list.append(np.max(fc3_weight))
    fc3_min_list.append(np.min(fc3_weight))

    # loss function and optimizer
    Loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # tensorboard
    tensorlog_path = 'CnnConnect_Mnist'
    if os.path.exists(tensorlog_path):
        shutil.rmtree(tensorlog_path)
    writer = SummaryWriter(tensorlog_path)
    
    for epoch in range(1, EPOCHS + 1):
        train(module=net, train_data=train_loader, optimizer_function=optimizer, epoch_num=epoch)
        test(model=net, test_data=test_loader, epoch_num=epoch, writer=writer)
        print(net.state_dict().keys())
        # conv1_weight = net.state_dict()['conv1.weight'].numpy()
        # fc1_weight = net.state_dict()['fc1.weight'].numpy()
        # fc2_weight = net.state_dict()['fc2.weight'].numpy()
 
        data_save_path = 'weight_data_cnn'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # save data
        torch.save(net, f'{data_save_path}/epoch_{epoch}')
        # np.save(f'{data_save_path}\\hidden1', conv1_weight)
        # np.save(f'{data_save_path}\\hidden2', fc1_weight)
        # np.save(f'{data_save_path}\\fc2_weight', fc2_weight)
        # print(f'hidden1 max: {np.max(conv1_weight)} min: {np.min(conv1_weight)}')
        # print(f'hidden2 max: {np.max(fc1_weight)} min: {np.min(fc1_weight)}')
        # print(f'fc2_weight max: {np.max(fc2_weight)} min: {np.min(fc2_weight)}')
        # save data
        conv1_max_list.append(np.max(conv1_weight))
        conv1_min_list.append(np.min(conv1_weight))
        fc1_max_list.append(np.max(fc1_weight))
        fc1_min_list.append(np.min(fc1_weight))
        fc2_max_list.append(np.max(fc2_weight))
        fc2_min_list.append(np.min(fc2_weight))
        fc3_max_list.append(np.max(fc3_weight))
        fc3_min_list.append(np.min(fc3_weight))
    
    # save range of weight datas
    np.savetxt('conv1_min_list.txt', conv1_min_list)
    np.savetxt('fc1_min_list.txt', fc1_min_list)
    np.savetxt('fc2_min_list.txt', fc2_min_list)
    np.savetxt('conv1_max_list.txt', conv1_max_list)
    np.savetxt('fc1_max_list.txt', fc1_max_list)
    np.savetxt('fc2_max_list.txt', fc2_max_list)
    np.savetxt('fc3_min_list.txt', fc3_min_list)
    np.savetxt('fc3_max_list.txt', fc3_max_list)