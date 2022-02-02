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
import cv2
import sys


# train function
def train(module, train_data, optimizer_function, epoch_num):
    for batch_idx, (t_data, target) in enumerate(train_data):
        # data -> binary
        t_data = t_data.view(t_data.size(0), -1)
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
            t_data = t_data.view(t_data.size(0), -1)
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

    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1, bias=False)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.out = nn.Linear(n_hidden2, n_output, bias=False)

    # connect inputs and outputs
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.out(x))
        return x


# hyper parameters
size_inputs = 16*16
size_hidden1 = 32
size_hidden2 = 32
size_outputs = 10
learning_rate = 0.01
BATCH_SIZE = 1
EPOCHS = 100


if __name__ == '__main__':
    # data precoding
    train_transformer = transforms.Compose([
        transforms.Resize(16), # down sampling
        transforms.ToTensor()
    ])

    # data loading
    train_loader = data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transformer),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=train_transformer),
        batch_size=BATCH_SIZE, shuffle=True)
    
    # compare images (28x28 vs 16x16)
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
                
        
        for batch_idx, (t_data, target) in enumerate(transform_train_data):
            t_data = t_data.view(16, 16)
            t_data_binary = np.ceil(t_data.numpy())
            t_data_binary = torch.from_numpy(t_data_binary)
            if batch_idx < 3:
                plt.figure(f'transformed data {batch_idx}')
                plt.imshow(t_data)
                plt.figure(f'data_binary {batch_idx}')
                plt.imshow(t_data_binary)
                print(t_data_binary)
        plt.show()
        sys.exit(-1)

    # cuda acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'  # in MNIST recognition 'GPu' is slower than 'cpu'
    print(f"Using {device} device")
    
    # create a network sample
    net = Net(n_feature=size_inputs, n_hidden1=size_hidden1, n_hidden2=size_hidden2, n_output=size_outputs).to(device)
    print(net.state_dict().keys())

    # record the weight datas as .npy form
    hid1_max_list, hid1_min_list = [], []
    hid2_max_list, hid2_min_list = [], []
    out_max_list, out_min_list = [], []
    nn.init.normal_(net.state_dict()['hidden1.weight'], mean=0, std=0.1)
    nn.init.normal_(net.state_dict()['hidden2.weight'], mean=0, std=0.1)
    nn.init.normal_(net.state_dict()['out.weight'], mean=0, std=0.1)
    hidden1_weight = net.state_dict()['hidden1.weight'].numpy()
    hidden2_weight = net.state_dict()['hidden2.weight'].numpy()
    out_weight = net.state_dict()['out.weight'].numpy()
    print(f'hidden1 max: {np.max(hidden1_weight)} min: {np.min(hidden1_weight)}')
    print(f'hidden2 max: {np.max(hidden2_weight)} min: {np.min(hidden2_weight)}')
    print(f'out_weight max: {np.max(out_weight)} min: {np.min(out_weight)}')
    hid1_max_list.append(np.max(hidden1_weight))
    hid1_min_list.append(np.min(hidden1_weight))
    hid2_max_list.append(np.max(hidden2_weight))
    hid2_min_list.append(np.min(hidden2_weight))
    out_max_list.append(np.max(out_weight))
    out_min_list.append(np.min(out_weight))

    # loss function and optimizer
    Loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # tensorboard
    tensorlog_path = 'FullConnect_Mnist'
    if os.path.exists(tensorlog_path):
        shutil.rmtree(tensorlog_path)
    writer = SummaryWriter(tensorlog_path)
    for epoch in range(1, EPOCHS + 1):
        train(module=net, train_data=train_loader, optimizer_function=optimizer, epoch_num=epoch)
        test(model=net, test_data=test_loader, epoch_num=epoch, writer=writer)
        print(net.state_dict().keys())
        hidden1_weight = net.state_dict()['hidden1.weight'].numpy()
        hidden2_weight = net.state_dict()['hidden2.weight'].numpy()
        out_weight = net.state_dict()['out.weight'].numpy()
 
        data_save_path = 'weight_data'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # save data
        torch.save(net, f'{data_save_path}/epoch_{epoch}')
        # np.save(f'{data_save_path}\\hidden1', hidden1_weight)
        # np.save(f'{data_save_path}\\hidden2', hidden2_weight)
        # np.save(f'{data_save_path}\\out_weight', out_weight)
        # print(f'hidden1 max: {np.max(hidden1_weight)} min: {np.min(hidden1_weight)}')
        # print(f'hidden2 max: {np.max(hidden2_weight)} min: {np.min(hidden2_weight)}')
        # print(f'out_weight max: {np.max(out_weight)} min: {np.min(out_weight)}')
        # save data
        hid1_max_list.append(np.max(hidden1_weight))
        hid1_min_list.append(np.min(hidden1_weight))
        hid2_max_list.append(np.max(hidden2_weight))
        hid2_min_list.append(np.min(hidden2_weight))
        out_max_list.append(np.max(out_weight))
        out_min_list.append(np.min(out_weight))
    np.save('hid1_min_list', hid1_min_list)
    np.save('hid2_min_list', hid2_min_list)
    np.save('out_min_list', out_min_list)
    np.save('hid1_max_list', hid1_max_list)
    np.save('hid2_max_list', hid2_max_list)
    np.save('out_max_list', out_max_list)