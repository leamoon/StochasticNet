import numpy as np
import os
import matplotlib.pyplot as plt
from Cnn import*
import torch
from torchvision import datasets, transforms
from torch.utils import data
import shutil
from torch.utils.tensorboard import SummaryWriter

def test(model, test_data, epoch_num, writer, decimal_number):
    """ 
    a function used to test the accuracy of neural network.

    Args:
        model (class): a neural network
        test_data (iterator): test datas
        epoch_num (int): repeat number of datasets used to train
        writer (tensorboard): write log file for data visition
        decimal_number (int): the significant number of weights
    """
    correct = 0
    with torch.no_grad():
        for t_data, target in test_data:
            t_data_binary = np.ceil(t_data.numpy())
            t_data = torch.from_numpy(t_data_binary)
            t_data, target = Variable(t_data), Variable(target)
            output = model(t_data)
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\nTest: Epoch:{} Accuracy: {}/{} ({:.2f}%) \n".format(epoch_num, correct, len(test_data.dataset),
                                                                     100. * correct / len(test_data.dataset)))
    # record data in tensorboard log
    # writer.add_scalar(f'Accuracy_{decimal_number}', 100. * correct / len(test_data.dataset), epoch_num)
    writer.add_scalar(f'Accuracy_{epoch_num}', 100. * correct / len(test_data.dataset), decimal_number)

if __name__ == '__main__':
    # load the weight data and save them as .txt file.
    net = Net()
    net = torch.load(f'weight_data_cnn\epoch_11')
    new_weights = net.state_dict()['conv1.weight'].numpy()
    print(new_weights)
    np.savetxt('conv1_weight.txt', np.array(new_weights[0][0]))
    new_weights = net.state_dict()['fc1.weight'].numpy()
    np.savetxt('fc1_weight.txt', new_weights)
    new_weights = net.state_dict()['fc2.weight'].numpy()
    np.savetxt('fc2_weight.txt', new_weights)
    new_weights = net.state_dict()['fc3.weight'].numpy()
    np.savetxt('fc3_weight.txt', new_weights)

    # check for weight datas
    file_list = ['fc1_max_list.txt', 'fc1_min_list.txt', 'fc2_max_list.txt', 'fc2_min_list.txt',
    'fc3_min_list.txt', 'fc3_max_list.txt', 'conv1_min_list.txt', 'conv1_max_list.txt']
    plt.figure()
    for file_path in file_list:
        if os.path.exists(file_path):
            datas = np.loadtxt(file_path)
            plt.plot(datas, label=f'{file_path}')
            plt.xlabel('Epoch', size=20)
            plt.ylabel('Value', size=20)
            plt.title('Weight', size=16)
    
    # indication line for Figs
    line_2 = [3]*len(datas)
    line_minus_2 = [-3]*len(datas)
    plt.plot(line_2, c='black', linestyle='--')
    plt.plot(line_minus_2, c='black', linestyle='--')
    plt.legend()
    plt.show()
    
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

    # create the Neural Network
    net = Net()
    
    # data visition
    tensorlog_path = 'FullConnect_Mnist'
    if os.path.exists(tensorlog_path):
        shutil.rmtree(tensorlog_path)
    writer = SummaryWriter(tensorlog_path)
    epoch_number_list = np.linspace(20, 40, 21, dtype=int)
    decimal_number_list = np.linspace(0, 16, 17, dtype=int)
    
    # for decimal_value in decimal_number_list:
    for epoch_value in epoch_number_list:
        for decimal_value in decimal_number_list:
            net = torch.load(f'weight_data\epoch_{epoch_value}')
            new_weights = net.state_dict()['out.weight'].numpy()
            new_weights = torch.from_numpy(np.round(new_weights, decimal_value))
            net.state_dict()['out.weight'].copy_(new_weights)
            new_weights = net.state_dict()['hidden1.weight'].numpy()
            new_weights = torch.from_numpy(np.round(new_weights, decimal_value))
            net.state_dict()['hidden1.weight'].copy_(new_weights)
            new_weights = net.state_dict()['hidden2.weight'].numpy()
            new_weights = torch.from_numpy(np.round(new_weights, decimal_value))
            net.state_dict()['hidden2.weight'].copy_(new_weights)
            test(model=net, test_data=test_loader, epoch_num=epoch_value, writer=writer, decimal_number=decimal_value)
