from traceback import print_tb
from click import echo
import numpy as np
import os
import matplotlib.pyplot as plt
from Test_FullCon import*
import torch
from torchvision import datasets, transforms
from torch.utils import data

if __name__ == '__main__':
    # check for weight datas
    file_list = ['out_max_list.npy', 'out_min_list.npy', 'hid1_max_list.npy', 'hid2_max_list.npy',
    'hid1_min_list.npy', 'hid2_min_list.npy']
    plt.figure()
    for file_path in file_list:
        if os.path.exists(file_path):
            datas = np.load(file_path)
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

    # create the Neural Network
    net = Net(n_feature=size_inputs, n_hidden1=size_hidden1, n_hidden2=size_hidden2, n_output=size_outputs)
    epoch_number = np.linspace(20, 40, 21, dtype=int)
    for epoch_value in epoch_number:
        weight_data_path = f'./weight_data/epoch_{epoch_value}'
        print('raw: {}'.format(net.state_dict()['hidden1.weight']))
        net.state_dict()['hidden1.weight'] = torch.tensor(np.load(f'{weight_data_path}\hidden1.npy'))
        print('changed: {}'.format(net.state_dict()['hidden1.weight']))