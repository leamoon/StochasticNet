from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, tanh
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils import data
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
        t_data, target = Variable(t_data), Variable(target)
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
# def test(model, test_data, epoch_num):
def test(model, test_data, epoch_num):
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

# Network structure
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
        for p in self.parameters():
            p.requires_grad=False
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



kernel = np.genfromtxt("constant_weight_conv1_zehan.txt") / 16
hidden_layer1_linear_trans = np.genfromtxt("fc1_weight_0617_2022.txt")
hidden_layer2_linear_trans = np.genfromtxt("fc2_weight_0617_2022.txt")
output_layer_linear_trans = np.genfromtxt("fc3_weight_0617_2022.txt")
# hidden_layer1_linear_trans = torch.Tensor(np.load("weight_data/hidden1.npy"))
# hidden_layer2_linear_trans = torch.Tensor(np.load("weight_data/hidden2.npy"))
# output_layer_linear_trans = torch.Tensor(np.load("weight_data/out_weight.npy"))

#hidden_layer1_linear_trans = np.round(hidden_layer1_linear_trans * 33) / 33
anwser = np.genfromtxt("value_list.txt")

correct_count = 0

def conv_2d_single_kernel(input_data, kernel, stride):
    """单个卷积核进行卷积，得到单个输出。
    由于是学习卷积实现原理这里简单处理，padding 是自动补全，
    相当于tf 里面的 "SAME"。
    Args:
        input_data: 卷积层输入，是一个 shape 为 [h, w] 
            的 np.array。
        kernel: 卷积核大小，形式如 [k_h, k_w]
        stride: stride， list [s_h, s_w]。
    Return:
        out: 卷积结果
    """
    h, w = input_data.shape
    kernel_h, kernel_w = kernel.shape

    stride_h, stride_w = stride

    out = np.zeros((h//stride_h, w//stride_w))
    for idx_h, i in enumerate(range(0, h-kernel_h+1, stride_h)):
        for idx_w, j in enumerate(range(0, w-kernel_w+1, stride_w)):
            window = input_data[i:i+kernel_h, j:j+kernel_w]
            out[idx_h, idx_w] = np.sum(window*kernel)
    return out

for i in range(1000):

    fig_data = np.genfromtxt("data_figures/fig{}.txt".format(i))
    fig_data = np.ceil(fig_data)
    # fig_data = (fig_data > 0.5) # 二值化
    # fig_data = np.ceil(fig_data) # 二值化
    #fig_data = np.round(fig_data) # 二值化
    #plt.imshow(fig_data, cmap='Greys')
    #plt.show()

    #input_layer = np.reshape(fig_data, (256, 1))
    conv_result = conv_2d_single_kernel(fig_data[0:15, 0:15], kernel, (2, 2))
    #print(conv_result[1:7,1:7])

    conv_result[conv_result < 0] = 0
    # conv_result[conv_result > 1] = 1

    relu_output = np.reshape(conv_result, (49, 1))
    # input_layer = torch.Tensor(input_layer)

    hidden_layer1_result = dot(hidden_layer1_linear_trans, relu_output)
    hidden_layer1_output = tanh(hidden_layer1_result)

    hidden_layer2_result = dot(hidden_layer2_linear_trans, hidden_layer1_output)
    hidden_layer2_output = tanh(hidden_layer2_result)

    output_layer_result = dot(output_layer_linear_trans, hidden_layer2_output)
    output_layer_tanh = tanh(output_layer_result)

    number = np.argmax(output_layer_tanh)
    print(number, int(anwser[i]), end = ' ')
    if number == int(anwser[i]):
        correct_count += 1
        print('T')
    else:
        print('F')
    #print(output_layer_tanh)

print(correct_count)