from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, tanh
import torch
import torch.nn as nn
from torch.autograd import Variable

hidden_layer1_linear_trans = np.genfromtxt("fc1_weight_0617_2022.txt")
hidden_layer2_linear_trans = np.genfromtxt("fc2_weight_0617_2022.txt")
output_layer_linear_trans = np.genfromtxt("fc3_weight_0617_2022.txt")
anwser = np.genfromtxt("value_list.txt")

def test(model, figure_data, target):
    correct = 0
    with torch.no_grad():
        t_data = torch.from_numpy(figure_data)
        t_data = t_data.reshape([1, 1, 16, 16])
        # print(t_data)
        # print(t_data.shape)
        target = torch.as_tensor(target)
        # t_data = t_data.view(t_data.size(1), -1)
        # t_data_binary = np.ceil(t_data.numpy())
        # t_data = torch.from_numpy(t_data_binary)
        t_data, target = Variable(t_data), Variable(target)
        output = model(t_data)
        pred = output.max(1, keepdim=True)[1] 
        correct += pred.eq(target.view_as(pred)).sum().item()
        if correct > 0:
            print('right')
            return True
        else:
            print('wrong')
            return False

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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
        # for p in self.parameters():
        #     p.requires_grad=False
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7, 30, bias=False)
        self.fc2 = nn.Linear(30, 30, bias=False)
        self.fc3 = nn.Linear(30, 10, bias=False)

    # connect inputs and outputs size: 15x15 -> 7x7 -> 30 -> 30 -> 10 
    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)
        x = x.view(-1, 7*7)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'  # in MNIST recognition 'GPu' is slower than 'cpu'
print(f"Using {device} device")

# create a network sample
net = Net().to(device)
# net = torch.load(f'weight_data_cnn\epoch_20')
constant_weights = np.genfromtxt('constant_weight_conv1_zehan.txt')
constant_weights = torch.from_numpy(np.multiply(constant_weights, 1/16))
print(constant_weights)
net.state_dict()['conv1.weight'].copy_(constant_weights)
net.state_dict()['fc1.weight'].copy_(torch.from_numpy(hidden_layer1_linear_trans))
net.state_dict()['fc2.weight'].copy_(torch.from_numpy(hidden_layer2_linear_trans))
net.state_dict()['fc3.weight'].copy_(torch.from_numpy(output_layer_linear_trans))
# hidden_layer1_linear_trans = net.state_dict()['hidden1.weight'].numpy()
# hidden_layer2_linear_trans = net.state_dict()['hidden2.weight'].numpy()
# output_layer_linear_trans = net.state_dict()['out.weight'].numpy()
correct_count = 0
count_pytoch = 0
kernel = np.genfromtxt("constant_weight_conv1_zehan.txt") / 16

for i in range(1000):

    fig_data = np.genfromtxt("data_figures/fig{}.txt".format(i))
    fig_data = np.ceil(fig_data)
    fig_data_copy = fig_data
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
    result = test(model=net, figure_data=fig_data_copy, target=anwser[i])
    if result:
        count_pytoch += 1

print(f'correct from zehan"s code: {correct_count}')
print(f'correct from pytorch: {count_pytoch}')
