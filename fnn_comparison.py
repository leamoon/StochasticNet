import re
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, tanh
import torch
import torch.nn as nn
from torch.autograd import Variable

hidden_layer1_linear_trans = np.genfromtxt("hidden1_weight.txt")
hidden_layer2_linear_trans = np.genfromtxt("hidden2_weight.txt")
output_layer_linear_trans = np.genfromtxt("out_weight.txt")
anwser = np.genfromtxt("value_list.txt")

def test(model, figure_data, target):
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
        t_data = torch.Tensor(figure_data)
        target = torch.as_tensor(target)
        t_data = t_data.view(t_data.size(1), -1)
        t_data_binary = np.ceil(t_data.numpy())
        t_data = torch.from_numpy(t_data_binary)
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

class Net(nn.Module):

    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1, bias=False)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.out = nn.Linear(n_hidden2, n_output, bias=False)

    # connect inputs and outputs
    def forward(self, x):
        # print(f'input: {x}')
        x = torch.tanh(self.hidden1(x))
        # print(f'hidden1 output: {x}')
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.out(x))
        return x


size_inputs = 16*16
size_hidden1 = 32
size_hidden2 = 32
size_outputs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'  # in MNIST recognition 'GPu' is slower than 'cpu'
print(f"Using {device} device")

# create a network sample
net = Net(n_feature=size_inputs, n_hidden1=size_hidden1, n_hidden2=size_hidden2, n_output=size_outputs).to(device)
net = torch.load(f'weight_data\epoch_20')
# hidden_layer1_linear_trans = net.state_dict()['hidden1.weight'].numpy()
# hidden_layer2_linear_trans = net.state_dict()['hidden2.weight'].numpy()
# output_layer_linear_trans = net.state_dict()['out.weight'].numpy()
correct_count = 0
count_pytoch = 0

for i in range(1000):

    fig_data = np.genfromtxt("data_figures/fig{}.txt".format(i))
    #fig_data = (fig_data > 0.5) # 二值化
    #plt.imshow(fig_data, cmap='Greys')
    #plt.show()

    input_layer = np.ceil(np.reshape(fig_data, (256, 1)))
    # input_layer = torch.Tensor(input_layer)

    hidden_layer1_result = dot(hidden_layer1_linear_trans, input_layer)
    hidden_layer1_output = tanh(hidden_layer1_result)
    # print(f'output from non-pytorch: {hidden_layer1_output}')

    hidden_layer2_result = dot(hidden_layer2_linear_trans, hidden_layer1_output)
    hidden_layer2_output = tanh(hidden_layer2_result)

    output_layer_result = dot(output_layer_linear_trans, hidden_layer2_output)
    output_layer_tanh = tanh(output_layer_result)
    # print(f'input from non-pytorch: {input_layer.T}')
    # print(f'output from non-pytorch: {hidden_layer1_output.T}')

    number = np.argmax(output_layer_tanh)
    print(number, int(anwser[i]), end = ' ')
    if number == int(anwser[i]):
        correct_count += 1
        print('T')
    else:
        print('F')
    #print(output_layer_tanh)
    result = test(model=net, figure_data=input_layer, target=anwser[i])
    if result:
        count_pytoch += 1

print(correct_count)
print(f'correct from pytorch: {count_pytoch}')
