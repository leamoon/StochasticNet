from turtle import forward
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, tanh

kernel = np.genfromtxt("CNN weights/conv_kernel_16.txt") / 16
#kernel = np.genfromtxt("CNN weights/conv1_weight.txt")
hidden_layer1_linear_trans = np.genfromtxt("CNN weights/fc1_weight.txt")
hidden_layer2_linear_trans = np.genfromtxt("CNN weights/fc2_weight.txt")
output_layer_linear_trans = np.genfromtxt("CNN weights/fc3_weight.txt")
# hidden_layer1_linear_trans = torch.Tensor(np.load("weight_data/hidden1.npy"))
# hidden_layer2_linear_trans = torch.Tensor(np.load("weight_data/hidden2.npy"))
# output_layer_linear_trans = torch.Tensor(np.load("weight_data/out_weight.npy"))

#hidden_layer1_linear_trans = np.round(hidden_layer1_linear_trans * 33) / 33
anwser = np.genfromtxt("DNN weights/value_list.txt")

correct_count = 0
wrong_count = 0

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

for i in range(10000):
    #if int(anwser[i]) != 9:
    #    continue

    fig_data = np.genfromtxt("data_figures/fig{}.txt".format(i))
    #fig_data = (fig_data > 0.5) # 二值化
    fig_data = np.ceil(fig_data) # 二值化
    #fig_data = np.round(fig_data) # 二值化
    #plt.imshow(fig_data, cmap='Greys')
    #plt.show()

    #input_layer = np.reshape(fig_data, (256, 1))
    conv_result = conv_2d_single_kernel(fig_data[0:15, 0:15], kernel, (2, 2))
    #print(conv_result[1:7,1:7])
    conv_result[conv_result < 0] = 0
    conv_result[conv_result > 1] = 1
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
        wrong_count += 1
        print('F')
    #print(output_layer_tanh)

print(correct_count, wrong_count)