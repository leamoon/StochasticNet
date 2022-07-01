from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, tanh

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

    # padding_h = (kernel_h-1) // 2
    # padding_w = (kernel_w-1) // 2
    # padding_data = np.zeros((h+padding_h*2, w+padding_w*2))
    # padding_data[padding_h:-padding_h, padding_w:-padding_w] = input_data

    out = np.zeros((h//stride_h, w//stride_w))
    for idx_h, i in enumerate(range(0, h-kernel_h+1, stride_h)):
        for idx_w, j in enumerate(range(0, w-kernel_w+1, stride_w)):
            window = input_data[i:i+kernel_h, j:j+kernel_w]
            out[idx_h, idx_w] = np.sum(window*kernel)
    return out

def cut_off_bipolar(arr):
    arr[arr < -1] = -1
    arr[arr > 1] = 1

if __name__ == "__main__":
    kernel = np.array([[4,7,2],[5,5,1],[-2,-1,0]])/16
    #print(np.round(kernel*16))
    #fig_data = np.genfromtxt("data_figures/fig4.txt")
    fig_data = np.genfromtxt("test_written_number.txt")
    #fig_data = (fig_data > 0.5)
    fig_data = np.ceil(fig_data)
    # plt.imshow(fig_data, cmap='Greys')
    # plt.show()

    conv_result = conv_2d_single_kernel(fig_data[0:15, 0:15], kernel, (2, 2))
    #print(conv_result[1:7,1:7])
    conv_result[conv_result < 0] = 0
    conv_result[conv_result > 1] = 1
    relu_output = np.reshape(conv_result, (49, 1))

    relu_output = relu_output

    hidden_layer1_linear_trans = np.genfromtxt("CNN weights/fc1_weight.txt")
    #cut_off_bipolar(hidden_layer1_linear_trans)
    hidden_layer2_linear_trans = np.genfromtxt("CNN weights/fc2_weight.txt")
    #cut_off_bipolar(hidden_layer2_linear_trans)
    output_layer_linear_trans = np.genfromtxt("CNN weights/fc3_weight.txt")
    #cut_off_bipolar(output_layer_linear_trans)
    
    hidden_layer1_result = dot(hidden_layer1_linear_trans, relu_output)
    hidden_layer1_output = tanh(hidden_layer1_result)
    #hidden_layer1_output = np.array([11,11,2,3,0,7,11,11,9,11,15,14,8,6,6, 1,4,7,4,2,10,4,10,12,6,7,5,7,1,12])/8-1

    hidden_layer2_result = dot(hidden_layer2_linear_trans, hidden_layer1_output)
    hidden_layer2_output = tanh(hidden_layer2_result)

    #print(np.round((relu_output.transpose())*16))
    print(np.round((hidden_layer2_output.transpose() + 1)*8))
    hidden_layer2_output = np.array([3,10,6,8,9,7,14,9,3,6,2,7,12,12,2, 13,12,6,8,1,9,6,3,4,6,3,11,7,9,9])/8-1
    #hidden_layer2_output = np.round((hidden_layer2_output + 1)*8)/8-1
    #print(np.sum(output_layer_linear_trans[3]*hidden_layer2_output.transpose()))

    #np.savetxt("CNN weights/fc3_input.txt", hidden_layer2_output.transpose())

    output_layer_result = dot(output_layer_linear_trans, hidden_layer2_output)
    output_layer_tanh = tanh(output_layer_result)
    print(output_layer_tanh.transpose())