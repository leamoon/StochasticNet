from matplotlib import pyplot as plt
import numpy as np

fig_data = np.genfromtxt("data_figures/fig{}.txt".format(0))
#fig_data = np.genfromtxt("test_written_number.txt")
#print(fig_data)
fig_data = (fig_data > 0.5) # 二值化
plt.imshow(fig_data, cmap='Greys')
plt.show()

# kernel = np.genfromtxt("CNN weights/conv1_weight.txt")
# K = kernel.copy()
# K[kernel < 0] = 0
# S = np.sum(K)
# print(S)
# print(kernel / S)

# weight = np.genfromtxt("CNN weights/fc1_weight.txt")
# plt.imshow(weight, cmap='bwr')
# plt.show()

# print(np.max(weight), np.min(weight))