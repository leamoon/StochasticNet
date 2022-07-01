from json.encoder import INFINITY
from nbt import nbt
from numpy import Inf, empty
import numpy as np
import os
from matplotlib import pyplot as plt

def get_number(i):
    return np.genfromtxt("codes/data_figures/fig{}.txt".format(i)) > 0.5 # 二值化

def plot(i):
    plt.imshow(get_number(i), cmap='Greys')
    plt.show()

def write_to_file(i):
    number_file = get_number(i)
    plot(i)
    nbtfile = nbt.NBTFile("nbt/test_white_board.schem",'rb')
    nbtfile["Palette"].tags.append(nbt.TAG_Int(value=2, name="minecraft:redstone_block"))
    nbtfile["PaletteMax"] = nbt.TAG_Int(value=3, name="PaletteMax")
    for i in range(15):
        for j in range(15):
            if number_file[i][j] > 0.5:
                k = (14 - i) * 15 * 4 + (14 - j) * 2 + 30 + 1
                print(k)
                nbtfile["BlockData"][k] = 2
    # for i in len(nbtfile['BlockData']):
    #     if str(nbtfile['BlockData'][i]) == str(nbtfile["Palette"]["minecraft:black_concrete"]):
    #         pass
    print(nbtfile["BlockData"])
    # print(nbtfile.pretty_tree())
    nbtfile.write_file("nbt/test_number_written.schem")

if __name__ == "__main__":
    os.chdir(os.path.pardir)
    write_to_file(15)