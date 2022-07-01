from json.encoder import INFINITY
from nbt import nbt
from numpy import Inf, empty
import numpy as np

stackable_object_label_list = ["minecraft:red_concrete", "minecraft:green_concrete", "minecraft:blue_concrete",\
    "minecraft:cyan_concrete", "minecraft:black_concrete", "minecraft:white_concrete", "minecraft:gray_concrete",
    "minecraft:light_blue_concrete", "minecraft:magenta_concrete", "minecraft:purplr_concrete"]
# use different kind of concrete to avoid stacking

def set_value_source(container_source, numerator, denominator):
    numerator = int(numerator)
    denominator = int(denominator)
    container_source['Items'] = nbt.TAG_List(type=nbt.TAG_Compound)
    if numerator == 1 and denominator == 1: # if a/b=1, just set 1 minecart
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = 0, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = "minecraft:minecart", name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_source['Items'].tags.append(item)
        return
    if numerator == 0: # if a/b=0, just set 1 concrete
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = 0, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = stackable_object_label_list[1], name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_source['Items'].tags.append(item)
        return
    for i in range(min(denominator - numerator, denominator - 1)): # fill non-stackable objects
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = i, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = stackable_object_label_list[i], name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_source['Items'].tags.append(item)
    for i in range(denominator - numerator, denominator - 1): # fill minecarts
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = i, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = "minecraft:minecart", name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_source['Items'].tags.append(item)

def set_value_detected(container_detected, numerator = 0, denominator = 1):
    container_detected['Items'] = nbt.TAG_List(type=nbt.TAG_Compound)
    if numerator != 0: # otherwise (i.e. a/b=0) set one concrete
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = 0, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = "minecraft:minecart", name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_detected['Items'].tags.append(item)
    else:
        item = nbt.TAG_Compound(name = "")
        item.tags.append(nbt.TAG_Byte(value = 0, name = "Slot"))
        item.tags.append(nbt.TAG_String(value = stackable_object_label_list[0], name = "id"))
        item.tags.append(nbt.TAG_Byte(value = 1, name = "Count"))
        container_detected['Items'].tags.append(item)

def equal_position(a, b): # if position is equal
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

def set_weight(nbtfile, weight_num, weight_den, crystall_period, source_pos, detected_pos):
    for i in nbtfile['BlockEntities']:
        if str(i["Id"]) == "minecraft:dropper":
            # pos_in_lattice is the relative position with respect to the lattice original point
            # crystall_period is the size (or shape) of the lattice in x,y,z
            # index denotes which lattice to fill, index [i,j,k] fills weight[i,j,k]
            pos_in_lattice = \
                (i["Pos"][0] % crystall_period[0], i["Pos"][1] % crystall_period[1], i["Pos"][2] % crystall_period[2])
            index = \
                (i["Pos"][0] // crystall_period[0], i["Pos"][1] // crystall_period[1], i["Pos"][2] // crystall_period[2])
            if equal_position(pos_in_lattice, detected_pos):
                set_value_detected(i, weight_num[index[0], index[1], index[2]], weight_den[index[0], index[1], index[2]])
            if equal_position(pos_in_lattice, source_pos):
                set_value_source(i, weight_num[index[0], index[1], index[2]], weight_den[index[0], index[1], index[2]])

def nearest_fraction(x, N): # output the nearest a/b to x, b <= N, also output error = |a/b-x|
    err = INFINITY
    a, b = 0, 0
    for j in range(1, N + 1):
        i = round(x * j)
        current_err = abs(i / j - x)
        if current_err < err:
            a, b = i, j
            err = current_err
    return a, b, err

def bipolar_to_frequency(x): # x in [-1,1] to [0,1], cut off x<-1 and x>1
    if x < -1:
        x = -1
    if x > 1:
        x = 1
    return (x + 1) / 2

def check_empty_dropper(filename, pos = (0,0,0)):
    def add_position(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    nbtfile = nbt.NBTFile(filename,'rb')
    for i in nbtfile['BlockEntities']:
        if str(i["Id"]) == "minecraft:dropper":
            if len(i["Items"]) == 0:
                new_pos = add_position(i["Pos"], pos)
                print('/tp', new_pos[0], '~', new_pos[2])

def check_anomalous_dropper(filename, pos = (0,0,0), layer=''):
    crystall_period = (0, 0, 0)
    detected = (0, 0, 0)
    if layer == 'fc1':
        crystall_period = (8, 2, 5)
        detected = (6, 0, 3)
    elif layer == 'fc2':
        crystall_period = (5, 2, 8)
        detected = (1, 1, 5)
    elif layer == 'fc3':
        crystall_period = (8, 2, 5)
        detected = (6, 0, 2)
    else:
        print("input correct layer")
        return

    def add_position(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    nbtfile = nbt.NBTFile(filename,'rb')
    get_at_least_one_detected = False
    for i in nbtfile['BlockEntities']:
        if str(i["Id"]) == "minecraft:dropper":
            pos_in_lattice = \
                (i["Pos"][0] % crystall_period[0], i["Pos"][1] % crystall_period[1], i["Pos"][2] % crystall_period[2])
            if not get_at_least_one_detected and equal_position(pos_in_lattice, detected):
                get_at_least_one_detected = True
            if equal_position(pos_in_lattice, detected) and len(i["Items"]) > 1:
                new_pos = add_position(i["Pos"], pos)
                print('/tp', new_pos[0], '~', new_pos[2])
    if not get_at_least_one_detected:
        print("the position might be wrong")

def fill_fc1_weight():
    weight = np.genfromtxt("CNN weights/fc1_weight.txt")[:,1:]
    weight = weight * 0.5 # factor 0.5

    shape = (15, 3, 16) # Half of the matrix 30*48 to shape (15, 3, 16) in x,y,z, y being the height
    weight = np.reshape(weight[:15], shape) # upper half
    # weight = np.reshape(weight[15:], shape) # lower half
    weight = np.flip(weight, axis=2) # since our setting has reversed direction w.r.t. z-axis
    weight = np.flip(weight, axis=1) # since our setting has reversed direction w.r.t. y-axis
    device_shape = shape # integral range [-1,1]
    weight_num = np.ones(device_shape) # weight-numerator
    weight_den = np.zeros(device_shape) # weight-denominator
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                a, b, err = nearest_fraction(bipolar_to_frequency(weight[i,j,k]), 10)
                # print(a, b)
                weight_num[i,j,k] = a
                weight_den[i,j,k] = b
                weight[i,j,k] = a / b
    print(weight[14,::-1,::-1])
    
    nbtfile = nbt.NBTFile("fc1_weight_upper_half.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 0, 3) # this two can be determined by print the positions of some droppers, using the code below
    source = (7, 0, 3)
    set_weight(nbtfile, weight_num, weight_den, crystall_period, source, detected)
    nbtfile.write_file("fc1_weight_upper_half_fill.schem")
    # nbtfile.write_file("fc1_weight_lower_half_fill.schem")

def fill_fc2_weight():
    weight = np.genfromtxt("CNN weights/fc2_weight.txt")
    weight = weight * 0.5 # factor 0.5

    shape = (15, 2, 15) # Half of the matrix 30*30 to shape (15, 3, 15)
    new_shape = (15, 2, 15) # after swapping x, z
    weight = np.reshape(weight[:15], shape) # upper half
    # weight = np.reshape(weight[15:], shape) # lower half
    # print(weight[0])
    weight = np.swapaxes(weight, 0, 2) # swap x, z
    weight = np.flip(weight, axis=2) # since our setting has reversed direction w.r.t. z-axis
    weight = np.flip(weight, axis=1) # since our setting has reversed direction w.r.t. y-axis
    device_shape = new_shape # integral range [-1,1]
    weight_num = np.ones(device_shape) # weight-numerator
    weight_den = np.zeros(device_shape) # weight-denominator
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                a, b, err = nearest_fraction(bipolar_to_frequency(weight[i,j,k]), 10)
                # print(a, b)
                weight_num[i,j,k] = a
                weight_den[i,j,k] = b
                weight[i,j,k] = a / b
    print(weight[:,::-1,-1].transpose())
    
    nbtfile = nbt.NBTFile("fc2_weight_upper_half.schem",'rb')
    crystall_period = (5, 2, 8)
    detected = (1, 1, 5) # this two can be determined by print the positions of some droppers, using the code below
    source = (1, 1, 6)
    
    set_weight(nbtfile, weight_num, weight_den, crystall_period, source, detected)
    nbtfile.write_file("fc2_weight_upper_half_fill.schem")
    # nbtfile.write_file("fc2_weight_lower_half_fill.schem")

def fill_fc3_weight():
    weight = np.genfromtxt("CNN weights/fc3_weight.txt")
    weight = weight * 0.5 # factor 0.5

    shape = (10, 2, 15) # 10*30 to shape (15, 3, 15)
    new_shape = shape
    weight = np.reshape(weight, shape)
    # print(weight[0])
    weight = np.flip(weight, axis=2) # since our setting has reversed direction w.r.t. z-axis
    weight = np.flip(weight, axis=1) # since our setting has reversed direction w.r.t. y-axis
    weight = np.flip(weight, axis=0) # since our setting has reversed direction w.r.t. x-axis
    device_shape = new_shape # integral range [-1,1]
    weight_num = np.ones(device_shape) # weight-numerator
    weight_den = np.zeros(device_shape) # weight-denominator
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                a, b, err = nearest_fraction(bipolar_to_frequency(weight[i,j,k]), 10)
                # print(a, b)
                weight_num[i,j,k] = a
                weight_den[i,j,k] = b
                weight[i,j,k] = a / b
    print(weight[-1,::-1,::-1])
    
    nbtfile = nbt.NBTFile("fc3_weight.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 0, 2) # this two can be determined by print the positions of some droppers, using the code below
    source = (7, 0, 2)
    # for i in range(0, 1000):
    #     if str(nbtfile['BlockEntities'][i]['Id']) == "minecraft:dropper":
    #         print(nbtfile['BlockEntities'][i]['Pos'])
    set_weight(nbtfile, weight_num, weight_den, crystall_period, source, detected)
    nbtfile.write_file("fc3_weight_fill.schem")

def test_fill_fc1_input(): # fill input of 1st layer
    input = np.genfromtxt("CNN weights/fc1_input.txt")[1:]
    test_weight = np.genfromtxt("CNN weights/fc1_weight.txt")[13,1:] * 0.5

    input[input > 1] = 1
    input[input < -1] = -1
    test_weight[test_weight > 1] = 1
    test_weight[test_weight < -1] = -1
    #print(np.dot(input, test_weight), np.tanh(np.dot(input, test_weight)))

    shape = (1, 3, 16)
    input = np.resize(input, shape)
    test_weight = np.resize(test_weight, shape)
    test_weight = np.flip(test_weight, axis=2) 
    test_weight = np.flip(test_weight, axis=1) 
    input = np.flip(input, axis=2) 
    input = np.flip(input, axis=1) 
    input_num = np.ones(shape)
    input_den = np.zeros(shape)
    test_weight_num = np.zeros(shape)
    test_weight_den = np.zeros(shape)
    print(np.round(input*16))
    for j in range(shape[1]):
        for k in range(shape[2]):
            a, b, err = nearest_fraction(bipolar_to_frequency(input[0,j,k]), 10)
            # print(a, b, bipolar_to_frequency(input[0,j,k]))
            input_num[0,j,k] = a
            input_den[0,j,k] = b
            input[0,j,k] = a / b# * 2 - 1
            a, b, err = nearest_fraction(bipolar_to_frequency(test_weight[0,j,k]), 10)
            test_weight_num[0,j,k] = a
            test_weight_den[0,j,k] = b
            test_weight[0,j,k] = a / b# * 2 - 1
    #print(np.sum(input * test_weight))
    print(test_weight)

    nbtfile = nbt.NBTFile("test_fc1_input_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (0, 0, 0)
    source = (1, 0, 0)
    set_weight(nbtfile, input_num, input_den, crystall_period, source, detected)
    nbtfile.write_file("test_fc1_input_fill.schem")
    
    nbtfile2 = nbt.NBTFile("test_fc1_weight_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 0, 3)
    source = (7, 0, 3)
    set_weight(nbtfile2, test_weight_num, test_weight_den, crystall_period, source, detected)
    nbtfile2.write_file("test_fc1_weight_fill.schem")

def test_fill_fc2_input(): # fill input of 2nd layer
    input = np.genfromtxt("CNN weights/fc2_input.txt")
    test_weight = np.genfromtxt("CNN weights/fc2_weight.txt")[12] * 0.5

    input[input > 1] = 1
    input[input < -1] = -1
    test_weight[test_weight > 1] = 1
    test_weight[test_weight < -1] = -1
    #print(np.dot(input, test_weight), np.tanh(np.dot(input, test_weight)))

    shape = (1, 2, 15)
    new_shape = (15, 2, 1) # after swapping x, z
    input = np.resize(input, shape)
    test_weight = np.resize(test_weight, shape)
    test_weight = np.swapaxes(test_weight, 0, 2) # swap x, z
    test_weight = np.flip(test_weight, axis=2) 
    test_weight = np.flip(test_weight, axis=1) 
    input = np.swapaxes(input, 0, 2) # swap x, z
    input = np.flip(input, axis=2) 
    input = np.flip(input, axis=1) 
    input_num = np.ones(new_shape)
    input_den = np.zeros(new_shape)
    test_weight_num = np.zeros(new_shape)
    test_weight_den = np.zeros(new_shape)
    #print(np.round(input*16))
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            a, b, err = nearest_fraction(bipolar_to_frequency(input[i,j,0]), 10)
            # print(a, b, bipolar_to_frequency(input[0,j,k]))
            input_num[i,j,0] = a
            input_den[i,j,0] = b
            input[i,j,0] = a / b
            a, b, err = nearest_fraction(bipolar_to_frequency(test_weight[i,j,0]), 10)
            test_weight_num[i,j,0] = a
            test_weight_den[i,j,0] = b
            test_weight[i,j,0] = a / b
    print(np.tanh(np.sum((input*2-1) * (test_weight*2-1))))
    print(input[:,::-1,-1].transpose())
    print(test_weight[:,::-1,-1].transpose())

    nbtfile = nbt.NBTFile("test_fc2_input_empty.schem",'rb')
    crystall_period = (5, 2, 8)
    detected = (0, 0, 0)
    source = (0, 0, 1)
    # for i in range(0, 200):
    #     if str(nbtfile['BlockEntities'][i]['Id']) == "minecraft:dropper":
    #         print(nbtfile['BlockEntities'][i]['Pos'])
    set_weight(nbtfile, input_num, input_den, crystall_period, source, detected)
    nbtfile.write_file("test_fc2_input_fill.schem")
    '''
    nbtfile2 = nbt.NBTFile("test_fc2_weight_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 0, 3)
    source = (7, 0, 3)
    set_weight(nbtfile2, test_weight_num, test_weight_den, crystall_period, source, detected)
    nbtfile2.write_file("test_fc2_weight_fill.schem")
    '''

def test_fill_fc3_input(): # fill input of 3rd layer
    input = np.genfromtxt("CNN weights/fc3_input.txt")
    test_weight = np.genfromtxt("CNN weights/fc3_weight.txt")[3] * 0.5

    input[input > 1] = 1
    input[input < -1] = -1
    test_weight[test_weight > 1] = 1
    test_weight[test_weight < -1] = -1
    #print(np.dot(input, test_weight), np.tanh(np.dot(input, test_weight)))

    shape = (1, 2, 15)
    input = np.resize(input, shape)
    test_weight = np.resize(test_weight, shape)
    test_weight = np.flip(test_weight, axis=2)
    test_weight = np.flip(test_weight, axis=1)
    input = np.flip(input, axis=2) 
    input = np.flip(input, axis=1) 
    input_num = np.ones(shape)
    input_den = np.zeros(shape)
    test_weight_num = np.zeros(shape)
    test_weight_den = np.zeros(shape)
    print(np.sum(input*test_weight))
    for j in range(shape[1]):
        for k in range(shape[2]):
            a, b, err = nearest_fraction(bipolar_to_frequency(input[0,j,k]), 10)
            # print(a, b, bipolar_to_frequency(input[0,j,k]))
            input_num[0,j,k] = a
            input_den[0,j,k] = b
            input[0,j,k] = a / b# * 2 - 1
            a, b, err = nearest_fraction(bipolar_to_frequency(test_weight[0,j,k]), 10)
            test_weight_num[0,j,k] = a
            test_weight_den[0,j,k] = b
            test_weight[0,j,k] = a / b# * 2 - 1
    print(np.tanh(np.sum((input*2-1) * (test_weight*2-1))))
    print(input[:,::-1,::-1])
    print(test_weight[:,::-1,::-1])

    nbtfile = nbt.NBTFile("test_fc3_input_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (0, 0, 0)
    source = (1, 0, 0)
    set_weight(nbtfile, input_num, input_den, crystall_period, source, detected)
    nbtfile.write_file("test_fc3_input_fill.schem")
    '''
    nbtfile2 = nbt.NBTFile("test_fc1_weight_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 0, 3)
    source = (7, 0, 3)
    set_weight(nbtfile2, test_weight_num, test_weight_den, crystall_period, source, detected)
    nbtfile2.write_file("test_fc1_weight_fill.schem")
    '''

# when writing the functions, pay attention to the actual orientation of the device
def test_fill_fc3_weight_double(): # fill 3rd layer weight, testing, [-2,2] range
    weight = np.genfromtxt("CNN weights/fc3_weight.txt")
    shape = (10, 2, 15) # Matrix 10*30 to shape (10, 2, 15) in x,y,z, y being the height
    weight = np.reshape(weight, shape)
    device_shape = (10, 4, 15) # 4 = 2*2, 2 is the integral range [-2,2], see the 1st stage report
    weight_num = np.ones(device_shape) # weight-numerator
    weight_den = np.zeros(device_shape) # weight-denominator
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # divided by 2 since we are using [-2,2] range
                a, b, err = nearest_fraction(bipolar_to_frequency(weight[i,j,k] / 2), 10)
                weight_num[i,j * 2,k] = a
                weight_num[i,j * 2 + 1,k] = a
                weight_den[i,j * 2,k] = b
                weight_den[i,j * 2 + 1,k] = b
    
    nbtfile = nbt.NBTFile("test_weight_empty.schem",'rb')
    crystall_period = (8, 2, 5)
    detected = (6, 1, 3) # this two can be determined by print the positions of some droppers, using the code below
    source = (7, 1, 3)
    # for i in range(1200, 1300):
    #     if str(nbtfile['BlockEntities'][i]['Id']) == "minecraft:dropper":
    #         print(nbtfile['BlockEntities'][i]['Pos'])
    #set_weight(nbtfile, weight_num, weight_den, crystall_period, source, detected)
    #nbtfile.write_file("test_weight_fill.schem")

def test_fill_fc3_input_double(): # fill 3rd layer input, testing, [-2,2] range
    input = np.genfromtxt("CNN weights/fc3_input.txt")
    shape = (1, 2, 15)
    input = np.resize(input, shape)
    input_num = np.ones(shape)
    input_den = np.zeros(shape)
    for j in range(shape[1]):
        for k in range(shape[2]):
            a, b, err = nearest_fraction(bipolar_to_frequency(input[0,j,k]), 10)
            # print(a, b, bipolar_to_frequency(input[0,j,k]))
            input_num[0,j,k] = a
            input_den[0,j,k] = b
    nbtfile = nbt.NBTFile("test_input_empty.schem",'rb')
    crystall_period = (8, 4, 5)
    detected = (0, 0, 0)
    source = (1, 0, 0)
    # for i in range(0, 20):
    #     if str(nbtfile['BlockEntities'][i]['Id']) == "minecraft:dropper":
    #         print(nbtfile['BlockEntities'][i]['Pos'])
    set_weight(nbtfile, input_num, input_den, crystall_period, source, detected)
    nbtfile.write_file("test_input_fill.schem")

if __name__ == "__main__":
    #test_fill_fc2_input()
    #fill_fc2_weight()

    # check_empty_dropper("fc1_weight_upper_half_check.schem")
    # check_empty_dropper("test_fc1_input_check.schem")

    check_empty_dropper("test_dropper_empty.schem", pos=(-244,0,-415))
    check_anomalous_dropper("test_dropper_empty.schem", pos=(-244,0,-415), layer='fc3')
