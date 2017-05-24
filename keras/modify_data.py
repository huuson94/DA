import math
import numpy as np
from random import uniform
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ca_alg():
    #constant
    a = 0.6380631366077803
    b = 0.5959486060529070
    q = 0.9339962957603650
    W = 0.2488702280083841
    A = 0.6366197723675813
    B = 0.5972997593539963
    H = 0.0214949004570452
    P = 4.9125013953033204

    U = uniform(0, 1)
    T = U - 0.5
    S = W - T * T
    if(S > 0):
        Z = T * (A / S + B)
        return Z
    while True:
        U = uniform(0, 1)
        U_ = uniform(0, 1)
        T = U - 0.5
        S = 1/4 - T * T
        Z = T * (a / S + b)
        if((S * S * ((1 + Z  * Z )*(H*U_ + P ) - q) + S) <= 0.5):
            break
    return Z


def ea_alg():
    #constant
    q = math.log(2)
    a = 5.7133631526454228
    b = 1.4142135623730950
    b = 3.4142135623730950
    c = -1.6734053240284925
    p = 0.9802581434685472
    A = 5.6005707569738080
    B = 3.3468106480569850
    H = 0.0026106723602095
    D = 0.0857864376269050

    U = uniform(0, 1)
    K = c
    U = U + U
    while (U < 1):
        U = U + U
        K = K + q
    U = U - 1
    if (U <= p):
        Z = K + A / (B - U)
        return Z
    while True:
        U = uniform(0, 1)
        U_ = uniform(0, 1)
        Y = a / (b - U)
        if( (U_ * H + D) * (b - U) * (b - U) < math.exp(-Y - c)):
            break
    Z = K + Y
    return Z

def na_alg():
    U = uniform(0,1)
    if(U < 1/2):
        B = 0
    else:
        B = 1
    V = ea_alg()
    S = V + V
    W = ca_alg()
    Z = math.sqrt(S / (1 + W * W))
    Y = W*Z
    if B == 0:
        return Z, Y
    else:
        return -Z, Y


def modify(inputs, epoch):
    rs = []
    input_shape = inputs[0].shape # color, w, h
    #rectangle_size = np.random.randint(low=5, high=input_shape[1], size=2)#w, h
    rectangle_size = [input_shape[1], input_shape[2]]
    part_w = input_shape[1] / rectangle_size[0]
    part_h = input_shape[2] / rectangle_size[1]
    number_parts = (input_shape[1] * input_shape[2]) / (part_w * part_h)
    number_parts = int(number_parts)
    samples_count = 5* np.log(epoch/2 + 1) + 1
    samples_count = int(samples_count)
    for input in inputs:
        #plt.imsave('before.png', np.swapaxes(np.swapaxes(input, 0, 1), 1, 2))
        #_, w, h = input.shape
        #number_parts_width = int(np.random.uniform(1, w))
        #number_parts_height = int(np.random.uniform(1, h))

        modified_input = input_random_by_part(input, part_w, part_h, number_parts, np.mean(np.random.normal(0.0, 1,size=samples_count)))
        #plt.imsave('after.png', np.swapaxes(np.swapaxes(modified_input, 0, 1), 1, 2))
        rs.append(modified_input)
    return np.array(rs, dtype="float32")

def input_random_by_part(input, part_w = 0, part_h = 0, number_parts = 1, change = 0):
    color, w, h = input.shape
    if number_parts == 1:
        index_part = 0
    else:
        index_part = np.random.randint(1, number_parts) - 1
    #print(input[0, index_part * width:((index_part) * width + 2), index_part * height:((index_part + 1) *height + 1)])
    start_x = index_part * part_w
    start_y = index_part * part_h
    if(color == 1):
        channel_random = 0
    else:
        channel_random = np.random.choice(3,2)
    input[channel_random, start_x:start_x + part_w, start_y:start_x + part_h] = \
        input[channel_random, start_x:start_x + part_w, start_y:start_x + part_h] + change
    #print(input[0, index_part * width:((index_part) * width + 2), index_part * height:((index_part + 1) *height + 1)])
    return input
#rotate image
def modify_1(inputs, epoch):
    rs = []
    samples_count = 7* np.log(epoch+1) + 1
    samples_count = int(samples_count)
    for index, input in enumerate(inputs):
        #plt.imsave(str(index) + 'before.png', np.swapaxes(np.swapaxes(input, 0, 1), 1, 2))
	rotated_image = np.swapaxes(np.swapaxes(input, 0, 1), 1, 2)
	rotated_image = misc.imrotate(rotated_image, np.mean(np.random.normal(0,10,samples_count)), interp='nearest')
	#plt.imsave(str(index) + 'rotate.png', rotated_image)
        rs.append(np.swapaxes(np.swapaxes(rotated_image, 1, 2), 0, 1))

    return np.array(rs, dtype="float32")

#rotate image
def modify_2(inputs, epoch):
    rs = []
    import matplotlib.pyplot as plt
    samples_count = 7* np.log(epoch+1) + 1
    samples_count = int(samples_count)
    for index, input in enumerate(inputs):
        #plt.imsave(str(index) + 'before.png', np.swapaxes(np.swapaxes(input, 0, 1), 1, 2))
	rotated_image = np.swapaxes(np.swapaxes(input, 0, 1), 1, 2)
	rotated_image = misc.imrotate(rotated_image, np.mean(np.random.normal(0,10,samples_count)), interp='nearest')
	#plt.imsave(str(index) + 'rotate.png', rotated_image)
        rs.append(np.swapaxes(np.swapaxes(rotated_image, 1, 2), 0, 1))
	if(index == 10):
	    exit(0)
    return np.array(rs, dtype="float32")

#Tuan
def modify_3(inputs, epoch):
    rec_count = int(10/(epoch //100 + 1))
    variance = 0.5/(epoch // 100 + 1)
    sample_count = 10 * (epoch // 100 + 1)
    modify_rec = []
    channel_random = []
    rs = []
    for index, input in enumerate(inputs):
        for i in range(rec_count):
            i, j = np.random.randint(23,size = 2)
            #i = np.random.randint(0,23)
            #j = np.random.randint(0,23)
            i_dis = np.random.randint(8,31 -i)
            j_dis = np.random.randint(8,31 -j)
	    modify_rec = np.random.normal(0, variance, (sample_count, 3, i_dis, j_dis))
 	    modify_rec = modify_rec.reshape(sample_count, 3, i_dis, j_dis)
	    #modify_rec = np.zeros([sample_count, 3, i_dis, j_dis]) + 0.3
            modify_rec = np.mean(modify_rec, axis=0)
	    channel_random = np.random.randint(0,2,np.random.randint(1,2))
	    input[channel_random, i:i+i_dis,j:j+j_dis] = input[channel_random, i:i+i_dis,j:j+j_dis] + modify_rec[channel_random]
	rs.append(input)
    return np.array(rs, dtype="float32")

#ope
def modify_4(inputs, epoch, at, bt):
    #modify
    rs = np.zeros(inputs.shape)
    for iter_train in range(len(inputs)):
        ope = bt[iter_train] - at[iter_train]
        rs[iter_train] = modify_ope(inputs[iter_train],ope,(float(epoch+4))**0.35)
        if ope == 0:
            bt[iter_train] += 1
        else:
            at[iter_train] += 1
    return np.array(rs, dtype="float32")

def modify_ope(img,ope,CountSample):

    img_modify = np.zeros((3,32,32),dtype = 'float32')
    img_modify[:,:,:] = img[:,:,:]

    i = np.random.random_integers(0,15)
    j = np.random.random_integers(0,15)
    i_dis = np.random.random_integers(16,31 -i)
    j_dis = np.random.random_integers(16,31 -j)
    k = 0
    k_dis = 3
    img_modify[:,i:i + i_dis +1,j:j+j_dis+1] += ope*(2.0*np.random.random((3,i_dis +1,j_dis+1))-1.0)/CountSample
    return img_modify

#use origin data, modify date interchange
def modify_5(inputs, pre_inputs, y_train, pre_y_train, epoch):
    if((epoch + 1) % 2 == 0):
        return inputs
    else:
        return modify_7(inputs, pre_inputs, y_train, pre_y_train)

def modify_6(inputs, epoch):
    if(epoch // 100 % 2 == 0):
        return inputs
    else:
        return modify(inputs, epoch)

#combine 3 to 1. only train new created data
def modify_7(x_train_ori, x_train_pre, y_train_ori, pre_y_train):
    new_inputs = np.empty([0, 3, 32, 32], dtype='float32') #new inputs
    new_y = np.empty(0,dtype='int32')

    y_train_ori_by_class = []
    pre_y_train_by_class = []
    for i in range(0, 10):
        y_train_ori_by_class.append(np.argwhere(y_train_ori == i))
        pre_y_train_by_class.append(np.argwhere(pre_y_train == i))
        images_each_class_count = len(y_train_ori_by_class[i])
        #from origin
        indices = np.random.randint(images_each_class_count, size=images_each_class_count)
        images = np.reshape(x_train_ori[y_train_ori_by_class[i][indices]],(images_each_class_count, 3, 32, 32))
        #from epoch - 1 round
        indices = np.random.randint(images_each_class_count, size=images_each_class_count)
        pre_images = np.reshape(x_train_pre[pre_y_train_by_class[i][indices]],(images_each_class_count, 3, 32, 32))

        new_images = np.mean(np.array([images,pre_images]), axis=0)
        new_inputs = np.concatenate((new_inputs, new_images), axis=0)
        empty_labels = np.empty(images_each_class_count, dtype='int32')
        empty_labels.fill(i)
        new_y = np.append(new_y, empty_labels)
    new_y_train = new_y.flatten()
    return new_inputs, new_y_train

# combine ori and previous to new. Train lan luot 1new-1ori
def modify_8(x_train_ori, x_train_pre, y_train_ori, y_train_pre):
    new_inputs = np.empty([0, 3, 32, 32], dtype='float32') #new inputs
    new_y = np.empty(0,dtype='int32')

    y_train_ori_by_class = []
    pre_y_train_by_class = []
    X_train = np.empty([0, 3, 32, 32], dtype='float32') #half of new and origin
    y_train = np.empty(0,dtype='int32')
    for i in range(0, 10):
        y_train_ori_by_class.append(np.argwhere(y_train_ori == i))
        pre_y_train_by_class.append(np.argwhere(y_train_pre == i))
        images_each_class_count = len(y_train_ori_by_class[i])
        #from origin
        indices = np.random.randint(len(y_train_ori_by_class[i]), size=images_each_class_count)
        images = np.reshape(x_train_ori[y_train_ori_by_class[i][indices]],(images_each_class_count, 3, 32, 32))
        #from epoch - 1 round
        indices = np.random.randint(len(pre_y_train_by_class[i]), size=images_each_class_count)
        pre_images = np.reshape(x_train_pre[pre_y_train_by_class[i][indices]],(images_each_class_count, 3, 32, 32))

        new_images = np.mean(np.array([images,pre_images]), axis=0)

        #half of origin merge with half of new ones
        indices = np.random.randint(int(images_each_class_count/2), size=(2, int(images_each_class_count/2)))
        x = np.concatenate((np.reshape(x_train_ori[y_train_ori_by_class[i][indices[0]]], (int(images_each_class_count/2), 3, 32, 32)), new_images[indices[1]]), axis=0)
        X_train = np.concatenate((X_train, x), axis=0)
        if(2*int(images_each_class_count/2) < images_each_class_count):
            X_train = np.concatenate((X_train, [x_train_ori[np.random.choice(y_train_ori_by_class[i].flatten())]]), axis=0)

        new_inputs = np.concatenate((new_inputs, new_images), axis=0)
        empty_labels = np.empty(images_each_class_count, dtype='int32')
        empty_labels.fill(i)
        y_train = np.append(y_train, empty_labels)
        new_y = np.append(new_y, empty_labels)
    new_y_train = new_y.flatten()
    y_train = y_train.flatten()
    return new_inputs, new_y_train, X_train, y_train


#combine 3 to 1. add to ori then train origin-train 1-1
def modify_9(x_train_ori, x_train_pre, y_train_ori, y_train_pre):
    img_shape = [1, 28, 28]
    new_inputs = np.empty([0, img_shape[0], img_shape[1], img_shape[2]], dtype='float32') #new inputs
    new_y = np.empty(0,dtype='int32')

    y_train_ori_by_class = []
    pre_y_train_by_class = []
    for i in range(0, 10):
        y_train_ori_by_class.append(np.argwhere(y_train_ori == i))
        pre_y_train_by_class.append(np.argwhere(y_train_pre == i))
        images_each_class_count = len(y_train_ori_by_class[i])
        indices = np.random.randint(images_each_class_count, size=(3 * images_each_class_count))
        #TODO: change x_train_ori to x_train_pre
        images_to_aver = np.reshape(x_train_ori[y_train_ori_by_class[i][indices]],
                                            (3, images_each_class_count, img_shape[0], img_shape[1], img_shape[2]) )
        new_images = np.mean(images_to_aver, axis=0)
        new_inputs = np.concatenate((new_inputs, new_images), axis=0)
        empty_labels = np.empty(images_each_class_count, dtype='int32')
        empty_labels.fill(i)
        new_y = np.append(new_y, empty_labels)
    new_y_train = new_y.flatten()
    return new_inputs, new_y_train
