#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from training_process_file import save_params
from training_process_file import save_history
from training_process_file import read_params

from visualize import plot_loss
from visualize import plot_conv1_weights

from scipy import misc
from modify_data import modify
from modify_data import modify_1
from modify_data import modify_3
from modify_data import modify_4

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()    # Write the data of stdout here to a text file as well

def load_test_data():
    import random
    files = os.listdir('cards')
    files.sort()
    #random.shuffle(files)
    X_test = []
    for index, file in enumerate(files):
        if not file.startswith('.') and file.endswith('.png'):
            print(file)
            image = misc.imread('cards/' + file, mode='L').astype('float32')
            crop_image = center_crop(image)
            crop_image = np.reshape(crop_image, (28, 28, 1))
            X_test.append(crop_image)

    X_test = np.asarray(X_test, dtype='float32')
    X_test = (X_test - 128) / 128.0
    #X_test = np.swapaxes(np.swapaxes(np.swapaxes(X_test, 0, 3), 1, 3), 1, 2)
    X_test = X_test.transpose(0, 3, 1,2)
    return X_test

def center_crop(origin_img):
    if(len(origin_img.shape) == 2):
        height, width= origin_img.shape
        new_arr = np.zeros((height, width, 3))
        new_arr[:,:,0] =origin_img
        new_arr[:,:,1] =origin_img
        new_arr[:,:,2] =origin_img
        origin_img= new_arr

    height, width, color = origin_img.shape
    min_size = height if height <= width else width
    #center crop
    startX = width//2 - min_size//2
    startY = height//2 - min_size//2
    center_cropped_img = origin_img[startY:startY+min_size, startX:startX+min_size,:]
    img = misc.imresize(center_cropped_img, (28, 28), interp='nearest').astype('float32')
    return img[:,:,0]

def build_cnn(input_var=None):
    snapshot_root = 'snapshot_models/mnist/'
    trained_params = read_params(snapshot_root + '170315123731/999_params')
    #lasagne.layers.set_all_param_values(network, trained_params)

    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.GlorotUniform())
            W= trained_params[0], b = trained_params[1])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W= trained_params[2], b = trained_params[3])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W= trained_params[4], b = trained_params[5])
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            W= trained_params[6], b = trained_params[7])

    return network



def test_cards():
    filename = 'cards/rs.txt'
    rs_file = open(filename, "w")
    sys.stdout = Tee(sys.stdout, rs_file)

    X_test = load_test_data()
    t =X_test[0].transpose(1,2,0).reshape(28,28)
    test_img = np.asarray([t, t, t])
    test_img = test_img.transpose(1,2,0)
    print(test_img.shape)
    misc.imsave('test.png',test_img)
    input_var = T.tensor4('inputs')
    network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_function = theano.function([input_var], prediction)
    values = predict_function(X_test)
    for value in values:
        value = value.tolist()
        print(value.index(max(value)))

def main(model='mlp', num_epochs=500, start_epoch=0):
    test_cards()

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['start_epoch'] = int(sys.argv[3])
        main(**kwargs)
