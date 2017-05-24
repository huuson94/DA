import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


def build_lnet5(input_var=None):
    snapshot_root = 'snapshot_models/cifar/'
    #trained_params = read_params(snapshot_root + '170413222020/499_params')
    #lasagne.layers.set_all_param_values(network, trained_params)

    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
            #W= trained_params[0], b = trained_params[1])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
            #W= trained_params[2], b = trained_params[3])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
            #W= trained_params[4], b = trained_params[5])
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
            #W= trained_params[6], b = trained_params[7])
    return network
