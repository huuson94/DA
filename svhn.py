#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import time
from shutil import copy2


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
from modify_data import modify_7
from modify_data import modify_8
from modify_data import modify_9

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


def load_dataset():
    import scipy.io as sio
    train_data = sio.loadmat('train_32x32.mat')

    x_train = (train_data['X'] - 128) / 128.0
    #x_train = np.swapaxes(np.swapaxes(np.swapaxes(x_train, 0, 3), 1, 3), 1, 2)
    x_train = x_train.transpose(3, 2,0,1)
    y_train = train_data['y']

    y_train = y_train.flatten()
    y_train = y_train - 1

    # extra_data = sio.loadmat('train_32x32.mat')
    # x_extra = (extra_data['X'] - 128)/128.0
    # #x_extra = np.swapaxes(np.swapaxes(np.swapaxes(x_extra, 0, 3), 1, 3), 1, 2)
    # x_extra = x_extra.transpose(3, 2,0,1)
    # y_extra = extra_data['y']
    # y_extra = y_extra.flatten()
    # y_extra = y_extra - 1

    # print("shuffle")
    # x_train, y_train = random_sample(x_train, y_train)
    # print("shuffle done")
    x_train, y_train, x_valid, y_valid = x_train[:-4000], y_train[:-4000], x_train[-4000:], y_train[-4000:]


    # print("shuffle")
    # x_extra, y_extra = random_sample(x_extra, y_extra)
    # print("shuffle done")
    # # ratio = 1/3.0
    # s_len = len(y_extra)
    # x_1, y_1, x_2, y_2= x_extra[:-2000], y_extra[:-2000], x_extra[-2000:], y_extra[-2000:]
    # x_train = np.concatenate((x_train, x_1), axis = 0)
    # y_train = np.concatenate((y_train, y_1), axis = 0)
    # x_valid = np.concatenate((x_valid, x_2), axis = 0)
    # y_valid = np.concatenate((y_valid, y_2), axis = 0)

    #print(X.shape)
    #print(x_val.shape)
    #for inx in range(5):
    #   plt.imsave('test'+str(inx)+'.png', np.swapaxes(np.swapaxes(x_val[inx], 0, 1), 1, 2))
        #print(x_val[inx].shape)
    #    print(y_val[inx])
    #exit(0)

    test_data = sio.loadmat('test_32x32.mat')
    x_test = (test_data['X'] - 128) / 128.0
    #x_test = np.swapaxes(np.swapaxes(np.swapaxes(x_test, 0, 3), 1, 3), 1, 2)
    x_test = x_test.transpose(3, 2,0,1)

    y_test = test_data['y']
    y_test = y_test.flatten()

    y_test = y_test -1
    #return x_train[:1500], y_train[:1500], x_valid[:1500], y_valid[:1500], x_test[:1500], y_test[:1500]
    return x_train.astype('float32'), y_train, x_valid.astype('float32'), y_valid, x_test.astype('float32'), y_test

def random_sample(inputs, targets):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs), dtype="int32").tolist()
    np.random.shuffle(indices)
    return np.array([inputs[i] for i in indices], dtype='float32'), np.array([targets[i] for i in indices], dtype='int32')


def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def build_custom_mlp(input_var=None):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)
    network = lasagne.layers.dropout(network, p=0.5)
    nonlin = lasagne.nonlinearities.rectify
    network = lasagne.layers.DenseLayer(
                network, 800, nonlinearity=lasagne.nonlinearities.rectify)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

def build_cnn(input_var=None):
    snapshot_root = 'snapshot_models/svhn/'
    #trained_params = read_params(snapshot_root + '170314184952/999_params')
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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(model='mlp', num_epochs=500, start_epoch=0):
    # Load the dataset
    print("Loading data...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model == 'cus':
        network = build_custom_mlp(input_var)
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return


    time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime())
    snapshot_root = 'snapshot_models/'
    snapshot_name = 'svhn/'+ time_stamp + '/'

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

    learning_rate_init = 1e-4
    learning_rate = theano.shared(np.array(learning_rate_init, dtype=theano.config.floatX))

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=learning_rate)
    #updates = lasagne.updates.momentum(loss, params, learning_rate = learning_rate)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    training_history = {}
    training_history['epoch_training_loss'] = []
    training_history['epoch_validation_loss'] = []
    training_history['epoch_validation_accuracy'] = []
    training_history['training_loss'] = []
    training_history['validation_loss'] = []
    training_history['epoch_learning_rate'] = []

    filename = snapshot_root + snapshot_name + 'rs.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    rs_file = open(filename, "a")


    sys.stdout = Tee(sys.stdout, rs_file)
    copy2('svhn.py', snapshot_root + snapshot_name)
    copy2('modify_data.py', snapshot_root + snapshot_name)

    # We iterate over epochs:
    iter_now = 0
    x_train_ori = x_train
    batch_size = 500
    y_train_by_class = []
    x_train_ori = x_train
    y_train_ori = y_train
    x_train_pre, y_train_pre = x_train, y_train
    #import matplotlib.pyplot as plt
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()
        if((epoch + 1) % 2 == 0):
            x_train = x_train_ori
            y_train = y_train_ori
        else:
            x_train, y_train  = modify_9(x_train_ori, x_train_pre, y_train_ori, y_train_pre)
            x_train_pre, y_train_pre = x_train, y_train
        print("Modify of {} took {:.3f}s".format(
            epoch + 1, time.time() - start_time))
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)

            train_batches += 1
            iter_now += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
	    val_batches += 1

	# Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        #plot loss
        training_history['epoch_training_loss'].append(train_err/train_batches)
        training_history['epoch_validation_loss'].append(val_err / val_batches)
        training_history['epoch_validation_accuracy'].append(val_acc/val_batches * 100)
        training_history['epoch_learning_rate'].append(learning_rate.get_value())
        #if(True):
        if(epoch != 0 and ((epoch + 1) % 50) == 0):
            snapshot_path_string = snapshot_root+snapshot_name+str(epoch)+"_"
            print("Save param: " + snapshot_path_string+"params")
            save_params(params, snapshot_path_string+"params")
            print("Creating snapshot: " + snapshot_path_string+'loss.png')
            plot_loss(training_history, snapshot_path_string)
            print("Creating weight visualize: " + snapshot_path_string + 'conv1weights.png')
            plot_conv1_weights(lasagne.layers.get_all_layers(network)[1], snapshot_path_string + '_conv1weights.png')
            print("Save training history")
            save_history(training_history, snapshot_path_string+"history")


    print("Trainning done")

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test, 500, shuffle=False):
	inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    save_params(params, snapshot_root + snapshot_name + 'last_params_' + sys.argv[2] + '_'+ time_stamp )


    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


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
