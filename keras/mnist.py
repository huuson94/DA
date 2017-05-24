'''trains a simple convnet on the mnist dataset.

gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a grid k520 gpu.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import time
import os
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from build_resnet18 import ResnetBuilder
from build_lnet5 import build_lnet5
import pickle
import numpy as np
from shutil import copy2
from modify_data import modify_8
from modify_data import modify
from modify_data import modify_9
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



K.set_image_data_format('channels_first')

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc= []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


batch_size = 128
nb_classes = 10
nb_epoch = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.reshape(x_train, (x_train.shape[0], 1, 28, 28))
x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28))

# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)


x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

from keras.models import load_model
model = load_model('snapshot_models/mnist/170525004326/model.h5')

# model = ResnetBuilder.build_resnet_18((1,28,28), 10)
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

# path = 'snapshot_models/mnist/170518145434/'
# history = open(path + 'history.pkl','r')
# data = pickle.load(history)
# print(data)
# model.load_weights(path + 'params.h5')
# score = model.evaluate(x_test, np_utils.to_categorical(y_test, 10), verbose=0)
# print('test score:', score[0])
# print('test accuracy:', score[1])
# exit(0)


history = LossHistory()

x_train_ori = x_train
y_train_ori = y_train
x_train_pre, y_train_pre = x_train, y_train
for epoch in xrange(0, 50):
    print("Epoch: " + `epoch`)
    start_time = time.time()
    if((epoch + 1) % 2 == 0):
        x_train = x_train_ori
        #y_train = y_train_ori
    else:
        x_train = modify(x_train_ori, epoch)
        #x_train_pre, y_train_pre = x_train, y_train
    print("Modify took {:.3f}s".format(
            time.time() - start_time))
    model.fit(x_train, np_utils.to_categorical(y_train, 10), batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_val,np_utils.to_categorical(y_val, 10)), callbacks=[history])

score = model.evaluate(x_test, np_utils.to_categorical(y_test, 10), verbose=0)
print('test score:', score[0])
print('test accuracy:', score[1])


#save history
time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime())
snapshot_root = 'snapshot_models/'
snapshot_name = 'snapshot_models/' + 'mnist'+ '/'+ time_stamp + '/'
history_file_name = snapshot_name+"history.pkl"

if not os.path.exists(os.path.dirname(history_file_name)):
    try:
        os.makedirs(os.path.dirname(history_file_name))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

history_file = open(history_file_name, 'wb')
pickle.dump(np.array([history.train_losses, history.val_losses, history.train_acc, history.val_acc]), history_file)
model.save(snapshot_name+'model.h5')
copy2('mnist.py', snapshot_name)

# summarize history for accuracy
plt.plot(history.train_acc)
plt.plot(history.val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(snapshot_name+'accuracy.png')
# summarize history for loss
plt.figure()
plt.plot(history.train_losses)
plt.plot(history.val_losses)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(snapshot_name+'loss.png')
