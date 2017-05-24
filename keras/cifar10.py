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
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

from build_resnet18 import ResnetBuilder
from build_lnet5 import build_lnet5
import pickle
import numpy as np
from shutil import copy2
from modify_data import modify_8
from modify_data import modify_9
from modify_data import modify_5
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

time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime())
snapshot_root = 'snapshot_models/'
snapshot_name = 'snapshot_models/' + 'cifar10'+ '/'+ time_stamp + '/'
history_file_name = snapshot_name+"history.pkl"

if not os.path.exists(os.path.dirname(history_file_name)):
    try:
        os.makedirs(os.path.dirname(history_file_name))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

history_file = open(history_file_name, 'wb')
copy2('cifar10.py', snapshot_name)
copy2('modify_data.py', snapshot_name)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger(snapshot_name+'resnet18_cifar10.csv')

batch_size = 32
nb_classes = 10
nb_epoch = 80

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis=0)
x_train /= 128.
x_test /= 128.
x_train = np.reshape(x_train, (x_train.shape[0], 3, 32, 32))
x_test = np.reshape(x_test, (x_test.shape[0], 3, 32, 32))

y_train = y_train.flatten()
y_test = y_test.flatten()

# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)


x_train, x_val = x_train[:-5000], x_train[-5000:]
y_train, y_val = y_train[:-5000], y_train[-5000:]

model = ResnetBuilder.build_resnet_18((3, 32, 32), 10)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# path = 'snapshot_models/cifar10/170522003935/'
# history = open(path + 'history.pkl','r')
# data = pickle.load(history)
# print(data)
# from keras.models import load_model
# model = load_model(path + 'model.h5')
# score = model.evaluate(x_test, np_utils.to_categorical(y_test, 10), verbose=0)
# print('test score:', score[0])
# print('test accuracy:', score[1])
# exit(0)


history = LossHistory()

x_train_ori = x_train
y_train_ori = y_train
x_train_pre, y_train_pre = x_train, y_train
for epoch in xrange(0, nb_epoch):
    print("Epoch: " + `epoch`)
    start_time = time.time()
    if((epoch + 1) % 2 == 0):
        x_train = x_train_ori
        y_train = y_train_ori
    else:
        x_train, y_train  = modify_9(x_train_ori, x_train_pre, y_train_ori, y_train_pre)
        x_train_pre, y_train_pre = x_train, y_train
    print("Modify took {:.3f}s".format(
    time.time() - start_time))
    #model.fit(x_train, np_utils.to_categorical(y_train, 10), batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_val,np_utils.to_categorical(y_val, 10)), callbacks=[history])
    model.fit(x_train, np_utils.to_categorical(y_train, 10),
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_val, np_utils.to_categorical(y_val, 10)),
              shuffle=True,
              callbacks=[history, lr_reducer, early_stopper, csv_logger])
score = model.evaluate(x_test, np_utils.to_categorical(y_test, 10), verbose=0)
print('test score:', score[0])
print('test accuracy:', score[1])


#save history
pickle.dump(np.array([history.train_losses, history.val_losses, history.train_acc, history.val_acc]), history_file)
model.save(snapshot_name+'model.h5')


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
