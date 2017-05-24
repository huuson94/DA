import os
import pickle
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#plt.figure(figsize=(70, 70))

root_path = '../snapshot_models/'
def compare(history_path, no):
    data_file = open(history_path, 'rb')
    data = pickle.load(data_file)
    min_index = int(min(len(data['modify_time']), len(data['train_time'])))
    time_axis = np.asarray(data['modify_time'][:min_index], dtype='float32') + np.asarray(data['train_time'][:min_index], dtype='float32')
    print(np.max(time_axis))
    plt.plot(time_axis, data['train_loss'],linewidth=0.4, label=no)
    return plt

def main(model):
    lines = [line.rstrip('\n') for line in open(model+'_rs_path.txt')]
    for line in lines:
        folder, rs_txt_name, rs_pkl_name, no = line.split("\t")
        compare(root_path + model + '/' + folder + '/' + rs_pkl_name, no)
    plt.xlabel('time')
    plt.ylabel('train loss')
    ymax = 3
    plt.ylim(0,ymax) # !
    plt.legend(loc='best')
    save_name = model+".eps"
    plt.savefig('time/train_loss/'+save_name, format='eps', dpi=900) # !

main(sys.argv[1])
