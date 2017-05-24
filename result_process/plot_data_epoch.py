import os
import pickle
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as


#plt.figure(figsize=(70, 70))

root_path = '../keras/snapshot_models/'
linestyles = ['-', '--', ':', 'dashed']
#markers = ['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
markers = []
markers_vc = ['o', 'x', '<', '>', '.', '_', 'v', ',']
markers_tl = ['o', 'x', '<', '>', '.', '_', 'v', ',']
styles_vc = markers_vc + [
    #r'$\lambda$',
    #r'$\bowtie$',
    #r'$\circlearrowleft$',
    #r'$\clubsuit$',
    #r'$\checkmark$'
    ]
styles_tl = markers_tl + [
    #r'$\lambda$',
    #r'$\bowtie$',
    #r'$\circlearrowleft$',
    #r'$\clubsuit$',
    #r'$\checkmark$'j
    ]
#cifar_res
#colors = ('r', 'b', 'g', 'c', 'm', 'y', 'k')
#cifar l5
colors_vc = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
colors_tl = ( 'm', 'y', 'k', 'r', 'g', 'b', 'c')

def compare1_val_acc(history_path, no, label, sub_plot):
    data_file = open(history_path, 'rb')
    data = pickle.load(data_file)
    ln, = sub_plot.plot(data[3] * 100,
                        marker = styles_vc[(int(no) - len(linestyles)) % len(styles_vc)],
                        markersize=3,
                        #markevery=len(data['epoch_validation_accuracy'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='vc: '+ label,
                        color = colors_vc[int(no) % len(colors_vc)])

    return ln

def compare1_train_loss(history_path, no, label, sub_plot):
    data_file = open(history_path, 'rb')
    data = pickle.load(data_file)
    ln, = sub_plot.plot(data[0],
                        marker = styles_tl[(int(no) - len(linestyles)) % len(styles_tl)],
                        markersize=3,
                        #markevery=len(data['epoch_validation_accuracy'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='tl: '+ label,
                        color = colors_tl[int(no) % len(colors_tl)])

    return ln

def main(model):
    lines = [line.rstrip('\n') for line in open(model+'_history_path.txt')]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    save_name = ''
    lns = []
    for line in lines:
        if(line.startswith('#')):
            continue
        folder, file_name, no, label, _ = line.split("\t")
        save_name = label + '_' + save_name
        lns1 = compare1_val_acc(root_path + model + "/" + folder + '/' + file_name, no, label, ax)
        lns.append(lns1)


    ax2 = ax.twinx()
    for line in lines:
        if(line.startswith('#')):
            continue
        folder, file_name, no, label, _ = line.split("\t")
        save_name = label + '_' + save_name
        lns2 = compare1_train_loss(root_path + model + "/" + folder + '/' + file_name, no, label, ax2)
        lns.append(lns2)

    ax.yaxis.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("val acc")
    ax2.set_ylabel("train loss")

    labs = [l.get_label() for l in lns]
    #ax.legend(lns, labs, loc=2)
    ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


    ax.set_ylim(98,99.6)
    ax2.set_ylim(0, 1)
    save_name = model+"/" +save_name+ ".eps"
    plt.savefig('epoch/vc_tl/' + save_name, format='eps', dpi=900) # !
    #print('test')

main(sys.argv[1])
