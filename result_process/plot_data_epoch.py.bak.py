import os
import pickle
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as


#plt.figure(figsize=(70, 70))

root_path = '../snapshot_models/'
linestyles = ['-', '--', ':', '.']
#markers = ['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
markers = []
markers = ['o', 'x', '<', '>', '.', '_', 'v', ',']
styles = markers + [
    #r'$\lambda$',
    #r'$\bowtie$',
    #r'$\circlearrowleft$',
    #r'$\clubsuit$',
    #r'$\checkmark$'
    ]

#cifar_res
#colors = ('r', 'b', 'g', 'c', 'm', 'y', 'k')
#cifar l5
colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')

def compare(history_path, no, label, fig, sub_plot):
    #plt.ylim(20,80)# !
    data_file = open(history_path, 'rb')
    data = pickle.load(data_file)
    t = np.arange(0, 500.0, 1)
    line1, = sub_plot.plot(data['epoch_validation_accuracy'],
                        marker = styles[(int(no) - len(linestyles)) % len(styles)],
                        markersize=3,
                        #markevery=len(data['epoch_validation_accuracy'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='vc: '+ label,
                        color = colors[int(no) % len(colors)])
    # sub_plot.set_ylabel('val acc', color='b')
    #sub_plot.plot(t, data['epoch_validation_accuracy'], 'r.')
    sub_plot.set_xlabel('epoch')
    sub_plot.set_ylabel('val acc')
    sub_plot.tick_params('y')

    # box = sub_plot.get_position()
    # sub_plot.set_position([box.x0, box.y0 + box.height * 0.1,
    #              box.width, box.height * 0.9])
    # sub_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
    # first_legend = plt.legend(handles=[line1], loc=1)
    # ax = plt.gca().add_artist(first_legend)

    sub_plot2 = sub_plot.twinx()
    sub_plot2.set_ylim([0, 0.1])
    #s2 = np.sin(2 * np.pi * t)
    # s2 = np.sin(2 * np.pi * t)
    # print(s2.shape)
    #print(data['epoch_training_loss'].shape)
    line2, = sub_plot2.plot(data['epoch_training_loss'], marker = styles[(int(no) - len(linestyles)) % len(styles)],
                        markersize=1,
                        #markevery=len(data['training_loss'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='tc: '+label,
                        color = colors[int(no) % len(colors)])
    sub_plot2.set_ylabel('train loss', color='r')
    sub_plot2.tick_params('y', colors='r')
    plt.legend(loc='best')
    # plt.legend(handles=[line2], loc=4)
    # plt.legend(bbox_to_anchor=(2.1, 2.05))
    #

    fig.tight_layout()
    return plt

def compare1(history_path, no, label, fig):
    data_file = open(history_path, 'rb')
    data = pickle.load(data_file)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sub_plot.plot(data['epoch_validation_accuracy'],
                        marker = styles[(int(no) - len(linestyles)) % len(styles)],
                        markersize=3,
                        #markevery=len(data['epoch_validation_accuracy'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='vc: '+ label,
                        color = colors[int(no) % len(colors)])
    ax2 = ax.twinx()
    ax2.plot(data['epoch_training_loss'], marker = styles[(int(no) - len(linestyles)) % len(styles)],
                        markersize=1,
                        #markevery=len(data['training_loss'])/1000.,
                        linestyle = linestyles[int(no) % len(linestyles)], linewidth=0.5,
                        label='tc: '+label,
                        color = colors[int(no) % len(colors)])
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("epoch")
    ax.set_ylabel("val acc")
    ax2.set_ylabel("train loss")
    ax2.set_ylim(0, 35)
    ax.set_ylim(-20,100)

    return plt

def main(model):
    lines = [line.rstrip('\n') for line in open(model+'_history_path.txt')]
    save_name = model+"/" +save_name+ ".eps"
    fig, sub_plot = plt.subplots()
    # plt.ylim(99,99.7)# !
    for line in lines:
        if(line.startswith('#')):
            continue
        folder, file_name, no, label = line.split("\t")
        save_name = label + '_' + save_name
        compare1(root_path + model + "/" + folder + '/' + file_name, no, label, fig, sub_plot)
    #plt.xlabel('epoch')
    #plt.legend(loc='best')
    # z = np.random.randint(0,3, size=10)
    # red_dot, = plt.plot(z, "ro", markersize=15)
    # # Put a white cross over some of the data.
    # white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)
    # plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])

    save_name = model+"/" +save_name+ ".eps"
    plt.savefig('epoch/val_acc/' + save_name) # !
    #print('test')

main(sys.argv[1])
