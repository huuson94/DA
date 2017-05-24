import pickle
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def view_history(path, view_range):
    data_file = open(path, 'rb')
    data = pickle.load(data_file)
    for index in range(1, view_range):
        print("validate acc "+ str(index)+ ':' + str(data['epoch_validation_accuracy'][-index]))


def compare(path1, path2):
	data_file1 = open(path1, 'rb')
	data1 = pickle.load(data_file1)
	data_file2 = open(path2, 'rb')
	data2 = pickle.load(data_file2)
	plt.figure() # !
	plt.plot(data1['epoch_validation_accuracy'], label='val acc 1 (epoch)')
	plt.plot(data2['epoch_validation_accuracy'], label='val acc 2 (epoch)')
	plt.xlabel('epoch')
	plt.ylabel('acc')
	ymax = 110
	plt.ylim(0,ymax) # !
	plt.legend(loc='best')
	save_name = "compare.png"
	plt.savefig(save_name) # !

compare(str(sys.argv[1]), str(sys.argv[2]))