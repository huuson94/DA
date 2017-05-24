import os
import pickle
import sys
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#plt.figure(figsize=(70, 70))

root_path = '../snapshot_models/svhn/'


def parse_rs_file(folder_name, file_name):
    lines = [line.rstrip('\n') for line in open(root_path + folder_name + '/' + file_name)]
    rs = {}
    rs['modify_time'] = [0.0]
    rs['train_time'] = [0.0]
    rs['val_acc'] = []
    rs['train_loss'] = []
    rs['val_loss'] = []
    for line_no, line in enumerate(lines):
        if(line.startswith("Modify")):
            m = re.search(r"\d\.\d+",line)
            rs['modify_time'].append(rs['modify_time'][-1] + float(m.group()))
        if(line.startswith("Epoch")):
            m = re.search(r"\d\.\d+",line)
            rs['train_time'].append(rs['train_time'][-1] + float(m.group()))
        if(line.endswith("%") and line.startswith("  validation")):
            m = re.search(r"\d{1,2}\.\d+",line)
            rs['val_acc'].append(float(m.group()))
        if(not line.endswith("%") and line.startswith("  validation")):
            m = re.search(r"\d{1,2}\.\d+",line)
            rs['val_loss'].append(float(m.group()))
        if(line.startswith("  training loss")):
            m = re.search(r"\d{1,2}\.\d+",line)
            rs['train_loss'].append(float(m.group()))

    del(rs['modify_time'][0])
    del(rs['train_time'][0])
    rs_file = open(root_path + folder_name + '/' + 'rs.pkl', 'wb')
    print(root_path + folder_name)
    pickle.dump(rs, rs_file)

def main():
    lines = [line.rstrip('\n') for line in open('svhn_rs_path.txt')]
    for line in lines:
        folder, rs_txt_name, rs_pkl_name, no = line.split("\t")
        parse_rs_file(folder, rs_txt_name)
main()
