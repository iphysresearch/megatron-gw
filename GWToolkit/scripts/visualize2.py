# coding=utf-8
import sys
sys.path.append('..')

import numpy as np
import os
import matplotlib.pyplot as plt
from scripts.utils import Data_utils


def load_file(path1,file1):
    npy_all1 = np.load(os.path.join(path1, file1))
    denoised1 = npy_all1[:int(npy_all1.shape[0]/3), :, :]   
    noisy1 = npy_all1[int(npy_all1.shape[0]/3):int(2*npy_all1.shape[0]/3), :, :]
    clean1 = npy_all1[int(2*npy_all1.shape[0]/3):, :, :]
    return denoised1,noisy1,clean1

def load_dir(path,type = 'data'):
    files = os.listdir(path)
    if type == 'data':
        file_list = [name for name in files 
                if name.startswith(type)]
        print('file num = {}'.format(len(file_list)))
        all_d = None
        all_n = None
        all_c = None
        for i in file_list:
            d,n,c = load_file(path,i)
        
            all_d = np.append(all_d,d,axis=0) if all_d is not None else d
            all_n = np.append(all_n,n,axis=0) if all_n is not None else n
            all_c = np.append(all_c,c,axis=0) if all_c is not None else c
        return all_d,all_n,all_c
    
    if type == 'par':
        file_list = [name for name in files 
                if name.startswith(type)]
        all_par = None
        for i in file_list:
            par = np.load(os.path.join(path,i))
            all_par = np.append(all_par,par,axis=0) if all_par is not None else par
        return all_par

            

    
    
def reshape_waveform(data):
    data_all = None
    idx2 = int(data.shape[2]/2)
    for j in range(data.shape[0]):
        data_tmp = None
        for i in range(data.shape[1]):
            if i == 0:
                data_tmp = data[j][0]
            if i>0 :  
                data_tmp = np.append(data_tmp,data[j][i][idx2:])
        data_all = np.vstack([data_all,data_tmp]) if data_all is not None else data_tmp
    return data_all
    

def main():

    path1 = 'data/vis-4/'

    sampling_frequency = 4096     # [Hz], sampling rate
    duration = 8                  # [sec], duration of a sample

    patch_size = 0.5  # [sec]
    overlap = 0.5     # [%]
    patch_size_sec = int(patch_size * sampling_frequency)
    #___________________
    time_range = [0,8]
    plot_range=[4,6.6]
    #___________________
    d,n,c = load_dir(path1)
    d = reshape_waveform(d)
    n = reshape_waveform(n)
    c = reshape_waveform(c)
    p = load_dir(path1,type='par')
    #print(d.shape,p.shape)

    utils = Data_utils(sampling_rate=sampling_frequency,time_duration=time_range[1]-time_range[0])

    range_index = [int(sampling_frequency*time_range[0]),int(sampling_frequency*time_range[1])] 

    Overlaps = []
    for i in range(d.shape[0]):
        Overlap1 = utils.get_overlap(d[i][range_index[0]:range_index[1]],c[i][range_index[0]:range_index[1]])
        Overlaps.append(Overlap1)

    for i in range(len(Overlaps)):
        if Overlaps[i] < 0.95:
            print('index: {}'.format(i))
            print('Overlap: {}'.format(Overlaps[i]))

    #print(Overlaps)
    #plt.hist(Overlaps,bins='auto')
    #plt.show()

    #print('overlap1 = {}'.format(Overlap1))


    return 0

if __name__ == '__main__':
    main()

