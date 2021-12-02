# coding=utf-8

import numpy as np
import os
import matplotlib.pyplot as plt

path = '../../vis/'
files = os.listdir(path)
sampling_frequency = 4096     # [Hz], sampling rate
duration = 8                  # [sec], duration of a sample

patch_size = 0.5  # [sec]
overlap = 0.5     # [%]
patch_size_sec = int(patch_size * sampling_frequency)

for file in files:
    if file == 'data-0.npy':
        npy_all = np.load(os.path.join(path, file))
        denoised = npy_all[:int(npy_all.shape[0]/3), :, :]
        noisy = npy_all[int(npy_all.shape[0]/3):int(2*npy_all.shape[0]/3), :, :]
        clean = npy_all[int(2*npy_all.shape[0]/3):, :, :]

        for index in range(clean.shape[0]):
            #plt.subplot(2,2,index+1)
            for i in range(clean.shape[1]):
                if i>13 and i%2==0:    #i % 2 == 0:
                    plt.subplot(3, 3, int(i/2)-6)
                    plt.plot(np.linspace(0 + patch_size * overlap * i, patch_size * (overlap * i + 1),
                                            num=patch_size_sec),
                             noisy[index][i], color='b')
                    plt.plot(np.linspace(0 + patch_size * overlap * i, patch_size * (overlap * i + 1),
                                            num=patch_size_sec),
                             denoised[index][i], color='r')
                    plt.plot(np.linspace(0 + patch_size * overlap * i, patch_size * (overlap * i + 1),
                                            num=patch_size_sec),
                             clean[index][i], color='g')    #, alpha=0.5)

                    #plt.xlabel('time [sec]')
                    #plt.legend(['noisy signal','denoised', 'clean'])
            plt.show()
            plt.close()
    # plt.plot(data.numpy()[0,23])
    # plt.plot(signal.numpy()[0,23])

print('done.')