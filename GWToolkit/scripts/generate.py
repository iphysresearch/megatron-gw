import sys
sys.path.append('..')
import os
from gwtoolkit.gw import WaveformDataset, waveform
from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
import torch
from torchvision import transforms
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
# import itertools
import numpy as np
# save waveform
from pathlib import Path
import h5py
# import tqdm
from time import time, strftime, localtime
from generate_psd import PSD_Sampler


def get_psd_num(psd_dir):
    files = os.listdir(psd_dir)
    psd_list = [name for name in files
                if name.startswith('psd-')]
    return len(psd_list)


def save_hdf5(train_waveform, fdir, h5_fn):
    p = Path(fdir)
    p.mkdir(parents=True, exist_ok=True)
    t_start = time()
    print("saving {} ...".format(h5_fn))
    f_data = h5py.File(p / h5_fn, 'w')
    for i in train_waveform.keys():
        data_name = i
        f_data.create_dataset(
            data_name,
            data=train_waveform[data_name],
            compression='gzip',
            compression_opts=9,
        )

    f_data.close()
    t_end = time()
    print("{} saved, used ".format(h5_fn) + strftime('%M:%S', localtime(t_end - t_start)))


def wfd_init_wraper(sampling_frequency=4096, duration=8, dets=['H1'],
                    conversion='BBH', waveform_approximant='IMRPhenomPv2',
                    reference_frequency=50, minimum_frequency=20.):

    # sampling_frequency = 4096     # [Hz], sampling rate
    # duration = 8                  # [sec], duration of a sample
    # conversion = 'BBH'
    # waveform_approximant = 'IMRPhenomPv2'
    # # waveform_approximant = 'SEOBNRv4P' ## TODO SEOBNRv4P 有报错
    # # waveform_approximant = 'IMRPhenomPv2_NRTidal'
    # # waveform_approximant = 'IMRPhenomXP'
    # reference_frequency = 50.
    # minimum_frequency = 20.
    waveform_arguments = dict(waveform_approximant=waveform_approximant,
                              reference_frequency=reference_frequency,
                              minimum_frequency=minimum_frequency)
    base = 'bilby'
    # dets = ['H1', 'L1'][:1]

    filename = '../tests/gw/demo.prior'   # default prior file

    # waveform dataset
    wfd = WaveformDataset(sampling_frequency=sampling_frequency,
                          duration=duration,
                          conversion=conversion)

    wfd.load_prior_source_detector(
        filename=filename,
        base=base,
        dets=dets,
        waveform_arguments=waveform_arguments)

    return wfd


def wfdt_init_wraper(num, wfd, duration=8):
    # patch_size = 0.5  # [sec]
    # overlap = 0.5     # [%]
    # num = 100         # number of samples in an epoch.
    target_time = 1126259456.3999023 + 6
    buffer_time = 2
    start_time = target_time-(duration - buffer_time)
    geocent_time = (target_time - 1.1, target_time + 1.1)
    SNR = 1          # 信噪比
    # SNR = 0          # 使用 prior 中定义的 distance 来调整信噪比
    target_optimal_snr_tuple = (0, SNR)
    norm_params_kind = 'minmax'
    # norm_data_kind = 'standard'

    stimulated_whiten_ornot = True   # Using stimulated whiten noise or not
    classification_ornot = None
    denoising_ornot = True           # For denoising
    rand_transform_data = None
    ###

    target_labels = ['mass_ratio', 'chirp_mass',                                  # mass dim
                     'luminosity_distance',                                       # SNR related
                     'dec', 'ra', 'theta_jn', 'psi', 'phase',                     # oritation related
                     'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',        # spin related
                     'geocent_time',                                              # arrival time related
                     'optimal_snr', 'matched_filter_snr'
                     ]
    # ######## Hyper parameters end ###############################

    composed_data = transforms.Compose([
        # Patching_data(patch_size=patch_size,
        #               overlap=overlap,
        #               sampling_frequency=sampling_frequency),
        # input:  (batch_size, 2               ,   duration * sampling_rate)
        # output: (batch_size, 2 * num_of_patch, patch_size * sampling_rate)
        ToTensor(),
        Normalize(mean=(0,)*num, std=(45.6,)*num),  # specify ~45.6 for std=1
    ])
    composed_params = None if denoising_ornot else transforms.Compose([
        Normalize_params(norm_params_kind,
                         wfd=wfd, labels=target_labels,
                         feature_range=(-1, 1)),
        ToTensor(),
    ])

    return WaveformDatasetTorch(wfd, num=num,
                                start_time=start_time,
                                geocent_time=geocent_time,
                                target_optimal_snr_tuple=target_optimal_snr_tuple,
                                target_labels=target_labels,
                                stimulated_whiten_ornot=stimulated_whiten_ornot,
                                transform_data=composed_data,
                                transform_params=composed_params,
                                rand_transform_data=rand_transform_data,
                                classification_ornot=classification_ornot,
                                denoising_ornot=denoising_ornot)


def generate_data(wfdt, loader, h5_dir, psd_dir, psd_num, loop_num, type):
    train_waveform = {'clean': np.array([]), 'noisy': np.array([]), 'params': np.array([])}
    for j in range(loop_num):
        for index in range(psd_num):
            wfdt.wfd.dets['H1'].load_asd_or_psd_file(asd_file='{}psd-{}.txt'.format(psd_dir, index))
            wfdt.update()  # dynamic update our prior in loader.
            for data, signal, noise, param in loader:
                train_waveform['clean'] = signal.numpy()
                train_waveform['noisy'] = data.numpy()
                train_waveform['params'] = param.numpy()
            #                                                       #    loop-index_psd-index
            h5_fn = type + "-{}_{}.hdf5".format(int(j+1), int(index+1))  # ex : train-5_10
            save_hdf5(train_waveform, h5_dir, h5_fn)
    print('Done')


def generate_dataset(waveform_num_per_file, h5_dir, read_psd, psd_num, loop_num, psd_dir='./psd/', sampling_rate=4096, type='train'):
    # real_psd = False
    if not read_psd:
        # 1. Generate PSDs
        # psd_num = 10
        psd_dir = './psd/'
        # ________________________________
        psd_sampler = PSD_Sampler(datadir='/workspace/zhaoty/O1_H1_all/', noise_interval=4096,
                                  target_sampling_rate=sampling_rate,
                                  dq_bits=(0, 1, 2, 3), inj_bits=(0, 1, 2, 4))
        # paras: (default)
        #           datadir='/workspace/zhaoty/O1_data/',
        #           random_seed=42, noise_interval=4096, signal_length=8,
        #           target_sampling_rate=4096,
        #           dq_bits=(0, 1, 2, 3),
        #           inj_bits=(0, 1, 2, 4)
        psd_sampler.generate_psds(psd_num)
        psd_sampler.save_psds(psd_dir)

    # 2. Init for WaveformDataset
    wfd = wfd_init_wraper(sampling_frequency=sampling_rate, duration=8, dets=['H1'],
                          conversion='BBH', waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50, minimum_frequency=20.)

    # 3. Init for WaveformDatasetTorch
    # ######## Hyper parameters start ###############################
    # waveform_num_per_file = 100
    wfdt = wfdt_init_wraper(waveform_num_per_file, wfd)

    shuffle = True

    # DataLoader objects
    loader = DataLoader(wfdt, batch_size=waveform_num_per_file, shuffle=shuffle, pin_memory=False,
                        num_workers=0,
                        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32-1)))

    # h5_dir = './valid_data/'
    # loop_num = 1
    generate_data(wfdt, loader, h5_dir, psd_dir, psd_num, loop_num, type)


def main():
    type = 'test'
    # total data num = waveform_per_file * psd_num * loop_num
    if type not in ['train', 'test', 'valid']:
        raise Exception('wrong type', type)

    if type == 'test':
        waveform_num_per_file = 100
        loop_num = 1
        h5_dir = './test/'
        read_psd = True
        # read psd file (txt)
        psd_dir = '../psds/'
        psd_num = get_psd_num(psd_dir)

    elif type == 'train':
        waveform_num_per_file = 10000
        loop_num = 5
        h5_dir = './bigdata/'
        read_psd = False
        psd_num = 10
        psd_dir = '../psds/'

    elif type == 'valid':
        waveform_num_per_file = 100
        loop_num = 1
        h5_dir = './valid/'
        read_psd = True

        psd_dir = '../psds/'
        psd_num = get_psd_num(psd_dir)

    generate_dataset(waveform_num_per_file, h5_dir, read_psd, psd_num, loop_num, psd_dir=psd_dir, type=type)
    return 0


if __name__ == '__main__':
    main()
