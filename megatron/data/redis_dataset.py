# coding=utf-8
# Copyright (c) 2022, PCL.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gravitational waveform dataset."""

import numpy as np
import torch
import h5py
import os
# from megatron import (
#     mpu,
#     print_rank_0
# )
import sys
import redis
sys.path.append('GWToolkit/')
from gwtoolkit.redis import DatasetTorchRedis
import msgpack_numpy as m
m.patch() 

from gwpy.signal import filter_design
from gwtoolkit.torch import Patching_data, Normalize_strain
from gwtoolkit.gw.readligo import FileList, getstrain_cvmfs
from gwpy.timeseries import TimeSeries
from numpy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import welch

class RedisDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, host='192.168.202.149', port=1234, seed=1234):

        connection_pool = redis.ConnectionPool(host=host, port=port,
                                               db=0, decode_responses=False)
        self.r = redis.Redis(connection_pool=connection_pool)
        self.data_keys = sorted(self.r.keys('data_*'))
        self.signal_keys = sorted(self.r.keys('signal_*'))
        self.params_keys = sorted(self.r.keys('params_*'))
        self.mask_keys = sorted(self.r.keys('mask_*'))
        self.seed_data_keys = sorted(self.r.keys('seed_data_*'))
        self.seed_signal_keys = sorted(self.r.keys('seed_signal_*'))
        self.seed_params_keys = sorted(self.r.keys('seed_params_*'))
        self.seed_mask_keys = sorted(self.r.keys('seed_mask_*'))

        # Set for patching tokens
        self.sampling_frequency = 16384
        self.duration = 8
        patch_size = 0.125  # [sec]
        overlap = 0.5     # [%]
        self.patching = Patching_data(
            patch_size=patch_size,
            overlap=overlap,
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            verbose=False,
        )
        self.token_shape = self.patching.output_shape
        self.patching_params = dict(patch_size=patch_size, overlap=overlap)

        num_token, num_length = self.token_shape[-2:]
        mid_index = int(num_length*overlap)
        self.rebuild_forer = lambda tokened_data: np.concatenate([d[:mid_index] for d in tokened_data[:-1]] + [tokened_data[-1]])
        self.rebuild_backer = lambda tokened_data: np.concatenate([tokened_data[0]] + [d[mid_index:] for d in tokened_data[1:]])

        # Params to store.
        self.name = name
        self.seed = seed

    def __len__(self):
        assert len(self.data_keys) == len(self.signal_keys)
        #return len(self.data_keys)*32
        return len(self.data_keys)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # idx_i = idx // 32
        # idx_j = idx % 32
        # noisy_input = self.fromRedis(self.data_keys[idx_i])[idx_j]
        # clean_input = self.fromRedis(self.signal_keys[idx_i])[idx_j]
        # param_np = np.expand_dims(np.real(self.fromRedis(self.params_keys[idx_i])[idx_j]), 0)

        # print("data:{}".format(self.data_keys[idx]))
        # print("signal:{}".format(self.signal_keys[idx]))
        # print("param:{}".format(self.params_keys[idx]))

        # deep copy, in case of "The given NumPy array is not writeable"
        if not self.fromRedis(self.seed_data_keys[idx]) == self.fromRedis(self.seed_signal_keys[idx]) == self.fromRedis(self.seed_params_keys[idx]) == self.fromRedis(self.seed_mask_keys[idx]):
            idx = (idx + self.seed) % len(self.data_keys)
            # print('============== ray is flushing dataset...........======================')

        noisy_input = np.copy(self.fromRedis(self.data_keys[idx]))
        noisy_input = noisy_input.squeeze(0)
        clean_input = np.copy(self.fromRedis(self.signal_keys[idx]))
        clean_input = clean_input.squeeze(0)
        param_np = np.copy(self.fromRedis(self.params_keys[idx]))
        param_np = np.real(param_np)
        mask_np = np.copy(self.fromRedis(self.mask_keys[idx]))[0]  # (2,)
        mask = np.zeros(self.duration * self.sampling_frequency)
        mask[mask_np[0]: mask_np[1]] = 1.0
        mask_input = (self.patching(mask[np.newaxis, ...])+0.2)/1.2
        mask_input = mask_input.squeeze(0)
        train_sample = {
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'mask': mask_input,
            'params': param_np
            }

        return train_sample

    def fromRedis(self, name):
        """Retrieve Numpy array from Redis key 'n'"""
        # Retrieve and unpack the data
        try:
            return m.unpackb(self.r.get(name))
        except TypeError:
            print('No this value')

    def metric(self, model_output, target_signal, mask, rebuild_on_forer=True):
        assert model_output.shape == target_signal.shape == mask.shape == (1, 127, 2048)
        # (1, 127, 2048) => (131072,)
        model_output = self.rebuild_forer(model_output[0]) if rebuild_on_forer else self.rebuild_backer(model_output[0])
        target_signal = self.rebuild_forer(target_signal[0]) if rebuild_on_forer else self.rebuild_backer(target_signal[0])
        mask = self.rebuild_forer(mask[0]) if rebuild_on_forer else self.rebuild_backer(mask[0])
        # (131072,) => (2,)
        return (self.calc_matches(target_signal, model_output),
                self.calc_matches(mask * target_signal, mask * model_output))

    def calc_matches(self, d1, d2):
        fft1 = np.fft.fft(d1)
        fft2 = np.fft.fft(d2)
        norm1 = np.mean(np.abs(fft1)**2)
        norm2 = np.mean(np.abs(fft2)**2)
        inner = np.mean(fft1.conj()*fft2).real
        return inner / np.sqrt(norm1 * norm2)


class DatasetTorchRealEvent(torch.utils.data.Dataset):
    """Waveform dataset using Redis

    Model_Pridict_Signal <= (1, 127, 2048) 

    Usage:
        >>> from gwtoolkit.redis.dataset import DatasetTorchRealEvent

        >>> valid_dataset = DatasetTorchRealEvent()
        >>> len(valid_dataset)
        input data"s shape: (1, 127, 2048)
        1
        >>> valid_dataset[0].shape
        (127, 2048)
        >>> match_long, match_short = valid_dataset.metric(Model_Pridict_Signal)
    """

    def __init__(self, ):
        self.sampling_frequency = 4096*4     # [Hz], sampling rate
        self.duration_long = 32            # [sec], duration of a sample
        self.duration = 8                  # [sec], duration of a sample
        self.target_time = 1126259462.425  # GWTC1_events['GW150914']['trigger-time']  # TODO
        buffer_time = self.duration_long / 2 - self.duration / 4
        self.start_time = self.target_time-(self.duration_long - buffer_time)
        self.geocent_time = (self.target_time-5.1, self.target_time+1.1)

        self.data_dir = '/workspace/zhaoty/dataset/O1_H1_All/'
        self.addr_asds = [
            # https://dcc.ligo.org/T1800044-v5/public
            '/workspace/zhaoty/GWToolkit/gwtoolkit/gw/prior_files/aLIGODesign.txt',
            # https://dcc.ligo.org/LIGO-T2000012/public
            '/opt/conda/lib/python3.8/site-packages/bilby/gw/detector/noise_curves/aLIGO_ZERO_DET_high_P_asd.txt',
            '/opt/conda/lib/python3.8/site-packages/bilby/gw/detector/noise_curves/aLIGO_early_asd.txt',
            '/opt/conda/lib/python3.8/site-packages/bilby/gw/detector/noise_curves/aLIGO_early_high_asd.txt',
            '/opt/conda/lib/python3.8/site-packages/bilby/gw/detector/noise_curves/aLIGO_mid_asd.txt',
            '/opt/conda/lib/python3.8/site-packages/bilby/gw/detector/noise_curves/aLIGO_late_asd.txt',
        ]
        self.use_which_design_ASD_for_whiten = None  # None: Use on-source for whiten

        # Set for patching tokens
        patch_size = 0.125  # [sec]
        overlap = 0.5     # [%]
        self.patching = Patching_data(
            patch_size=patch_size,
            overlap=overlap,
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            verbose=False,
        )
        self.token_shape = self.patching.output_shape
        self.patching_params = dict(patch_size=patch_size, overlap=overlap)

        num_token, num_length = self.token_shape[-2:]
        mid_index = int(num_length*overlap)
        self.rebuild_forer = lambda tokened_data: np.concatenate([d[:mid_index] for d in tokened_data[:-1]] + [tokened_data[-1]])
        self.rebuild_backer = lambda tokened_data: np.concatenate([tokened_data[0]] + [d[mid_index:] for d in tokened_data[1:]])

        # Set for standardizing strains
        feature_range = (-1, 1)
        self.normfunc = Normalize_strain(feature_range)

        self.strain_valid = None
        self.time_valid = None
        self.denoised_time_valid = None
        self.denoised_strain_valid = None
        self.from_real_event()

        # (32 * 16384) => (1, 127, 2048)
        self.input_strain, self.dMin, self.dMax, self.target_signal = self.strain_preprocessing(self.strain_valid)

        # Mask
        t = np.arange(self.start_time+self.duration_long//2-self.duration//2,
                      self.start_time+self.duration_long//2-self.duration//2+self.duration, 1/self.sampling_frequency)
        left = self.target_time - 0.4
        right = self.target_time + 0.1
        self.mask = self.patching((((t > left) & (t < right)) * 1.0)[np.newaxis, ...])

    def __len__(self):
        # print(f'input data"s shape: {self.input_strain.shape}')
        # return len(self.input_strain)
        return 128

    def __getitem__(self, idx):

        noisy_input = self.input_strain[0]
        clean_input = self.target_signal[0]
        mask_input = (self.mask[0] + 0.2)/1.2
        param_np = np.ones((1, 19), dtype=noisy_input.dtype)
        train_sample = {
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'mask': mask_input,
            'params': param_np}

        return train_sample

    def from_real_event(self):
        print('Fetching real event data...')
        # Return whiten and denoised strain for GW150914

        bp = filter_design.bandpass(50, 250, self.sampling_frequency)
        notches = [filter_design.notch(line, self.sampling_frequency) for
                   line in (60, 120, 180)]
        self.zpk = filter_design.concatenate_zpks(bp, *notches)

        filelist = FileList(directory=self.data_dir)
        self.strain_valid, self.time_valid = getstrain_cvmfs(self.start_time, self.start_time + self.duration_long, 'H1', filelist, inj_dq_cache=0)
        hdata = TimeSeries(self.strain_valid, times=self.time_valid, channel='H1')
        hfilt = hdata.whiten(4, 2).filter(self.zpk, filtfilt=True)
        self.denoised_time_valid = hfilt.times.value
        self.denoised_strain_valid = hfilt.value

    def strain_preprocessing(self, strain):
        # TODO need to support multi-detectors
        if self.use_which_design_ASD_for_whiten is not None:
            # # designed PSDs
            data = np.loadtxt(self.addr_asds[2])
            ASDf, ASD = data[:, 0], data[:, 1]
        else:
            # # current on-source PSDs
            ASDf, ASD = None, None

        # [1, self.num_duration_long]
        strain_whitened = self.whiten(strain, self.sampling_frequency, ASDf, ASD)
        data, dMin, dMax = self.normfunc.transform_data(self.patching(self.cut_from_long(strain_whitened)[np.newaxis, ...]))

        signal_data = TimeSeries(strain_whitened, times=self.time_valid, channel='H1').filter(self.zpk, filtfilt=True)
        signal = self.normfunc.transform_signalornoise(self.patching(self.cut_from_long(signal_data)[np.newaxis, ...]), dMin, dMax)
        return data, dMin, dMax, signal

    def signal_postprocessing(self, signal):
        return self.normfunc.inverse_transform_signalornoise(signal, self.dMin, self.dMax)

    def metric(self, output_whiten_signal, rebuild_on_forer=True):
        # (1, 127, 2048) => (131072,) => (2,)
        # output_whiten_signal = self.signal_postprocessing(output_whiten_signal)
        output_whiten_signal = self.rebuild_forer(output_whiten_signal[0]) if rebuild_on_forer else self.rebuild_backer(output_whiten_signal[0])
        return (self.calc_matches(self.rebuild_forer(self.target_signal[0]), output_whiten_signal),
                self.calc_matches(self.cut_for_target(self.rebuild_forer(self.target_signal[0])), self.cut_for_target(output_whiten_signal)))

    def cut_from_long(self, data):
        left_index = int((self.duration_long - self.duration)/2*self.sampling_frequency)
        right_index = int((self.duration_long + self.duration)/2*self.sampling_frequency)
        return data[left_index: right_index]

    def cut_for_target(self, data):
        time = self.cut_from_long(self.time_valid)
        if len(data) == self.duration_long * self.sampling_frequency:
            data = self.cut_from_long(data)
        elif len(data) == self.duration * self.sampling_frequency:
            pass
        else:
            raise
        return data[(time > self.target_time - 0.4) & (time < self.target_time + 0.1)]

    def whiten(self, signal, sampling_rate, f_ASD=None, ASD=None, return_tilde=False, return_asd=False):
        """
        Ref:
            https://github.com/moble/MatchedFiltering/blob/3d7f3b5ca1383e26b80e64c82a703fe054f16c40/utilities.py#L155
        """
        f_signal = rfftfreq(len(signal), 1./sampling_rate)
        if ASD is not None:
            psd = np.abs(interp1d(f_ASD, ASD**2, bounds_error=False, fill_value=np.inf)(f_signal))
        else:
            f_psd, psd = welch(signal, sampling_rate, nperseg=2**int(np.log2(len(signal)/8.0)), scaling='density')
            psd = np.abs(interp1d(f_psd, psd, bounds_error=False, fill_value=np.inf)(f_signal))
        signal_filtered_tilde = rfft(signal) / np.sqrt(0.5 * sampling_rate * psd)
        if return_tilde:
            return irfft(signal_filtered_tilde), signal_filtered_tilde
        elif return_asd:
            return irfft(signal_filtered_tilde), (f_signal, np.sqrt(psd))
        else:
            return irfft(signal_filtered_tilde)

    def calc_matches(self, d1, d2):
        fft1 = np.fft.fft(d1)
        fft2 = np.fft.fft(d2)
        norm1 = np.mean(np.abs(fft1)**2)
        norm2 = np.mean(np.abs(fft2)**2)
        inner = np.mean(fft1.conj()*fft2).real
        return inner / np.sqrt(norm1 * norm2)
