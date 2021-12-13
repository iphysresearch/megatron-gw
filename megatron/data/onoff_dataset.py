# coding=utf-8
# Copyright (c) 2020, PCL.  All rights reserved.
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
from megatron import (
    mpu,
    print_rank_0
)
import sys
sys.path.append('../../GWToolkit')
from gwtoolkit.gw import WaveformDataset
from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
from gwtoolkit.gw.gwosc_cvmfs import getstrain_cvmfs, FileList
from gwtoolkit.utils import pickle_read
import scipy.signal
from bilby.core import utils


class OnsourceDataset(torch.utils.data.Dataset):

    def __init__(self, data_prefix, seed=1234):

        self.sampling_frequency = 4096
        self.seg = 0.5
        self.step = 0.25
        self.duration = 8
        self.patches = int((self.duration - self.seg) // self.step) + 1
        self.step_samples = int(self.seg * self.sampling_frequency)

        self.conversion = 'BBH'
        self.waveform_approximant = 'IMRPhenomPv2'
        self.reference_frequency = 50.
        self.minimum_frequency = 20.
        self.waveform_arguments = dict(waveform_approximant=self.waveform_approximant,
                                reference_frequency=self.reference_frequency,
                                minimum_frequency=self.minimum_frequency)
        self.base = 'bilby'
        self.dets = ['H1', 'L1'][:1]

        self.filename = '../../GWToolkit/tests/gw/demo.prior'   # default prior file

        # waveform dataset
        self.wfd = WaveformDataset(sampling_frequency=self.sampling_frequency,
                            duration=self.duration,
                            conversion=self.conversion)

        self.wfd.load_prior_source_detector(
            filename=self.filename,
            base=self.base,
            dets=self.dets,
            waveform_arguments=self.waveform_arguments)

        self.data_dir = '/workspace/zhaoty/dataset/O1_H1_All/'
        self.wfd.dets['H1'].load_from_GWOSC(self.data_dir, 1024, selected_hdf_file_ratio=0)
        self.GWTC1_events = pickle_read('/workspace/zhaoty/GWToolkit/gwtoolkit/gw/metadata/GWTC1_events.pkl')
        self.filelist = FileList(directory=self.data_dir)
        self.ifo = 'H1'

        # GW151012 GW151226 GW150914
        self.target_time = self.GWTC1_events['GW150914']['trigger-time']
        self.PSD_strain, _, _, _ = getstrain_cvmfs(self.target_time - 6  -1024, self.target_time - 6 , self.ifo, self.filelist)
        self.seg_sec = 0.1
        self.freq, self.Pxx = scipy.signal.welch(self.PSD_strain, fs=self.sampling_frequency,
                                    nperseg=self.seg_sec*self.sampling_frequency, )

    def __getitem__(self, idx):
        # This should be a barrier but nccl barrier assumes
        # device_index=rank which is not the case for model
        # parallel case
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
        assert counts[0].item() == (
            torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

        self.wfd.dets['H1'].ifo.power_spectral_density = self.wfd.dets['H1'].ifo.power_spectral_density.from_power_spectral_density_array(self.freq, self.Pxx)
        start_time = self.target_time - idx * 0.5 + 50        # step=0.5, -200 +50
        strain, time, dqmask, injmask = getstrain_cvmfs(start_time, start_time+self.duration, self.ifo, self.filelist)
        strain = strain[::4]
        time = time[::4]
        freq_domain_strain, freq = self.wfd.dets['H1'].time_to_frequency_domain(strain)
        whiten_freq_domain_strain = freq_domain_strain / self.wfd.dets['H1'].amplitude_spectral_density_array
        whiten_time_domain_strain = utils.infft(whiten_freq_domain_strain, self.sampling_frequency)

        noisy_input = np.zeros([self.patches, self.step_samples], dtype=whiten_time_domain_strain.dtype)
        clean_input = np.zeros([self.patches, self.step_samples], dtype=whiten_time_domain_strain.dtype)

        for ind in range(self.patches):
            start_idx = int(ind * self.step * self.sampling_frequency)
            noisy_input[ind] = whiten_time_domain_strain[start_idx:start_idx + self.step_samples]
            #clean_input[ind] = clean_np[0, start_idx:start_idx + self.step_samples]
        # print("========", start_time)
        train_sample = {
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'params': np.array(start_time)}

        return train_sample

class OffsourceDataset(torch.utils.data.Dataset):

    def __init__(self, data_prefix, seed=1234):

        self.sampling_frequency = 4096
        self.seg = 0.5
        self.step = 0.25
        self.duration = 8
        self.patches = int((self.duration - self.seg) // self.step) + 1
        self.step_samples = int(self.seg * self.sampling_frequency)

        self.conversion = 'BBH'
        self.waveform_approximant = 'IMRPhenomPv2'
        self.reference_frequency = 50.
        self.minimum_frequency = 20.
        self.waveform_arguments = dict(waveform_approximant=self.waveform_approximant,
                                reference_frequency=self.reference_frequency,
                                minimum_frequency=self.minimum_frequency)
        self.base = 'bilby'
        self.dets = ['H1', 'L1'][:1]

        self.filename = '../../GWToolkit/tests/gw/demo.prior'   # default prior file

        # waveform dataset
        self.wfd = WaveformDataset(sampling_frequency=self.sampling_frequency,
                            duration=self.duration,
                            conversion=self.conversion)

        self.wfd.load_prior_source_detector(
            filename=self.filename,
            base=self.base,
            dets=self.dets,
            waveform_arguments=self.waveform_arguments)

        self.data_dir = '/workspace/zhaoty/dataset/O1_H1_All'
        self.wfd.dets['H1'].load_from_GWOSC(data_dir, duration, selected_hdf_file_ratio=0)

        self.wfd.dets['H1'].update_time_domain_strain_from_GWOSC(seg_sec=2)
        self.noise = self.wfd.dets['H1'].time_domain_whitened_strain
        self.target_optimal_snr = 20
        self.alpha = 1

    def __getitem__(self, idx):
        # This should be a barrier but nccl barrier assumes
        # device_index=rank which is not the case for model
        # parallel case
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
        assert counts[0].item() == (
            torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

        self.wfd._update_waveform()
        self.start_time = self.wfd.dets['H1'].gwosc.start
        self.buffer_time = 2
        self.wfd.parameters['geocent_time'] = np.asarray([self.start_time+(self.duration - self.buffer_time)])

        self.external_parameters = {k: self.wfd.parameters[k][0] for k in self.wfd._external_parameters}
        temp = self.wfd.dets['H1']
        self.alpha = self.target_optimal_snr / self.wfd.dets['H1'].optimal_snr(self.wfd.frequency_waveform_response[0] )
        signal, time_array = temp.frequency_to_time_domain(temp.whiten(self.alpha * temp.get_detector_response(self.wfd.frequency_waveform_polarizations, self.external_parameters)))
        data = signal + self.noise
        noisy_input = np.zeros([self.patches, self.step_samples], dtype=self.noise.dtype)
        clean_input = np.zeros([self.patches, self.step_samples], dtype=self.noise.dtype)

        for ind in range(self.patches):
            start_idx = int(ind * self.step * self.sampling_frequency)
            noisy_input[ind] = data[start_idx:start_idx + self.step_samples]
            clean_input[ind] = signal[start_idx:start_idx + self.step_samples]

        train_sample = {
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'params': np.array(ind)}

        return train_sample
