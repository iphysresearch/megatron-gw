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

class GwDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, seed=1234):

        # Params to store.
        self.name = name
        self.seed = seed

        # Build the samples.
        self.samples = get_samples(data_prefix, self.name)
        self.noisy = self.samples['noisy']
        self.clean = self.samples['clean']
        self.params = self.samples['params']
        assert self.noisy.shape == self.clean.shape
        self.fs = 4096
        self.seg = 0.5
        self.step = 0.25
        self.duration = 8
        self.patches = int((self.duration - self.seg) // self.step) + 1
        self.step_samples = int(self.seg * self.fs)

    def __len__(self):
        return self.noisy.shape[0]

    def __getitem__(self, idx):
        tmp_idx = idx % self.noisy.shape[0]
        noisy_np = self.noisy[tmp_idx]
        clean_np = self.clean[tmp_idx]
        param_np = np.real(self.params[tmp_idx])
        noisy_input = np.zeros([self.patches, self.step_samples], dtype=self.noisy.dtype)
        clean_input = np.zeros([self.patches, self.step_samples], dtype=self.clean.dtype)
        for ind in range(self.patches):
            start = int(ind * self.step * self.fs)
            noisy_input[ind] = noisy_np[0, start:start + self.step_samples]
            clean_input[ind] = clean_np[0, start:start + self.step_samples]

        return {
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'params': param_np}

def get_samples(data_prefix, name):
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    dataset = {}
    if name in ["valid", "test"]:
        data_path = os.path.join(data_prefix, name + '.hdf5')
        f_data = h5py.File(data_path, 'r')
        for data_name in ['noisy', 'clean']:
            dataset[data_name] = f_data[data_name][:, :, :]
        dataset['params'] = f_data['params'][:, :]
        f_data.close()
    else:
        for i in range(1, 2): #11):
            #data_path = os.path.join(data_prefix, "{}-{}.hdf5".format(name, i))
            data_path = os.path.join(data_prefix, "{}.hdf5".format(name))
            f_data = h5py.File(data_path, 'r')
            if i == 1:
                for data_name in ['noisy', 'clean']:
                    dataset[data_name] = f_data[data_name][:, :, :]
                dataset['params'] = f_data['params'][:, :]
            else:
                for data_name in ['noisy', 'clean']:
                    dataset[data_name] = np.append(dataset[data_name], f_data[data_name][:, :, :], axis=0)
                dataset['params'] = np.append(dataset['params'], f_data['params'][:, :], axis=0)
            f_data.close()

    return dataset

