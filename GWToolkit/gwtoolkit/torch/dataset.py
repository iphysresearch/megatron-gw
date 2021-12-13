"""
Data set based on PyTorch
"""

from collections import namedtuple
from collections.abc import Iterable
import torch
# from torch.utils.data import Dataset
import pandas
import numpy


class WaveformDatasetTorch(torch.utils.data.Dataset):
    """Waveform dataset

    Usage:
    >>> wfdt = WaveformDatasetTorch(...)
    >>> wfdt.update()
    """

    def __init__(self, wfd, num, start_time, geocent_time,
                 target_optimal_snr_tuple=None,
                 target_labels=None,
                 stimulated_whiten_ornot=False,
                 transform_data=None,
                 transform_params=None,
                 rand_transform_data=None,
                 classification_ornot=None,
                 denoising_ornot=None):
        """
        # !@param Args: ........................................................................................................................
        #   wfd             :   WaveformDataset class
        #   num             :   waveform number
        #   start_time      :   signal start GPS time (GW150914 GPS_time - 6s)
        #   geocent_time    :   tuple, optional, (minimum, maximum)
                                The GPS arrival time of the signal data
                                (GPS_time-0.1, GPS_time+0.1)
        #   target_optimal_snr_tuple    :   tuple, (target_detector_index, target_optimal_snr) :: (0, 0)
        #   target_labels   :   params labels str :
                                ['mass_ratio', 'chirp_mass',
                                 'luminosity_distance',
                                 'dec', 'ra', 'theta_jn', 'psi', 'phase',
                                 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                                 'geocent_time']
        #   stimulated_whiten_ornot     :   whiten or not
        #   transform_data  :   torchvision.transforms.transforms.Compose(
            [
                # ! Patching_data(
                        patch_size              : 0.5
                        overlap,                : 0.5
                        sampling_frequency      : 4096
                        ),
                # ! output: (b c h)
                        b : batch_size,
                        c : (duration / (patch_size * (1 - overlap)) - 1) * len(dets),
                   Example:     8            0.5              0.5              2 (H1 and L1)  [result :: 62]
                        h : sample_frequency * patch_size,   here: 2048
            ]
        )

        #   transform_params            :   torchvision.transforms.transforms.Compose([ Normalize_params() ]) for rescale to (-1,1)
        #   rand_transform_data         :   torchvision.transforms.transforms.Compose([ ToTensor() ])
        #   classification_ornot        :   if for classification -- train & target or just train data
        # !................................................................................................................................................................................................

        """
        assert isinstance(num, int)
        assert isinstance(start_time, float)
        assert isinstance(geocent_time, Iterable)
        assert (isinstance(target_optimal_snr_tuple, Iterable) if target_optimal_snr_tuple is not None else True)
        Record = namedtuple('Record', 'num start_time geocent_time \
                             target_labels \
                             target_optimal_snr_tuple \
                             stimulated_whiten_ornot \
                             classification_ornot \
                             denoising_ornot')
        self.var = Record(num, start_time, geocent_time,
                          self._set_target_labels(target_labels),
                          target_optimal_snr_tuple,
                          stimulated_whiten_ornot,
                          classification_ornot,
                          denoising_ornot)
        self.wfd = wfd
        self.transform_data = transform_data
        self.transform_params = transform_params
        self.rand_transform_data = rand_transform_data

        self.time_array = None
        self.data_block = None
        self.signal_block = None
        self.noise_block = None
        self.params_block = None
        self.label_block = None
        self.update()

    def __len__(self):
        if self.var.classification_ornot is not None:
            return 2*self.var.num
        return self.var.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if idx == 0:  # Update self.data_block and self.params_block
        #     self.update()

        if self.rand_transform_data:
            self.data_block[idx] = self.rand_transform_data(self.data_block[idx])

        if self.var.denoising_ornot is not None:
            return (self.data_block[idx], self.signal_block[idx], self.noise_block[idx], self.params_block[idx])
        if self.var.classification_ornot is not None:
            return (self.data_block[idx], self.label_block[idx])
        return (self.data_block[idx], self.params_block[idx])

    def update(self):
        """
        Update noise and waveform responce for
        self.data_block and self.params_block/self.label_block
        """
        # data =
        # (signal_block,  # Pure signals, (num, len(wfd.dets), ...)
        #  signal_meta,  # parameters of the signals, dict
        #  noise_block,  # Pure colored detector noises, (num, len(wfd.dets), ...)
        #  data_whitened, # mixed signal+noise data whitened by stimulated dets' PSD
        # )
        data = self.wfd.time_waveform_response_block(
            self.var.num,
            self.var.start_time,
            self.var.geocent_time,
            self.var.target_optimal_snr_tuple,
        )
        if self.var.stimulated_whiten_ornot:
            self.data_block = data[3]
            self.signal_block = data[4]
            self.noise_block = data[5]
        else:
            self.data_block = data[0] + data[2]

        # Consider the target params labels
        # self.params_block = pandas.DataFrame({key: data[1][key]
        #                                       for key in data[1].keys()
        #                                       if key in self.var.target_labels}).values
        self.params_block = pandas.DataFrame({key: data[1][key]
                                              for key in data[1].keys()
                                              if key in self.var.target_labels
                                              if 'snr' not in key})
        for key in self.var.target_labels:
            if 'snr' not in key:
                continue
            for i in range(len(self.wfd.dets)):
                self.params_block[key+f'_{i}'] = data[1][key][:,i]
        self.params_block = self.params_block.values

        if self.var.classification_ornot is not None:
            self.label_block = numpy.concatenate((numpy.ones(len(self.data_block)),
                                                  numpy.zeros(len(data[2]))))
            self.data_block = numpy.concatenate((self.data_block, data[2]))

        if self.transform_data:
            self.data_block = self.transform_data(self.data_block)
            if self.var.denoising_ornot is not None:
                self.signal_block = self.transform_data(self.signal_block)
                self.noise_block = self.transform_data(self.noise_block)
        if self.transform_params:
            self.params_block = self.transform_params(self.params_block)

    def _set_target_labels(self, labels=None):
        """
        Set target labels for dataset
        """
        # default 15 labels
        # _labels = ('mass_ratio', 'chirp_mass',
        #            'luminosity_distance',
        #            'dec', 'ra', 'theta_jn', 'psi', 'phase',
        #            'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
        #            'geocent_time')
        return labels
        # return list(set(_labels) & set(labels)) if labels is not None else _labels
        # self.data_block = signal_block + noise_block
        # self.time_array = self.wfd.time_array

    def transform_data_block(self, data_block):
        """
        data_block is a block
        """
        if self.transform_data:
            data_block = self.transform_data(data_block)
        if self.rand_transform_data:
            return self.rand_transform_data(data_block)
        return data_block

    def transform_inv_params(self, params_block):
        """
        inverse transform to physical parameters
        """
        return self.transform_params.transforms[0].standardize_inv(params_block)
