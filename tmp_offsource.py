"""
<class 'numpy.ndarray'>
[-108.75144716   52.61873073   44.92666656 ...   -7.75700653  -14.35923332
  -96.70833528]
(32768,)
float64
"""
import sys
sys.path.append('./GWToolkit/')
from gwtoolkit.gw import WaveformDataset
from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
import numpy

import matplotlib.pyplot as plt

# 1. Init for WaveformDataset
sampling_frequency = 4096     # [Hz], sampling rate
duration = 8                  # [sec], duration of a sample
conversion = 'BBH'
waveform_approximant = 'IMRPhenomPv2'
# waveform_approximant = 'SEOBNRv4P' ## TODO
# waveform_approximant = 'IMRPhenomPv2_NRTidal'
# waveform_approximant = 'IMRPhenomXP'
reference_frequency = 50.
minimum_frequency = 20.
waveform_arguments = dict(waveform_approximant=waveform_approximant,
                          reference_frequency=reference_frequency,
                          minimum_frequency=minimum_frequency)
base = 'bilby'
dets = ['H1', 'L1'][:1]

filename = './GWToolkit/tests/gw/demo.prior'   # default prior file

# waveform dataset
wfd = WaveformDataset(sampling_frequency=sampling_frequency,
                      duration=duration,
                      conversion=conversion)

wfd.load_prior_source_detector(
    filename=filename,
    base=base,
    dets=dets,
    waveform_arguments=waveform_arguments)

data_dir = '/workspace/zhaoty/dataset/O1_H1_All'
wfd.dets['H1'].load_from_GWOSC(data_dir, duration, selected_hdf_file_ratio=0)

wfd.dets['H1'].update_time_domain_strain_from_GWOSC(seg_sec=2)   # seg_sec 表示计算 PSD 的切段长度
noise = wfd.dets['H1'].time_domain_whitened_strain

# plt.plot(wfd.dets['H1'].time_array, noise)

wfd._update_waveform()
start_time = wfd.dets['H1'].gwosc.start
buffer_time = 2
wfd.parameters['geocent_time'] = numpy.asarray([start_time+(duration - buffer_time)])

target_optimal_snr = 50
target_optimal_snr /= wfd.dets['H1'].optimal_snr(wfd.frequency_waveform_response[0] )

external_parameters = {k: wfd.parameters[k][0] for k in wfd._external_parameters}
temp = wfd.dets['H1']
signal, time_array = temp.frequency_to_time_domain(temp.whiten(target_optimal_snr * temp.get_detector_response(wfd.frequency_waveform_polarizations, external_parameters)))

# plt.plot(time_array, signal)

data = noise + signal

print(type(data))
print(data)
print(data.shape)
print(data.dtype)