"""
<class 'numpy.ndarray'>
[ 7.23505964 -2.36367101  6.06850284 ... -0.71043373  3.44410641
 -2.48990505]
(32768,)
float64
1126259456.391
"""
import sys
sys.path.append('../../GWToolkit')
from gwtoolkit.gw import WaveformDataset
from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
import numpy

from gwtoolkit.gw.gwosc_cvmfs import getstrain_cvmfs, FileList
from gwtoolkit.utils import pickle_read
import scipy.signal
from bilby.core import utils
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

filename = '../../GWToolkit/tests/gw/demo.prior'   # default prior file

# waveform dataset
wfd = WaveformDataset(sampling_frequency=sampling_frequency,
                      duration=duration,
                      conversion=conversion)

wfd.load_prior_source_detector(
    filename=filename,
    base=base,
    dets=dets,
    waveform_arguments=waveform_arguments)

data_dir = '/workspace/zhaoty/dataset/O1_H1_All/'
wfd.dets['H1'].load_from_GWOSC(data_dir, 1024, selected_hdf_file_ratio=0)
GWTC1_events = pickle_read('/workspace/zhaoty/GWToolkit/gwtoolkit/gw/metadata/GWTC1_events.pkl')
filelist = FileList(directory=data_dir)
ifo = 'H1'

# GW151012 GW151226 GW150914
target_time = GWTC1_events['GW150914']['trigger-time'] 
PSD_strain, _, _, _ = getstrain_cvmfs(target_time - 6  -1024, target_time - 6  , ifo, filelist)
seg_sec = 0.1
freq, Pxx = scipy.signal.welch(PSD_strain, fs=sampling_frequency,
                               nperseg=seg_sec*sampling_frequency, )

# 
wfd.dets['H1'].ifo.power_spectral_density = wfd.dets['H1'].ifo.power_spectral_density.from_power_spectral_density_array(freq, Pxx)

start = target_time - 6         # step=0.5, -200 +50
strain, time, dqmask, injmask = getstrain_cvmfs(start, start+duration, ifo, filelist)
strain = strain[::4]
time = time[::4]
freq_domain_strain, freq = wfd.dets['H1'].time_to_frequency_domain(strain)
whiten_freq_domain_strain = freq_domain_strain / wfd.dets['H1'].amplitude_spectral_density_array
whiten_time_domain_strain = utils.infft(whiten_freq_domain_strain, sampling_frequency)

print(type(whiten_time_domain_strain))
print(whiten_time_domain_strain)
print(whiten_time_domain_strain.shape)
print(whiten_time_domain_strain.dtype)
print(start)