import pytest
import sys
sys.path.append('/home/zty/GWToolkit')
sys.path.append('..')
from gwtoolkit.gw import WaveformDataset
from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
import numpy

# 1. Init for WaveformDataset
sampling_frequency = 4096
duration = 8
conversion = 'BBH'
waveform_approximant = 'IMRPhenomPv2'
# waveform_approximant = 'SEOBNRv4P' ## TODO SEOBNRv4P 有报错
# waveform_approximant = 'IMRPhenomPv2_NRTidal'
# waveform_approximant = 'IMRPhenomXP'
reference_frequency = 50.
minimum_frequency = 20.
waveform_arguments = dict(waveform_approximant=waveform_approximant,
                          reference_frequency=reference_frequency,
                          minimum_frequency=minimum_frequency)
base = 'bilby'
dets = ['H1', 'L1']
wfd = WaveformDataset(sampling_frequency=sampling_frequency,
                      duration=duration,
                      conversion=conversion)
filename = 'tests/gw/demo.prior'
wfd.load_prior_source_detector(
    filename=filename,
    base=base,
    dets=dets,
    waveform_arguments=waveform_arguments)
###

# 2. Init for WaveformDatasetTorch
patch_size = 0.5
overlap = 0.5
num = 50
target_time = 1126259456.3999023 + 6
buffer_time = 2
start_time = target_time-(duration - buffer_time)
geocent_time = (target_time-0.1, target_time+0.1)
target_optimal_snr_tuple = (0, 0)
norm_params_kind = 'minmax'
###

target_labels = ['mass_ratio', 'chirp_mass',
                 'luminosity_distance',
                 'dec', 'ra', 'theta_jn', 'psi', 'phase',
                 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                 'geocent_time']
# Hyper parameters end ###############################
composed_data = transforms.Compose([
    Patching_data(patch_size=patch_size,
                  overlap=overlap,
                  sampling_frequency=sampling_frequency),
    # output: (b c h)
])
rand_transform_data = transforms.Compose([
    ToTensor(),
])
composed_params = transforms.Compose([
    Normalize_params(norm_params_kind,
                     wfd=wfd, labels=target_labels,
                     feature_range=(-1, 1)),
    ToTensor(),
])


@pytest.fixture
def init_wfct():
    return lambda stimulated_whiten_ornot, classification_ornot: \
        WaveformDatasetTorch(wfd, num=num,
                             start_time=start_time,
                             geocent_time=geocent_time,
                             target_optimal_snr_tuple=target_optimal_snr_tuple,
                             target_labels=target_labels,
                             stimulated_whiten_ornot=stimulated_whiten_ornot,
                             transform_data=composed_data,
                             transform_params=composed_params,
                             rand_transform_data=rand_transform_data,
                             classification_ornot=classification_ornot)


def test_wfct(init_wfct):
    for stimulated_whiten_ornot, classification_ornot in itertools.product([True, False], [True, None]):
        wfct = init_wfct(stimulated_whiten_ornot, classification_ornot)
        if classification_ornot:
            assert wfct.data_block.shape == (num * 2, 62, 2048)
            assert wfct.label_block.shape == (num * 2,)
            assert len(wfct) == num * 2
        else:
            assert wfct.data_block.shape == (num, 62, 2048)
            assert wfct.params_block.shape == (num, 15)
            assert len(wfct) == num
            assert wfct.transform_data_block(wfct.data_block).shape == wfct.data_block.shape
            assert wfct.transform_inv_params(wfct.params_block).shape == wfct.params_block.shape

        # DataLoader objects
        loader = DataLoader(wfct, batch_size=3, shuffle=True, pin_memory=False,
                            num_workers=0, worker_init_fn=lambda _: numpy.random.seed(
                                int(torch.initial_seed()) % (2**32-1)))
        for d, l in loader:
            break

        if classification_ornot:
            assert d.shape == torch.Size([3, 62, 2048])
            assert l.shape == torch.Size([3])
        else:
            assert d.shape == torch.Size([3, 62, 2048])
            assert l.shape == torch.Size([3, 15])
