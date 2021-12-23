"""
Data set based on PyTorch / Redis
"""
from gwpy.signal import filter_design
from gwtoolkit.torch import Patching_data, Normalize_strain
from gwtoolkit.gw.readligo import FileList, getstrain_cvmfs
from gwpy.timeseries import TimeSeries
from numpy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import welch
import numpy as np

import torch
import redis
import msgpack_numpy as m
m.patch()               # Important line to monkey-patch for numpy support!


class DatasetTorchRedis(torch.utils.data.Dataset):
    """Waveform dataset using Redis

    Usage:
    >>> dataset = DatasetTorchRedis()

    Ref: https://stackoverflow.com/a/60537304/8656360
    """

    def __init__(self, host='localhost', port=6379):
        connection_pool = redis.ConnectionPool(host=host, port=port,
                                               db=0, decode_responses=False)
        self.r = redis.Redis(connection_pool=connection_pool)
        self.data_keys = sorted(self.r.keys('data_*'))
        self.signal_keys = sorted(self.r.keys('signal_*'))
        self.params_keys = sorted(self.r.keys('params_*'))

    def __len__(self):
        assert len(self.data_keys) == len(self.signal_keys)
        return len(self.data_keys)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        return (self.fromRedis(self.data_keys[idx]),
                self.fromRedis(self.signal_keys[idx]),
                self.fromRedis(self.params_keys[idx]))

    def fromRedis(self, name):
        """Retrieve Numpy array from Redis key 'n'"""
        # Retrieve and unpack the data
        try:
            return m.unpackb(self.r.get(name))
        except TypeError:
            print('No this value')


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
        self.input_strain, self.dMin, self.dMax = self.strain_preprocessing(self.strain_valid)  


    def __len__(self):
        print(f'input data"s shape: {self.input_strain.shape}')
        return len(self.input_strain)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input_strain[idx]

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
        return data, dMin, dMax

    def signal_postprocessing(self, signal):
        return self.normfunc.inverse_transform_signalornoise(signal, self.dMin, self.dMax)

    def metric(self, output_signal, rebuild_on_forer=True):
        # (1, 127, 2048) => (131072,) => (2,)
        output_whiten_signal = self.signal_postprocessing(output_signal)
        output_whiten_signal = self.rebuild_forer(output_whiten_signal[0]) if rebuild_on_forer else self.rebuild_backer(output_whiten_signal[0])
        return (self.calc_matches(self.cut_from_long(self.denoised_strain_valid) , output_whiten_signal),
                self.calc_matches(self.cut_for_target(self.denoised_strain_valid), self.cut_for_target(output_whiten_signal)))

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
        return data[(time > self.target_time - 0.1) & (time < self.target_time + 0.05)]

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

if __name__ == "__main__":
    import sys
    sys.path.append('/workspace/zhaoty/GWToolkit/')
    from gwtoolkit.redis import DatasetTorchRedis

    dataset = DatasetTorchRedis()
    print(len(dataset))
    data, signal, params = dataset[4]
    print(data.shape, signal.shape, len(params.keys()), params['geocent_time'].shape)
