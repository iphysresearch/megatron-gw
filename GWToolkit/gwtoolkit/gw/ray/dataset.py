"""
Data set based on Ray and PyTorch
"""

from gwtoolkit.gw import WaveformDataset
from gwtoolkit.torch import Patching_data, Normalize_strain
import numpy as np
import pandas as pd

from tqdm import tqdm

from numpy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import welch

from bilby.gw import utils as gwutils
import ray
from torch.utils.data import IterableDataset

from ray.data.impl.torch_iterable_dataset import \
    TorchIterableDataset

# runtime_env = {"working_dir": "/workspace/zhaoty/GWToolkit/",
#                "excludes": ['/workspace/zhaoty/GWToolkit/notebooks/',
#                             '/workspace/zhaoty/GWToolkit/.git/']}
runtime_env = {"working_dir": "/workspace/zhouy/megatron-gw/GWToolkit/",
               "excludes": ['/workspace/zhouy/megatron-gw/GWToolkit/notebooks/']}

ray.shutdown()
ray.init(num_cpus=None, num_gpus=0, runtime_env=runtime_env)
print(ray.cluster_resources())
# {'GPU': 8.0,
#  'object_store_memory': 10000000000.0,
#  'CPU': 96.0,
#  'accelerator_type:V100': 1.0,
#  'node:172.17.0.2': 1.0,
#  'memory': 1588282640384.0}


@ray.remote
class RayDataset():
    """
    Dataset for Ray
    """
    def __init__(self):
        """Instantiate a ...
        """
        noise_interval=1024
        num_length=2
        data_dir='/workspace/zhaoty/dataset/O1_H1_All/'
        selected_hdf_file_ratio=0.5
        
        ##### For raw LIGO data as noises  FIXME
        noise_interval=32
        num_length=1
        data_dir='/workspace/zhaoty/dataset/GW150914_around/'
        selected_hdf_file_ratio=1.0
        ##### For raw LIGO data as noises

        # 1. Init for WaveformDataset
        self.sampling_frequency = 4096*4     # [Hz], sampling rate
        self.duration_long = 32            # [sec], duration of a sample
        self.duration = 8                  # [sec], duration of a sample
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
        self.dets = ['H1', 'L1'][:1]  # TODO

        # filename = '/workspace/zhaoty/GWToolkit/tests/gw/demo.prior'   # default prior file
        filename = '/workspace/zhaoty/GWToolkit/gwtoolkit/gw/prior_files/default.prior'   # default prior file

        # waveform dataset
        self.wfd = WaveformDataset(sampling_frequency=self.sampling_frequency,
                                   duration=self.duration_long,
                                   conversion=conversion)

        self.wfd.load_prior_source_detector(
            filename=filename,
            base=base,
            dets=self.dets,
            waveform_arguments=waveform_arguments,
            verbose=False)

        ####################################################################################
        # Set for waveforms
        # target_time = 1126259456.3999023 + 6
        self.target_time = 1126259462.425  # GWTC1_events['GW150914']['trigger-time']  # TODO
        buffer_time = self.duration_long / 2 - self.duration / 4
        self.start_time = self.target_time-(self.duration_long - buffer_time)
        self.geocent_time = (self.target_time-5.1, self.target_time+1.1)

        ####################################################################################
        # Set for noises
        self.data_dir = data_dir
        self.noise_interval = noise_interval
        self.selected_hdf_file_ratio = selected_hdf_file_ratio
        self.num_length = num_length

        # Used for estimate PSDs
        self.seg_sec = 1
        self.random_timedomain_ratio = 0.3

        # Used for generate backgrounds
        self.freq = None
        self.Pxx = None

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
        low, high = 13, 30
        self.strategy_for_target_optimal_snr = (low, high)  # or None or Float/Int
        # PSD used for whiten
        self.fasd = None
        self.asd = None

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

        # Set for standardizing strains
        feature_range = (-1, 1)
        self.normfunc = Normalize_strain(feature_range)

        # Set for filters targeting GW150914


        # Init
        self.update_waveforms()
        while True:
            try:
                self.update_backgrounds(4)
                break
            except ValueError:
                pass

    @property
    def target_optimal_snr(self):
        if isinstance(self.strategy_for_target_optimal_snr, tuple):
            return np.random.uniform(*self.strategy_for_target_optimal_snr)
        elif self.strategy_for_target_optimal_snr is not None:
            return self.strategy_for_target_optimal_snr
        else:
            return None

    def show_token_shape(self):
        return self.token_shape

    def show_parameters(self):
        return self.wfd.parameters

    def show_num_duration_long(self):
        return self.duration_long*self.sampling_frequency

    def show_num_duration(self):
        return self.duration*self.sampling_frequency

    def show_sampling_frequency(self):
        return self.sampling_frequency

    def show_geocent_time(self):
        return self.geocent_time

    def show_patching_func(self):
        return self.patching

    def setup_filters(self):
        # Return whiten and denoised strain for GW150914
        from gwpy.signal import filter_design
        from gwtoolkit.gw.readligo import FileList, getstrain_cvmfs
        from gwpy.timeseries import TimeSeries

        bp = filter_design.bandpass(50, 250, self.sampling_frequency)
        notches = [filter_design.notch(line, self.sampling_frequency) for
                   line in (60, 120, 180)]
        self.zpk = filter_design.concatenate_zpks(bp, *notches)

        filelist = FileList(directory=self.data_dir)
        strain_valid, time_valid = getstrain_cvmfs(self.start_time, self.start_time + self.duration_long, 'H1', filelist, inj_dq_cache=0)
        hdata = TimeSeries(strain_valid, times=time_valid, channel='H1')
        hfilt = hdata.whiten(4, 2).filter(self.zpk, filtfilt=True)
        return (time_valid, strain_valid), (hfilt.times.value, hfilt.value)

    def calc_matches(self, d1, d2):
        fft1 = np.fft.fft(d1)
        fft2 = np.fft.fft(d2)
        norm1 = np.mean(np.abs(fft1)**2)
        norm2 = np.mean(np.abs(fft2)**2)
        inner = np.mean(fft1.conj()*fft2).real
        return inner / np.sqrt(norm1 * norm2)

    def update_backgrounds(self, level):
        """
        Level:
        4: Reselect % of hdf5s -> Resample strains from selected hdf5s -> Restimate PSD by resegmenting -> Regenete BG from PSD
        3:                        Resample strains from selected hdf5s -> Restimate PSD by resegmenting -> Regenete BG from PSD
        2:                                                                Restimate PSD by resegmenting -> Regenete BG from PSD
        1:                                                                                                 Regenete BG from PSD
        0: Use raw data from LIGO
        """
        assert level in [1, 2, 3, 4, 0], f'You should input `level` within [1, 2, 3, 4, 0], but you input `level`={level}'
        level_list = [0]*(4-level)+[1]*level
        # Loop for noises
        for det in self.dets:
            if level_list[0]:
                # Select % of hdf5 files (Randomly)
                self.wfd.dets[det].load_from_GWOSC(self.data_dir, self.noise_interval, self.selected_hdf_file_ratio,
                                                   self.num_length, verbose=False)
            if level == 0:
                self.wfd.dets[det].gwosc.update_strain()  # Don't know why it works....
                return

            if level_list[1]:
                # Sampling strain_multi and time_multi (Randomly)
                self.wfd.dets[det].gwosc.update_randomly_strain()
            if level_list[2]:
                # Estimate PSD from strain_multi/time_multi using welch_modified (Randomly)
                self.freq, self.Pxx = self.wfd.dets[det].gwosc.estimate_randomly_psd_cvmfs(self.seg_sec,
                                                                                           self.random_timedomain_ratio,
                                                                                           noverlap=0)
            if level_list[3]:
                # It will cover the PSD and generate new noises (Randomly)
                self.wfd.dets[det].ifo.power_spectral_density = \
                    self.wfd.dets[det].ifo.power_spectral_density.from_power_spectral_density_array(self.freq, self.Pxx)
                self.wfd.dets[det].update()  # corresponding to duration_long
                # wfd.dets[det].ifo.maximum_frequency = 512  # Can it be used?

    def update_waveforms(self):
        # Loop for waveforms
        self.wfd._update_waveform()  # update internal parameters

        # update external parameters / start_time + level=1
        # self.wfd.update_detector_response(start_time=self.start_time, geocent_time=self.geocent_time)

        # update external parameters (on the right bound) / start_time + level=1
        right_bound_geocent_time = self.geocent_time[1]
        self.wfd.update_detector_response(start_time=self.start_time,
                                          geocent_time=(right_bound_geocent_time-0.001,
                                                        right_bound_geocent_time))

        self.wfd.parameters.update(dict(
            optimal_snr=np.asarray([]),
            matched_filter_snr=np.asarray([]),
        ))

    def calc_SNR(self, alpha, asd):
        # TODO need to support multi-detectors
        det = 'H1'
        data = self.wfd.dets[det].ifo.strain_data.time_domain_strain + self.wfd.time_waveform_response[0] * alpha
        return \
            np.sqrt(gwutils.optimal_snr_squared(
                signal=self.wfd.frequency_waveform_response[0][self.wfd.dets[det].ifo.strain_data.frequency_mask] * alpha,
                power_spectral_density=np.power(asd,2)[self.wfd.dets[det].ifo.strain_data.frequency_mask],
                duration=self.wfd.dets[det].ifo.strain_data.duration).real), \
            gwutils.matched_filter_snr(
                        signal=self.wfd.frequency_waveform_response[0][self.wfd.dets[det].ifo.strain_data.frequency_mask] * alpha,
                        frequency_domain_strain=self.wfd.dets[det].time_to_frequency_domain(data)[0][self.wfd.dets[det].ifo.strain_data.frequency_mask],
                        power_spectral_density=np.power(asd,2)[self.wfd.dets[det].ifo.strain_data.frequency_mask],
                        duration=self.wfd.dets[det].ifo.strain_data.duration)

    def cut_from_long(self, data):
        left_index = int((self.duration_long - self.duration)/2*self.sampling_frequency)
        right_index = int((self.duration_long + self.duration)/2*self.sampling_frequency)
        return data[left_index: right_index]

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

    def generate_sample(self, use_which_design_ASD_for_whiten=None, target_optimal_snr=None, return_data=False):

        alpha = 1
        # TODO need to support multi-detectors
        det = 'H1'
        if use_which_design_ASD_for_whiten is not None:
            # # designed PSDs
            data = np.loadtxt(self.addr_asds[2])
            ASDf, ASD = data[:, 0], data[:, 1]
        else:
            # # current on-source PSDs
            ASDf, ASD = None, None

        # whiten the background
        noise, (fasd, asd) = self.whiten(self.wfd.dets[det].ifo.strain_data.time_domain_strain,
                                         self.sampling_frequency, ASDf, ASD, return_asd=True)
        signal = self.whiten(self.wfd.time_waveform_response[0],  # H1 # TODO
                             self.sampling_frequency, fasd, asd)

        if target_optimal_snr is not None:
            optimal_snr, _ = self.calc_SNR(alpha, asd)
            alpha = target_optimal_snr / optimal_snr
            optimal_snr_, matched_filter_snr = self.calc_SNR(alpha, asd)
            assert np.allclose(target_optimal_snr, optimal_snr_), \
                f'target_optimal_snr:{target_optimal_snr} vs optimal_snr_:{optimal_snr_}'
            if return_data:
                data = signal * alpha + noise
                return data, signal * alpha, noise, target_optimal_snr, matched_filter_snr
            return signal * alpha, noise, target_optimal_snr, matched_filter_snr
        else:
            optimal_snr, matched_filter_snr = self.calc_SNR(alpha, asd)
            if return_data:
                data = signal * alpha + noise
                return data, signal * alpha, noise, optimal_snr, matched_filter_snr
            return signal * alpha, noise, optimal_snr, matched_filter_snr

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

    def signal_preprocessing(self, signal, dMin, dMax):
        # TODO need to support multi-detectors
        if self.use_which_design_ASD_for_whiten is not None:
            # # designed PSDs
            data = np.loadtxt(self.addr_asds[2])
            ASDf, ASD = data[:, 0], data[:, 1]
        else:
            # # current on-source PSDs
            ASDf, ASD = None, None

        # [1, self.num_duration_long]
        signal_whitened = self.whiten(signal, self.sampling_frequency, ASDf, ASD)
        return self.normfunc.transform_signalornoise(self.patching(self.cut_from_long(signal_whitened)[np.newaxis, ...]),
                                                     dMin, dMax)

    def strain_postprocessing(self, strain, dMin, dMax):
        return self.normfunc.inverse_transform_data(strain, dMin, dMax)

    def signal_postprocessing(self, signal, dMin, dMax):
        return self.normfunc.inverse_transform_signalornoise(signal, dMin, dMax)

    def _noise(self):
        # TODO need to support multi-detectors
        det = 'H1'
        if self.use_which_design_ASD_for_whiten is not None:
            # # designed PSDs
            data = np.loadtxt(self.addr_asds[2])
            ASDf, ASD = data[:, 0], data[:, 1]
        else:
            # # current on-source PSDs
            ASDf, ASD = None, None

        # whiten the background
        noise, (fasd, asd) = self.whiten(self.wfd.dets[det].ifo.strain_data.time_domain_strain,
                                         self.sampling_frequency, ASDf, ASD, return_asd=True)
        return noise, fasd, asd

    def _signal(self, fasd, asd):
        alpha = 1
        signal = self.whiten(self.wfd.time_waveform_response[0],  # H1 # TODO
                             self.sampling_frequency, fasd, asd)

        if self.target_optimal_snr is not None:
            optimal_snr, _ = self.calc_SNR(alpha, asd)
            alpha = self.target_optimal_snr / optimal_snr
            _, matched_filter_snr = self.calc_SNR(alpha, asd)
            self.wfd.parameters.update(dict(
                optimal_snr=np.asarray(self.target_optimal_snr),
                matched_filter_snr=np.asarray(matched_filter_snr),
            ))
        else:
            optimal_snr, matched_filter_snr = self.calc_SNR(alpha, asd)
            self.wfd.parameters.update(dict(
                optimal_snr=np.asarray(optimal_snr),
                matched_filter_snr=np.asarray(matched_filter_snr),
            ))

        return signal * alpha, self.wfd.parameters

    def onesample(self, level):
        self.update_waveforms()  # update waveforms and noises in level 1
        # self.update_backgrounds(level)
        (data, signal, noise,
         optimal_snr, matched_filter_snr) = self.generate_sample(self.use_which_design_ASD_for_whiten,
                                                                 self.target_optimal_snr, return_data=True)
        self.wfd.parameters.update(dict(
            optimal_snr=np.asarray(optimal_snr),
            matched_filter_snr=np.asarray(matched_filter_snr),
        ))
        return (self.patching(self.cut_from_long(data)[np.newaxis, ...]),
                self.patching(self.cut_from_long(signal)[np.newaxis, ...]),
                self.patching(self.cut_from_long(noise)[np.newaxis, ...]),
                self.wfd.parameters,
                )  # (1, 31, 2048)
        # return data[np.newaxis,:23] # (1, 131072)
        # return np.concatenate([data[np.newaxis, ...]]*2, axis=0)[np.newaxis, ...]

    def longonesample(self, level):
        self.update_waveforms()  # update waveforms and noises in level 1
        self.update_backgrounds(level)
        (signal, noise,
         optimal_snr, matched_filter_snr) = self.generate_sample(self.use_which_design_ASD_for_whiten,
                                                                 self.target_optimal_snr)
        self.wfd.parameters.update(dict(
            optimal_snr=np.asarray(optimal_snr),
            matched_filter_snr=np.asarray(matched_filter_snr),
        ))
        return (signal[np.newaxis, ...],  # (1, 32*4096)
                noise[np.newaxis, ...],
                self.wfd.parameters,
                )

    def onedozen(self, num=2, level=1):
        # start = time.time()
        result_ids = [ray.put(self.onesample(level)) for _ in tqdm(range(num), disable=True)]

        data = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        signal = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        noise = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        params = {key: np.empty(shape=0, dtype=value.dtype) for key, value in self.wfd.parameters.items()}
        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            data = np.concatenate([data, ray.get(done_id[0])[0]])
            signal = np.concatenate([signal, ray.get(done_id[0])[1]])
            noise = np.concatenate([noise, ray.get(done_id[0])[2]])
            params = {key: np.append(params[key], value) for key, value in ray.get(done_id[0])[3].items()}

        # print(time.time() - start)
        # print(result.shape)
        return data, signal, noise, params

    def to_torch(self, pipe, batch_size, batch_format='native', prefetch_blocks=0, drop_last=False):
        """
        Returns:
            A torch IterableDataset.
        Ref:
            https://docs.ray.io/en/latest/_modules/ray/data/dataset.html#Dataset.to_torch
        """
        def make_generator():
            for batch in pipe.iter_batches(
                    batch_size=batch_size,
                    batch_format=batch_format,
                    prefetch_blocks=prefetch_blocks,
                    drop_last=drop_last):
                # print([(i, data.shape, signal.shape, noise.shape, params['mass_ratio'].shape) for i, (data, signal, noise, params) in enumerate(batch)])
                data_cache = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
                signal_cache = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
                noise_cache = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
                params_cache = {key: np.empty(shape=0, dtype=value.dtype) for key, value in self.wfd.parameters.items()}
                for data, signal, noise, params in batch:
                    data_cache = np.concatenate([data_cache, data])
                    signal_cache = np.concatenate([signal_cache, signal])
                    noise_cache = np.concatenate([noise_cache, noise])
                    params_cache = {key: np.append(params_cache[key], value) for key, value in params.items()}
                yield data_cache, signal_cache, noise_cache, params_cache
                # features_tensor = torch.cat([torch.as_tensor(features_tensor) for features_tensor in batch], dim=0)
                # yield (features_tensor)
        return TorchIterableDataset(make_generator)

    def pipeline(self, num_range, num_repeat, batch_size, level=1, datasets=None, test=None):
        if datasets is not None:
            ds = ray.data.range(num_range).map(lambda x: self.onedozen_datasets(datasets, level))
        elif test:
            ds = ray.data.range(num_range)
            # ds = ds.map(lambda x: self.pipe_update_waveforms(x))
            ds = ds.map(lambda x: self.pipe_noise())
            ds = ds.map(lambda x: self.pipe_signal(*x))
            ds = ds.map(lambda x: self.pipe_patching_cut_from_long(*x))
        else:
            ds = ray.data.range(num_range).map(lambda x: self.onesample(level))
        # pipe = ds.window(blocks_per_window=3).repeat(5)#.random_shuffle_each_window()
        # pipe = ds.window(blocks_per_window=2).repeat(num_repeat).foreach_window(lambda w: w.random_shuffle())  # .random_shuffle_each_window()
        pipe = ds.repeat(num_repeat)  #.foreach_window(lambda x: )  # .random_shuffle_each_window()
        return self.to_torch(pipe, batch_size=batch_size)

    def pipe_update_waveforms(self, x):
        self.update_waveforms()  # update waveforms and noises in level 1
        return x

    def pipe_noise(self):
        # self.update_backgrounds(level)
        return self._noise()

    def pipe_signal(self, noise, fasd, asd):
        signal, params = self._signal(fasd, asd)
        return signal + noise, signal, noise, params

    def pipe_noise_signal(self):
        return self.pipe_signal(*self.pipe_noise())

    def pipe_patching_cut_from_long(self, data, signal, noise, params):
        return (self.patching(self.cut_from_long(data)[np.newaxis, ...]),
                self.patching(self.cut_from_long(signal)[np.newaxis, ...]),
                self.patching(self.cut_from_long(noise)[np.newaxis, ...]),
                params
                )  # (1, 31, 2048)


@ray.remote
class RayDatasetTorch():
    """
    Dataset for Ray and Torch
    """
    def __init__(self, num_dataset):
        """Instantiate a ...
        """
        self.datasets = [RayDataset.remote() for _ in range(num_dataset)]
        self.token_shape = ray.get(self.datasets[0].show_token_shape.remote())
        self.num_duration_long = ray.get(self.datasets[0].show_num_duration_long.remote())
        self.num_duration = ray.get(self.datasets[0].show_num_duration.remote())
        self.patching = ray.get(self.datasets[0].show_patching_func.remote())
        # self.cut_from_long = self.datasets[0].cut_from_long
        self.sampling_frequency = ray.get(self.datasets[0].show_sampling_frequency.remote())
        geocent_time = ray.get(self.datasets[0].show_geocent_time.remote())
        self.geocent_time_shuffle = geocent_time[1] - geocent_time[0]

        # Set for standardizing strains
        feature_range = (-1, 1)
        self.normfunc = Normalize_strain(feature_range)

    def cut_from_long(self, data):
        left_index = int((self.num_duration_long - self.num_duration) / 2)
        right_index = int((self.num_duration_long + self.num_duration) / 2)
        return data[..., left_index: right_index]

    def onedozen_datasets(self, level=1):
        # start = time.time()
        result_ids = [dataset.onesample.remote(level) for dataset in tqdm(self.datasets, disable=True)]

        data = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        signal = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        noise = np.empty(shape=(0, *self.token_shape[1:]), dtype=np.float64)
        params = {key: np.empty(shape=0, dtype=value.dtype) for key, value in ray.get(self.datasets[0].show_parameters.remote()).items()}
        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            data = np.concatenate([data, ray.get(done_id[0])[0]])
            signal = np.concatenate([signal, ray.get(done_id[0])[1]])
            noise = np.concatenate([noise, ray.get(done_id[0])[2]])
            params = {key: np.append(params[key], value) for key, value in ray.get(done_id[0])[3].items()}

        # print(time.time() - start)
        # print(result.shape)
        return data, signal, noise, params

    def onedozen_longdatasets(self, level=1):
        # start = time.time()
        result_ids = [dataset.longonesample.remote(level) for dataset in tqdm(self.datasets, disable=True)]

        # data = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
        signal = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
        noise = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
        params = {key: np.empty(shape=0, dtype=value.dtype) for key, value in ray.get(self.datasets[0].show_parameters.remote()).items()}
        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            # data = np.concatenate([data, ray.get(done_id[0])[0]])
            signal = np.concatenate([signal, ray.get(done_id[0])[0]])
            noise = np.concatenate([noise, ray.get(done_id[0])[1]])
            params = {key: np.append(params[key], value) for key, value in ray.get(done_id[0])[2].items()}

        # print(time.time() - start)
        # print(result.shape)
        return (#data,
                signal, noise, params)

    def shulffle_geocent_time(self, x):
        # data, signal, noise, params = x
        signal, noise, params = x

        len_time = np.random.uniform(0, self.geocent_time_shuffle)
        len_roll = int(len_time * self.sampling_frequency)
        signal = np.pad(signal[..., len_roll:], pad_width=((0, 0),)*(signal.ndim-1) + ((0, len_roll),), constant_values=0)
        params['geocent_time'] = params['geocent_time'] - len_time
        # return data, signal, noise, params
        return signal, noise, params

    def pipeline(self, num_range, num_repeat, batch_size, level=1):
        assert num_repeat > 0
        ds = ray.data.range(num_range).map(lambda x: self.onedozen_longdatasets(level))
        # ds = ray.data.range(num_range).map(lambda x: self.onedozen_datasets(level))
        # elif test:
        #     ds = ray.data.range(num_range)
        #     # ds = ds.map(lambda x: self.pipe_update_waveforms(x))
        #     ds = ds.map(lambda x: self.pipe_noise())
        #     ds = ds.map(lambda x: self.pipe_signal(*x))
        #     ds = ds.map(lambda x: self.pipe_patching_cut_from_long(*x))
        # else:
        #     ds = ray.data.range(num_range).map(lambda x: self.onesample(level))
        # pipe = ds.window(blocks_per_window=3).repeat(5)#.random_shuffle_each_window()
        # pipe = ds.window(blocks_per_window=2).repeat(num_repeat).foreach_window(lambda w: w.random_shuffle())  # .random_shuffle_each_window()

        pipe = ds.repeat(num_repeat).foreach_window(lambda x: x.map(self.shulffle_geocent_time))  # .random_shuffle_each_window()
        return self.to_torch(pipe, batch_size=batch_size)

    def to_torch(self, pipe, batch_size, batch_format='native', prefetch_blocks=0, drop_last=False):
        """
        Returns:
            A torch IterableDataset.
        Ref:
            https://docs.ray.io/en/latest/_modules/ray/data/dataset.html#Dataset.to_torch
        """
        def make_generator():
            for batch in pipe.iter_batches(
                    batch_size=batch_size,
                    batch_format=batch_format,
                    prefetch_blocks=prefetch_blocks,
                    drop_last=drop_last):
                # print([(i, data.shape, signal.shape, noise.shape, params['mass_ratio'].shape) for i, (data, signal, noise, params) in enumerate(batch)])
                # data_cache = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
                signal_cache = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
                noise_cache = np.empty(shape=(0, self.num_duration_long), dtype=np.float64)
                params_cache = {key: np.empty(shape=0, dtype=value.dtype) for key, value in ray.get(self.datasets[0].show_parameters.remote()).items()}
                # for data, signal, noise, params in batch:
                for signal, noise, params in batch:
                    # data_cache = np.concatenate([data_cache, data])
                    signal_cache = np.concatenate([signal_cache, signal])
                    noise_cache = np.concatenate([noise_cache, noise])
                    params_cache = {key: np.append(params_cache[key], value) for key, value in params.items()}
                # # Apply the patching -> standardizing
                data_cache = signal_cache + noise_cache
                data_cache, dMin, dMax = self.normfunc.transform_data(self.patching(self.cut_from_long(data_cache)))
                signal_cache = self.normfunc.transform_signalornoise(self.patching(self.cut_from_long(signal_cache)), dMin, dMax)
                noise_cache = self.normfunc.transform_signalornoise(self.patching(self.cut_from_long(noise_cache)), dMin, dMax)
                params_cache['dMax'] = dMax[..., 0, 0]
                params_cache['dMin'] = dMin[..., 0, 0]
                # yield data_cache, signal_cache, noise_cache, params_cache
                yield data_cache, signal_cache, pd.DataFrame(params_cache).values
                # features_tensor = torch.cat([torch.as_tensor(features_tensor) for features_tensor in batch], dim=0)
                # yield (features_tensor)
        return TorchIterableDataset(make_generator)


if __name__ == "__main__":
    import sys
    sys.path.append('/workspace/zhouy/megatron-gw/GWToolkit/')
    from gwtoolkit.gw.ray import RayDatasetTorch, ray
    import time

    def update_level(i):
        if i%100==4:
            return 4    
        elif i%50==0:
            return 3
        elif i%10==0:
            return 2
        else:
            return 1

    batch_size = 128
    num_dataset = 32 if batch_size >= 32 else batch_size
    num_range = batch_size//num_dataset
    num_repeat = 50

    datasets = RayDatasetTorch.remote(num_dataset=num_dataset)

    index = 0
    while True:
        index += 1
        level = update_level(index)
        pipeline = datasets.pipeline.remote(num_range, num_repeat, batch_size, level=level)
        start = time.time()
        for i, (data, signal, noise, params) in enumerate(ray.get(pipeline)):
            end = time.time()
            print(f'batch={i}, time: {end-start:.4f}sec', data.shape, signal.shape, noise.shape, params['geocent_time'][:2].tolist())
            start = end
