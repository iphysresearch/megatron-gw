"""
Raw data and data processing from GWOSC
"""
import numpy
from .readligo import FileList, getstrain_cvmfs
from .hdffiles import NoiseTimeline
import scipy.signal


class GWOSC():
    """
    GWOSC
    """
    def __init__(self, ifo, data_dir, sampling_frequency, noise_interval, selected_hdf_file_ratio,
                 dq_bits=(0, 1, 2, 3), inj_bits=(0, 1, 2, 4), num_length=1):
        """Instantiate a InterferometerList

        noise_interval [sec] : used for estimating PSD
        dq_bits:
            :7 bits in the data quality bitmask:
                0 : 1
                1 : 3
                2 : 7
                3 : 15
                4 : 31
                5 : 63
                6 : 127
        inj_bits:
            :5 bits in the injection bitmask:
                0 : 1
                1 : 3
                2 : 7
                3 : 15
                4 : 31
        """
        random_seed = 42

        self.dq_bits = dq_bits
        self.inj_bits = inj_bits
        self.noise_interval = noise_interval
        self.num_length = num_length

        self.ifo = ifo  # 'H1'  # TODO 支持多 dets
        self.sampling_frequency = sampling_frequency

        self.start = 0  # a start-GPS by sampling
        self.start_list = []  # a list of start-GPS by sampling
        self.strain = None
        self.strain_multi = numpy.empty((num_length, noise_interval * 16384))  # TODO 16384
        self.time = None
        self.time_multi = numpy.empty((num_length, noise_interval * 16384))  # TODO 16384

        # Used for scanning the raw data # TODO 下面的类应该提炼成自己的代码
        self.noise_timeline = NoiseTimeline(background_data_directory=data_dir,
                                            selected_hdf_file_ratio=selected_hdf_file_ratio,
                                            random_seed=random_seed)
        # Used for load strain
        self.filelist = FileList(directory=data_dir)

    def update_strain(self, ):
        """Pick a valid samples from streaming data
        start : [GPS]

        FYI:
            * self.noise_timeline.sampleing:
                524 µs ± 7.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            * getstrain_cvmfs:
                CPU times: user 6.8 s, sys: 1.91 s, total: 8.71 s Wall time: 8.71 s
        """
        self.start = self.noise_timeline.sampleing(self.noise_interval, self.dq_bits, self.inj_bits, 1)[0]
        self.strain, self.time = getstrain_cvmfs(self.start, self.start+self.noise_interval,
                                                 self.ifo, self.filelist, inj_dq_cache=0)

    def update_randomly_strain(self, ):
        """Pick a valid samples from streaming data
        start : [GPS]

        FYI:
            * self.noise_timeline.sampleing:
                524 µs ± 7.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            * getstrain_cvmfs:
                CPU times: user 6.8 s, sys: 1.91 s, total: 8.71 s Wall time: 8.71 s
        """
        self.start_list = self.noise_timeline.sampleing(self.noise_interval, self.dq_bits, self.inj_bits, self.num_length)
        for i in range(self.num_length):
            self.strain_multi[i], self.time_multi[i] = getstrain_cvmfs(self.start_list[i], self.start_list[i]+self.noise_interval,
                                                                       self.ifo, self.filelist, inj_dq_cache=0)

    @property
    def original_sampling_rate(self):
        try:
            return 1/(self.time[1] - self.time[0])
        except:
            return 1/(self.time_multi[0][1] - self.time_multi[0][0])

    def estimate_psd_cvmfs(self, seg_sec, **kwds):
        """Calculate the power spectral density of a time serie randomly from valid raw stream data.

        Parameters
        ----------
        kwds: keywords (default)
        Additional keyword arguments are passed on to the `scipy.signal.welch` method.
            window : str or tuple or array_like, optional
                Desired window to use. If `window` is a string or tuple, it is
                passed to `get_window` to generate the window values, which are
                DFT-even by default. See `get_window` for a list of windows and
                required parameters. If `window` is array_like it will be used
                directly as the window and its length must be nperseg. Defaults
                to a Hann window.
            nperseg : int, optional
                Length of each segment. Defaults to None, but if window is str or
                tuple, is set to 256, and if window is array_like, is set to the
                length of the window.
            noverlap : int, optional
                Number of points to overlap between segments. If `None`,
                ``noverlap = nperseg // 2``. Defaults to `None`.
            nfft : int, optional
                Length of the FFT used, if a zero padded FFT is desired. If
                `None`, the FFT length is `nperseg`. Defaults to `None`.
            detrend : str or function or `False`, optional
                Specifies how to detrend each segment. If `detrend` is a
                string, it is passed as the `type` argument to the `detrend`
                function. If it is a function, it takes a segment and returns a
                detrended segment. If `detrend` is `False`, no detrending is
                done. Defaults to 'constant'.
            return_onesided : bool, optional
                If `True`, return a one-sided spectrum for real data. If
                `False` return a two-sided spectrum. Defaults to `True`, but for
                complex data, a two-sided spectrum is always returned.
            scaling : { 'density', 'spectrum' }, optional
                Selects between computing the power spectral density ('density')
                where `Pxx` has units of V**2/Hz and computing the power
                spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
                is measured in V and `fs` is measured in Hz. Defaults to
                'density'
            axis : int, optional
                Axis along which the periodogram is computed; the default is
                over the last axis (i.e. ``axis=-1``).
            average : { 'mean', 'median' }, optional
                Method to use when averaging periodograms. Defaults to 'mean'.
            random_timedomain_ratio: None or (0,1), optional

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        Pxx : ndarray
            Power spectral density or power spectrum of x.
        """
        # Estimate the PSD based the original sampling rate of raw data
        freq, Pxx = scipy.signal.welch(self.strain, fs=self.original_sampling_rate,
                                       nperseg=seg_sec*self.original_sampling_rate, **kwds)

        # if is_interpolate:  # TODO maybe droped and insert it to whiten
        #     f = scipy.interpolate.interp1d(freq, Pxx)
        #     freq = numpy.fft.rfftfreq(noise_interval * self.target_sampling_rate,
        #                               1.0/self.target_sampling_rate)
        #     return freq, f(freq)
        # else:
        return freq, Pxx

    def estimate_randomly_psd_cvmfs(self, seg_sec, random_timedomain_ratio, **kwds):
        # Estimate the PSD based the original sampling rate of raw data
        freq, Pxx = welch_modified(self.strain_multi, fs=self.original_sampling_rate,
                                   nperseg=seg_sec*self.original_sampling_rate,
                                   random_timedomain_ratio=random_timedomain_ratio,
                                   **kwds)
        return freq, Pxx


def csd_modified(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean', random_timedomain_ratio=None):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's method.
    """
    Pxy_list = []
    for xx, yy in zip(x, y):
        freqs, _, Pxy = scipy.signal.spectral._spectral_helper(xx, yy, fs, window, nperseg, noverlap, nfft,
                                                               detrend, return_onesided, scaling, axis,
                                                               mode='psd')

        Pxy_list.append(Pxy)
    Pxy = numpy.concatenate(Pxy_list, axis=-1)
    print(Pxy.shape)

    if random_timedomain_ratio:
        indexs = numpy.random.randint(Pxy.shape[-1], size=int(Pxy.shape[-1] * random_timedomain_ratio))
        Pxy = Pxy[:, indexs]
        print(Pxy.shape)

    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == 'median':
                # numpy.median must be passed real arrays for the desired result
                if numpy.iscomplexobj(Pxy):
                    Pxy = (numpy.median(numpy.real(Pxy), axis=-1)
                           + 1j * numpy.median(numpy.imag(Pxy), axis=-1))
                    Pxy /= _median_bias(Pxy.shape[-1])
                else:
                    Pxy = numpy.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == 'mean':
                Pxy = Pxy.mean(axis=-1)
            else:
                raise ValueError('average must be "median" or "mean", got %s'
                                 % (average,))
        else:
            Pxy = numpy.reshape(Pxy, Pxy.shape[:-1])
    return freqs, Pxy




def welch_modified(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
                   detrend='constant', return_onesided=True, scaling='density',
                   axis=-1, average='mean', random_timedomain_ratio=None):
    r"""
    Estimate power spectral density using Welch's method.
    """
    freqs, Pxx = csd_modified(x, x, fs=fs, window=window, nperseg=nperseg,
                              noverlap=noverlap, nfft=nfft, detrend=detrend,
                              return_onesided=return_onesided, scaling=scaling,
                              axis=axis, average=average,
                              random_timedomain_ratio=random_timedomain_ratio)

    return freqs, Pxx.real
    # @property
    # def time_strain(self):
    #     return self.time, self.strain

    # @property
    # def frequency_PSD(self, kwds):
    #     return self.estimate_psd_cvmfs(self, **kwds)


    # # Noise
    # @property
    # def frequency_colored_noise(self):
    #     """Generate a whitened noise w.r.t self.dets in frequency domain
    #     """
    #     strain = numpy.empty(shape=(len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
    #     for i, (_, det) in enumerate(self.dets.items()):
    #         strain[i, :] = det.frequency_domain_strain
    #     return strain

    # @property
    # def time_colored_noise(self):
    #     """Generate a colored noise w.r.t self.dets in time domain
    #     """
    #     strain = numpy.empty(shape=(len(self.dets.keys()), self.num_t), dtype=numpy.float64)
    #     for i, (_, det) in enumerate(self.dets.items()):
    #         strain[i, :] = det.time_domain_strain
    #     return strain

    # @property
    # def frequency_whitened_noise(self):
    #     """Generate a whitened noise w.r.t self.dets in frequency domain
    #     """
    #     strain = numpy.empty(shape=(len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
    #     for i, (_, det) in enumerate(self.dets.items()):
    #         strain[i, :] = det.frequency_domain_whitened_strain
    #     return strain

    # @property
    # def time_whitened_noise(self):
    #     """Generate a whitened noise w.r.t self.dets in time domain
    #     """
    #     strain = numpy.empty(shape=(len(self.dets.keys()), self.num_t), dtype=numpy.float64)
    #     for i, (_, det) in enumerate(self.dets.items()):
    #         strain[i] = det.time_domain_whitened_strain
    #     return strain


    # # Noise block (unwhittened / whittened)
    # def frequency_colored_noise_block(self, num, disable=True):
    #     """Generate a data block with colored detector noises in frequency domain
    #     num: int
    #         Number of data
    #     """
    #     block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
    #     for i in tqdm(range(num), disable=disable):
    #         self._update_noise()
    #         block[i] = self.frequency_colored_noise
    #     return block

    # def time_colored_noise_block(self, num, disable=True):
    #     """Generate a data block with colored detector noises in time domain
    #     num: int
    #         Number of data
    #     """
    #     block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
    #     for i in tqdm(range(num), disable=disable):
    #         self._update_noise()
    #         block[i] = self.time_colored_noise
    #     return block

    # def frequency_whitened_noise_block(self, num, disable=True):
    #     """Generate a data block with whitened detector noises in frequency domain
    #     num: int
    #         Number of data
    #     """
    #     block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
    #     for i in tqdm(range(num), disable=disable):
    #         self._update_noise()
    #         block[i] = self.frequency_whitened_noise
    #     return block

    # def time_whitened_noise_block(self, num, disable=True):
    #     """Generate a data block with whitened detector noises in time domain
    #     num: int
    #         Number of data
    #     """
    #     block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
    #     for i in tqdm(range(num), disable=disable):
    #         self._update_noise()
    #         block[i] = self.time_whitened_noise
    #     return block
    

class Glitch_Sampler(GWOSC):
    def __init__(self, glitch_dir = '../tests/gw/trainingset_v1d1_metadata.csv', signal_length=8):
        super().__init__(ifo='H1', data_dir='/workspace/zhaoty/dataset/O1_H1_All/',
                         sampling_frequency=4096, noise_interval=1024,
                 dq_bits=(0, 1, 2, 3), inj_bits=(0, 1, 2, 4))
        self.glitch_meta_data = pd.read_csv(glitch_dir)
        self.peak_time = self.glitch_meta_data.peak_time.astype('float64') + self.glitch_meta_data.peak_time_ns.astype('float64') * 1e-9
        self.start_time = self.glitch_meta_data.start_time.astype('float64') + self.glitch_meta_data.start_time_ns.astype('float64') * 1e-9
        self.duration = self.glitch_meta_data.duration
        self.signal_length = signal_length
    
    def get_start_time(self):
        idx = numpy.random.randint(0,high=len(self.glitch_meta_data))
        if self.duration[idx] > self.signal_length:
            return int(self.start_time[idx] - numpy.random.rand(1) * (self.duration[idx] - self.signal_length +0.5))
        else:
            return int(self.start_time[idx] + numpy.random.rand(1) * (self.signal_length - self.duration[idx] +0.5))
    
    def get_strain(self,):
        strain = None
        while strain is None:
            try:
                start_time = self.get_start_time()
                strain, time = getstrain_cvmfs(start_time, start_time+self.signal_length,
                                                 self.ifo, self.filelist, inj_dq_cache=0)
            except:
                pass
        sampling_factor = int(self.original_sampling_rate / self.sampling_frequency)
        strain = strain[::sampling_factor]
        return strain
