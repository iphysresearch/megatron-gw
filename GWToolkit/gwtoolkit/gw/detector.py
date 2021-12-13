"""
Detector
"""
import numpy
import bilby
from bilby.core import utils
from bilby.gw import utils as gwutils
from scipy.signal.windows import tukey
from .utils import overlap
from .gwosc_cvmfs import GWOSC


class Detector():
    """
    Detector
    """
    def __init__(self, name, sampling_frequency, duration, start_time=0):
        """Instantiate a InterferometerList

        The InterferometerList is a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Parameters
        ==========
        interferometers: iterable
            The list of interferometers
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float (default: 0)
            The GPS start-time of the data
        """
        self.ifo = None
        self.gwosc = None

        self.roll_off = 0.2
        self.start_time = start_time
        # Set up interferometers.  In this case we'll use two interferometers
        # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
        # sensitivity
        self.ifo = bilby.gw.detector.get_empty_interferometer(name)
        self.update(sampling_frequency, duration, start_time)

    def load_from_GWOSC(self, data_dir, noise_interval, selected_hdf_file_ratio, num_length=1):
        self.gwosc = GWOSC(self.ifo.name, data_dir, self.sampling_frequency, noise_interval, selected_hdf_file_ratio,
                           num_length=num_length)

    def update_psd_from_GWOSC(self, seg_sec, **kwds):
        self.gwosc.update_strain()
        freq, Pxx = self.gwosc.estimate_psd_cvmfs(seg_sec, **kwds)

        # It will cover the PSD and generate new noises
        self.ifo.power_spectral_density = self.ifo.power_spectral_density.from_power_spectral_density_array(freq, Pxx)
        self.update()

    def update_time_domain_strain_from_GWOSC(self, seg_sec, **kwds):
        self.update_psd_from_GWOSC(seg_sec, **kwds)

        sampling_factor = int(self.gwosc.original_sampling_rate / self.sampling_frequency)
        # It will cover the time_domain_strain with GPS time_array
        self.ifo.strain_data.set_from_time_domain_strain(self.gwosc.strain[::sampling_factor],
                                                         self.sampling_frequency,
                                                         # self.gwosc.noise_interval,
                                                         # start_time=self.gwosc.start,
                                                         time_array=self.gwosc.time[::sampling_factor])
        # Note that `self.update()` will recover the above setting

    def load_asd_or_psd_file(self, asd_file=None, psd_file=None):
        """
        asd_file: str
            Name of the ASD file
        psd_file: str
            Name of the PSD file
        """
        if asd_file:
            self.ifo.power_spectral_density.asd_file = asd_file
        elif psd_file:
            self.ifo.power_spectral_density.psd_file = psd_file
        self.update()

    @property
    def frequency_domain_strain(self):
        """ Returns the frequency domain strain
        This is the frequency domain strain normalised to units of
        strain / Hz, obtained by a one-sided Fourier transform of the
        time domain data, divided by the sampling frequency.
        """
        return self.ifo.strain_data.frequency_domain_strain

    @property
    def time_domain_strain(self):
        """ The time domain strain, in units of strain
        (Based on self.ifo.frequency_domain_strain and self.sampling_frequency)
        """
        return utils.infft(self.ifo.frequency_domain_strain, self.sampling_frequency)

    @property
    def frequency_domain_whitened_strain(self):
        """ Calculates the whitened data by dividing data by the amplitude spectral density
        Returns
        =======
        array_like: The whitened data
        Details: self.ifo.strain_data.frequency_domain_strain / self.ifo.amplitude_spectral_density_array
        """
        return self.ifo.whitened_frequency_domain_strain

    @property
    def time_domain_whitened_strain(self):
        """ Calculates the whitened data by dividing data by the amplitude spectral density
        Returns
        =======
        array_like: The whitened data
        Details: self.ifo.strain_data.frequency_domain_strain / self.ifo.amplitude_spectral_density_array
        """
        return self.frequency_to_time_domain(self.ifo.whitened_frequency_domain_strain)[0]

    @property
    def amplitude_spectral_density_array(self):
        """ Returns the amplitude spectral density (ASD) given we know a power spectral density (PSD)
        Returns
        =======
        array_like: An array representation of the ASD
        """
        return self.ifo.amplitude_spectral_density_array

    @property
    def power_spectral_density_array(self):
        """ Returns the power spectral density (PSD)
        This accounts for whether the data in the interferometer has been windowed.
        Returns
        =======
        array_like: An array representation of the PSD
        """
        return self.ifo.power_spectral_density_array

    def whiten(self, frequency_domain_strain):
        """ Calculates the whitened data by dividing frequency strain by the amplitude spectral density
        Returns
        =======
        array_like: The whitened data
        """
        return frequency_domain_strain / self.amplitude_spectral_density_array

    def time_to_frequency_domain(self, time_domain_strain):
        """From frequency domain to time domain w.r.t self.sampling_frequency
        Return (frequency_domain_strain, frequency_array)
        """
        window = self.time_domain_window(len(time_domain_strain))
        _frequency_domain_strain, frequency_array = utils.nfft(
            time_domain_strain * window, self.sampling_frequency)
        return _frequency_domain_strain * self.ifo.strain_data.frequency_mask, frequency_array

    def frequency_to_time_domain(self, frequency_domain_strain):
        """From frequency domain to time domain w.r.t self.sampling_frequency
        Return (time_domain_strain, time_array)
        """
        return utils.infft(frequency_domain_strain, self.sampling_frequency), self.ifo.strain_data.time_array

    @property
    def frequency_array(self):
        """ Frequency array for the waveforms. Automatically updates if sampling_frequency or duration are updated.
        Returns
        =======
        array_like: The frequency array
        """
        return self.ifo.strain_data.frequency_array

    @property
    def time_array(self):
        """ Time array for the waveforms. Automatically updates if sampling_frequency or duration are updated.
        Returns
        =======
        array_like: The time array
        """
        return self.ifo.strain_data.time_array

    def update(self, sampling_frequency=None, duration=None, start_time=None):
        """Set the `Interferometer.strain_data` from a power spectal density

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float (default: 0)
            The GPS start-time of the data
        """
        if sampling_frequency:
            self.sampling_frequency = sampling_frequency
        if duration:
            self.duration = duration
        if start_time:
            self.start_time = start_time
        self.ifo.set_strain_data_from_power_spectral_density(self.sampling_frequency,
                                                             self.duration,
                                                             self.start_time)
        # Below, try to aviod protected-access:
        # self.ifo.strain_data._frequency_mask_updated = False
        self.ifo.minimum_frequency = self.ifo.strain_data.minimum_frequency

    def get_detector_response(self, waveform_polarizations, parameters):
        """Get the detector response for a particular waveform

        Parameters
        ==========
        waveform_polarizations: dict
            polarizations of the waveform ('plus'/'cross')
        parameters: dict
            parameters describing position and time of arrival of the signal

        Returns (frequency domain)
        =======
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """
        for key in ['ra', 'dec', 'geocent_time', 'psi']:
            assert key in parameters.keys(), print('No {} in your parameters dict'.format(key))
        assert len(waveform_polarizations['plus']) == len(waveform_polarizations['cross'])\
                                                   == len(self.ifo.strain_data.frequency_mask)
        return self.ifo.get_detector_response(waveform_polarizations, parameters)

    def _show_detectors(self):
        """Print the details of detectors and asd/psd."""
        print(',\n'.join(str(self.ifo).split(', ')))

    @property
    def alpha(self):
        """Parameter to pass to tukey window, how much of segment falls into windowed part."""
        return 2 * self.roll_off / self.duration

    def time_domain_window(self, length, roll_off=None, alpha=None):
        """
        Window function to apply to time domain data before FFTing.
        This defines self.window_factor as the power loss due to the windowing.
        See https://dcc.ligo.org/DocDB/0027/T040089/000/T040089-00.pdf
        Parameters
        ==========
        roll_off: float
            Rise time of window in seconds
        alpha: float
            Parameter to pass to tukey window, how much of segment falls
            into windowed part
        Returns
        ======
        window: array
            Window function over time array
        """
        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = alpha * self.duration / 2
        # self.window_factor = np.mean(window ** 2)
        return tukey(length, alpha=self.alpha)

    def optimal_snr(self, signal):
        """
        Parameters
        ==========
        signal: array_like
            Array containing the signal

        Returns
        =======
        float: The optimal signal to noise ratio possible squared
        w.r.t self.ifo.power_spectral_density_array[self.ifo.strain_data.frequency_mask]

        More details:
        https://github.com/lscsoft/bilby/blob/9daf210aad06ecec3ee592a7f6138095f67ac074/bilby/gw/detector/interferometer.py#L417
        """
        return numpy.sqrt(self.ifo.optimal_snr_squared(signal).real)

    def matched_filter_snr(self, signal, data):
        """
        Parameters
        ==========
        signal: array_like
            Array containing the signal
        data: array_like, signal + frequency_colored_noise

        Returns
        =======
        float: The matched filter signal to noise ratio squared
        w.r.t self.ifo.power_spectral_density_array[self.ifo.strain_data.frequency_mask]
            and current self.ifo.strain_data.frequency_domain_strain[self.ifo.strain_data.frequency_mask]
        So you can `self.update()` to refresh it.

        More details:
        https://github.com/lscsoft/bilby/blob/9daf210aad06ecec3ee592a7f6138095f67ac074/bilby/gw/detector/interferometer.py#L534
        """
        return gwutils.matched_filter_snr(
            signal=signal[self.ifo.strain_data.frequency_mask],
            frequency_domain_strain=data[self.ifo.strain_data.frequency_mask],
            power_spectral_density=self.ifo.power_spectral_density_array[self.ifo.strain_data.frequency_mask],
            duration=self.ifo.strain_data.duration)

    def inner_product(self, signal):
        """
        Parameters
        ==========
        signal: array_like
            Array containing the signal
        Returns
        =======
        float: The optimal signal to noise ratio possible squared
        w.r.t self.ifo.power_spectral_density_array[self.ifo.strain_data.frequency_mask]
            and current self.ifo.strain_data.frequency_domain_strain[self.ifo.strain_data.frequency_mask]
        So you can `self.update()` to refresh it.

        More details:
        https://github.com/lscsoft/bilby/blob/9daf210aad06ecec3ee592a7f6138095f67ac074/bilby/gw/detector/interferometer.py#L516
        """
        return self.ifo.inner_product(signal)

    def overlap(self, signal_a, signal_b):
        """Match/Overpal between signal_a and signal_b
        w.r.t self.ifo.power_spectral_density_array
              self.ifo.minimum_frequency
              self.ifo.maximum_frequency
              self.duration
        More general overlop function for `gwtoolkit.gw.utils.overlap`
        """
        return overlap(signal_ab=(signal_a, signal_b),
                       power_spectral_density=self.ifo.power_spectral_density_array,
                       delta_frequency=1/self.duration,
                       lower_cut_off=self.ifo.minimum_frequency,
                       upper_cut_off=self.ifo.maximum_frequency)
