"""
base
"""
# import numpy


class Waveform():
    """Initialize a waveform signal.
    """
    def __init__(self, sampling_frequency, duration):
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        # self.start_time = 0.0
        self.frequency_array = None
        self.time_array = None

    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_frequency / 2.0

    @f_max.setter
    def f_max(self, f_max):
        """Set the sampling rate based on maximum frequency
        """
        self.sampling_frequency = 2.0 * f_max

    @property
    def delta_t(self):
        """Set the time revolution based on sampling rate
        """
        return 1.0 / self.sampling_frequency

    @delta_t.setter
    def delta_t(self, delta_t):
        """Set the sampling rate based on time revolution
        """
        self.sampling_frequency = 1.0 / delta_t

    @property
    def delta_f(self):
        """Set the frequency revolution based on time duration
        """
        return 1.0 / self.duration

    @delta_f.setter
    def delta_f(self, delta_f):
        """Set the time duration based on frequency revolution
        """
        self.duration = 1.0 / delta_f

    @property
    def num_t(self):
        """Set the number of time sampling points based on
           time duration and sampling rate
        """
        return int(self.duration * self.sampling_frequency)

    @property
    def num_f(self):
        """Set the number of frequency sampling points
           based on the maximum and frequency revolution
        """
        return int(self.f_max / self.delta_f) + 1

    # @property
    # def time_array(self):
    #     """Array of times at which waveforms are sampled (num_t)."""
    #     return numpy.linspace(self.start_time, self.start_time+self.duration,
    #                           num=self.num_t,
    #                           endpoint=False,
    #                           dtype=numpy.float32)

    # @property
    # def frequency_array(self):
    #     """Array of postive frequencies at which waveforms are sampled (num_f)."""
    #     return numpy.linspace(0.0, self.f_max,
    #                           num=self.num_f, endpoint=True,
    #                           dtype=numpy.float32)

    # @property
    # def sample_fft_frequencies(self):
    #     """
    # Return the Discrete Fourier Transform sample frequencies.

    # The returned float array `f` contains the frequency bin centers
    # in cycles per unit of the sample spacing (with zero at the start).
    # For instance, if the sample spacing is in seconds, then the
    # frequency unit is cycles/second.

    # Given a window length `num_t` and a sample spacing `delta_t`::

    # f = [0, 1, ...,   num_t/2-1,     -num_t/2, ..., -1] / (delta_t*num_t) if num_t is even
    # f = [0, 1, ..., (num_t-1)/2, -(num_t-1)/2, ..., -1] / (delta_t*num_t) if num_t is odd
    #     """
    #     return numpy.fft.fftfreq(self.num_t, self.delta_t).astype(numpy.float32)
