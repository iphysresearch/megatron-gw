"""
Utils
"""
import numpy


def overlap(signal_ab, power_spectral_density=None, delta_frequency=None,
            lower_cut_off=None, upper_cut_off=None):
    """Match/Overpal between (signal_a and signal_b)
    Learned from: https://github.com/lscsoft/bilby/blob/9daf210aad06ecec3ee592a7f6138095f67ac074/bilby/gw/utils.py#L270
    """
    (signal_a, signal_b) = signal_ab
    low_index = int(lower_cut_off / delta_frequency)
    up_index = int(upper_cut_off / delta_frequency)
    integrand = numpy.conj(signal_a) * signal_b
    integrand = integrand[low_index:up_index] / power_spectral_density[low_index:up_index]
    norm_a = normalize_strain(signal_a, power_spectral_density, delta_frequency, lower_cut_off, upper_cut_off)
    norm_b = normalize_strain(signal_b, power_spectral_density, delta_frequency, lower_cut_off, upper_cut_off)
    integral = (4 * delta_frequency * integrand) / norm_a / norm_b
    return sum(integral).real


def normalize_strain(
    signal, psd=None, delta_f=None, lower_cut_off=None, upper_cut_off=None
):
    """Normalizing a waveform

    Learned from:
    https://github.com/lscsoft/bilby/blob/9daf210aad06ecec3ee592a7f6138095f67ac074/test/integration/test_waveforms.py#L368
    """
    low_index = int(lower_cut_off / delta_f)
    up_index = int(upper_cut_off / delta_f)
    integrand = numpy.conj(signal) * signal
    integrand = integrand[low_index:up_index] / psd[low_index:up_index]
    integral = sum(4 * delta_f * integrand)
    return numpy.sqrt(integral).real
