import sys
import os 
sys.path.append(os.getcwd())
import pytest
from gwtoolkit.gw.detector import Detector
from gwtoolkit.gw.source import Source
import numpy


param = dict(
    mass_1=36., mass_2=29.,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)


@pytest.fixture
def init_detector():
    det = Detector(name='H1', sampling_frequency=4096, duration=1, start_time=0)
    assert det._show_detectors() == print(det.ifo)
    return det


@pytest.fixture
def init_source():
    return Source(
        base='bilby', conversion='BBH', sampling_frequency=8192, duration=8
    )


def test_property(init_detector, init_source):
    assert len(init_detector.time_domain_strain) == 4096
    assert len(init_detector.frequency_domain_strain) == 2049
    assert len(init_detector.frequency_domain_whitened_strain) == 2049
    assert len(init_detector.time_domain_whitened_strain) == 4096
    assert init_detector.ifo.strain_data.start_time == 0
    assert numpy.allclose(init_detector.frequency_array,
                          numpy.linspace(0.0, init_detector.sampling_frequency / 2.0,
                                         num=2049, endpoint=True,
                                         dtype=numpy.float32))
    assert numpy.allclose(init_detector.time_array,
                          numpy.linspace(init_detector.start_time, init_detector.start_time+1-1/4096,
                                         num=4096, endpoint=True,
                                         dtype=numpy.float32))

    init_detector.update(duration=8, sampling_frequency=4096*2, start_time=100)
    assert len(init_detector.time_domain_strain) == 65536
    assert len(init_detector.frequency_domain_strain) == 32769
    assert len(init_detector.frequency_domain_whitened_strain) == 32769
    assert len(init_detector.time_domain_whitened_strain) == 65536
    assert init_detector.ifo.strain_data.start_time == 100

    assert numpy.allclose(init_detector.power_spectral_density_array ** 2,
                          init_detector.amplitude_spectral_density_array)

    psd_file = './tests/gw/test_AdV_psd.txt'
    asd_file = './tests/gw/test_AdV_asd.txt'
    init_detector.load_asd_or_psd_file(psd_file=psd_file)
    init_detector.load_asd_or_psd_file(asd_file=asd_file)
    # assert len(wfg.frequency_domain_strain()[key]) == init_source.num_f

    freq_strain, freq = init_detector.time_to_frequency_domain(init_detector.time_domain_strain)
    assert numpy.allclose(freq_strain, init_detector.frequency_domain_strain, atol=1e-22)
    assert numpy.allclose(freq, init_detector.ifo.frequency_array)
    tim_strain, tim = init_detector.frequency_to_time_domain(init_detector.frequency_domain_strain)
    assert numpy.allclose(tim_strain, init_detector.time_domain_strain)
    assert numpy.allclose(tim, init_detector.ifo.time_array)

    init_detector.update(duration=8, sampling_frequency=8192, start_time=1126259642.413-6)
    wfg = init_source.waveform_generator(param)
    detector_response = init_detector.get_detector_response(wfg.frequency_domain_strain(),
                                                            dict(psi=2.659,
                                                                 geocent_time=1126259642.413,
                                                                 ra=1.375,
                                                                 dec=-1.2108))
    assert len(detector_response) == len(init_detector.ifo.frequency_array) == len(wfg.frequency_array)

    for _ in range(5):
        init_detector.update()
        init_detector.matched_filter_snr(detector_response, init_detector.frequency_domain_strain) \
            == init_detector.inner_product(detector_response) / init_detector.optimal_snr(detector_response)

    assert numpy.allclose(init_detector.overlap(signal_a=detector_response, signal_b=detector_response), 1)
    assert init_detector.overlap(signal_a=detector_response, signal_b=init_detector.frequency_domain_strain) < 0.1
