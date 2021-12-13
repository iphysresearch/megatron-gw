import pytest
from gwtoolkit.gw.waveform import WaveformDataset
import itertools
# import numpy


@pytest.fixture
def init_wfc():
    return WaveformDataset(sampling_frequency=4096, duration=8)


def test_property(init_wfc):
    assert init_wfc.f_max == 2048.0
    assert init_wfc.delta_f == 0.125
    assert init_wfc.delta_t == 0.000244140625
    assert init_wfc.num_f == 16385
    assert init_wfc.num_t == 32768

    init_wfc.f_max = 4096.0
    assert init_wfc.sampling_frequency == 8192.0
    assert init_wfc.duration == 8
    assert init_wfc.f_max == 4096.0
    assert init_wfc.delta_f == 0.125
    assert init_wfc.delta_t == 0.0001220703125
    assert init_wfc.num_f == 32769
    assert init_wfc.num_t == 65536

    init_wfc.delta_t = 1/8192/2
    assert init_wfc.sampling_frequency == 16384.0
    assert init_wfc.duration == 8
    assert init_wfc.f_max == 8192.0
    assert init_wfc.delta_f == 0.125
    assert init_wfc.delta_t == 6.103515625e-05
    assert init_wfc.num_f == 65537
    assert init_wfc.num_t == 131072

    init_wfc.delta_f = 1/4
    assert init_wfc.sampling_frequency == 16384.0
    assert init_wfc.duration == 4
    assert init_wfc.f_max == 8192.0
    assert init_wfc.delta_f == 0.25
    assert init_wfc.delta_t == 6.103515625e-05
    assert init_wfc.num_f == 32769
    assert init_wfc.num_t == 65536

    init_wfc.duration = 16
    assert init_wfc.sampling_frequency == 16384.0
    assert init_wfc.duration == 16
    assert init_wfc.f_max == 8192.0
    assert init_wfc.delta_f == 0.0625
    assert init_wfc.delta_t == 6.103515625e-05
    assert init_wfc.num_f == 131073
    assert init_wfc.num_t == 262144

    # assert len(init_wfc.sample_times) == init_wfc.num_t
    # assert len(init_wfc.sample_frequencies) == init_wfc.num_f
    # assert len(init_wfc.sample_fft_frequencies) == init_wfc.num_t
    # assert init_wfc.sample_times.dtype == init_wfc.sample_frequencies.dtype \
    #                                    == init_wfc.sample_fft_frequencies.dtype \
    #                                    == numpy.float32


param = dict(
    mass_1=36., mass_2=29.,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)
dets = ['H1', 'L1', 'V1', 'K1', 'A1', 'GEO600', 'CE', 'ET'][:-2]


def test_prior(init_wfc):
    for filename in [None, './gwtoolkit/gw/prior_files/default.prior']:
        init_wfc.load_prior_source_detector(filename=filename, dets=dets[:2])
        init_wfc.prior.update_prior_range(mass_1=(10, 200), mass_2=(20, 80))
        init_wfc.prior.append_uniform_prior('geocent_time', 100, 101, latex_label='$t_c$', unit='$s$')
        init_wfc.prior.append_sine_prior('tilt_1', 0, 3.141592653589793,
                                         latex_label='$\\theta_1$', unit=None, boundary=None)
        init_wfc.prior.append_cosine_prior('dec', -1.5707963267948966, 1.5707963267948966,
                                           latex_label='$\\mathrm{DEC}$', unit=None, boundary=None)
        init_wfc.prior.append_uniformsourceframe_prior(name='luminosity_distance', minimum=1e2, maximum=5e3)
        for l, r in zip(['mass_1', 'mass_2', 'luminosity_distance'], [(10, 200), (20, 80), (1e2, 5e3)]):
            assert (init_wfc.prior[l].minimum, init_wfc.prior[l].maximum) == r
        init_wfc.prior.append_powerlaw_prior(name='luminosity_distance', alpha=2, minimum=50, maximum=2000,
                                             unit='Mpc', latex_label='$d_L$')
        assert (init_wfc.prior[l].minimum, init_wfc.prior[l].maximum) == (50, 2000)
    assert init_wfc.prior._show_priors() == print(init_wfc.prior)

    for conversion in ['BBH', 'BNS']:
        init_wfc.conversion = conversion
        init_wfc.load_prior_source_detector(dets=dets[:2])
        assert [len(value) for _, value in init_wfc.prior.sampling(size=1000).items()] == [1000, ]*14
        samples = init_wfc.prior.sampling(subset=['chirp_mass', 'luminosity_distance'], size=100)
        assert init_wfc.prior['chirp_mass'].minimum < samples['chirp_mass'].min()
        assert init_wfc.prior['chirp_mass'].maximum > samples['chirp_mass'].max()
        assert init_wfc.prior['luminosity_distance'].minimum < samples['luminosity_distance'].min()
        assert init_wfc.prior['luminosity_distance'].maximum > samples['luminosity_distance'].max()


def test_wfd(init_wfc):
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=150.,
                              minimum_frequency=20.)
    init_wfc.load_prior_source_detector(base='bilby',
                                        dets=dets,
                                        waveform_arguments=waveform_arguments)

    assert init_wfc.frequency_colored_noise.shape == (len(dets), 16385)
    assert init_wfc.frequency_whitened_noise.shape == (len(dets), 16385)
    assert init_wfc.time_colored_noise.shape == (len(dets), 16384*2)
    assert init_wfc.time_whitened_noise.shape == (len(dets), 16384*2)
    for kind in ['plus', 'cross']:
        assert len(init_wfc.frequency_waveform_polarizations[kind]) == 16385
        assert len(init_wfc.time_waveform_polarizations[kind]) == 16384*2
    assert init_wfc.frequency_colored_noise_block(20).shape == (20, len(dets), 16385)
    assert init_wfc.frequency_whitened_noise_block(20).shape == (20, len(dets), 16385)
    assert init_wfc.time_colored_noise_block(20).shape == (20, len(dets), 16384*2)
    assert init_wfc.time_whitened_noise_block(20).shape == (20, len(dets), 16384*2)

    for geocent_time, target_snr in itertools.product([(101, 102, None)], [None, (0, 0), (0, 30)]):
        (freq_block,
         freq_meta,
         noise_freq_block,
         freq_block_whitened) = init_wfc.frequency_waveform_response_block(10, 100,
                                                                           geocent_time=geocent_time,
                                                                           target_optimal_snr_tuple=target_snr)
        (time_block,
         time_meta,
         noise_time_block,
         time_block_whiten,
         _,
         _) = init_wfc.time_waveform_response_block(10, 100,
                                                                    geocent_time=geocent_time,
                                                                    target_optimal_snr_tuple=target_snr)
        assert freq_block.shape == noise_freq_block.shape == freq_block_whitened.shape == (10, len(dets), 16385)
        assert time_block.shape == noise_time_block.shape == time_block_whiten.shape == (10, len(dets), 16384*2)
        assert False not in [value.shape == (10, len(dets))
                             if '_snr' in key else len(value) == 10
                             for key, value in freq_meta.items()]
        assert False not in [value.shape == (10, len(dets))
                             if '_snr' in key else len(value) == 10
                             for key, value in time_meta.items()]
        if isinstance(target_snr, tuple) and target_snr[1]:
            assert False not in (freq_meta['optimal_snr'][:, target_snr[0]] == target_snr[1]).tolist()
            assert False not in (time_meta['optimal_snr'][:, target_snr[0]] == target_snr[1]).tolist()
