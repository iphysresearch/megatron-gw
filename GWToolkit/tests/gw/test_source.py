import pytest
from gwtoolkit.gw.source import Source
import itertools


param1 = dict(
    mass_1=36., mass_2=29.,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)
param2 = dict(
    mass_ratio=0.5, chirp_mass=29,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)


@pytest.fixture
def init_source():
    source = Source(base='bilby', conversion='BBH', sampling_frequency=2048, duration=4)
    return source


def test_property(init_source):
    for key, param in itertools.product(['plus', 'cross'], [param1, param2]):
        wfg = init_source.waveform_generator(param)
        assert len(wfg.time_domain_strain()[key]) == init_source.num_t
        assert len(wfg.frequency_domain_strain()[key]) == init_source.num_f

    for key, conversion, param in itertools.product(['plus', 'cross'], ['BNS', None], [param1, param2]):
        init_source._is_conversion_bbh_or_bns(conversion)
        assert init_source.waveform_approximant == ('IMRPhenomPv2_NRTidal' if conversion == 'BNS' else 'IMRPhenomPv2')
        assert init_source.frequency_domain_source_model.__name__ == ('lal_binary_neutron_star'
                                                                      if conversion == 'BNS'
                                                                      else 'lal_binary_black_hole')
        assert init_source.parameter_conversion.__name__ == ('convert_to_lal_binary_neutron_star_parameters'
                                                             if conversion == 'BNS'
                                                             else 'convert_to_lal_binary_black_hole_parameters')
        wfg = init_source.waveform_generator(param)
        assert len(wfg.time_domain_strain()[key]) == init_source.num_t
        assert len(wfg.frequency_domain_strain()[key]) == init_source.num_f

    init_source.base = 'pycbc'
    for key, conversion, param in itertools.product(['plus', 'cross'], ['BBH', 'BNS', None], [param1, param2]):
        init_source._is_conversion_bbh_or_bns(conversion)
        assert init_source.waveform_approximant == ('IMRPhenomPv2_NRTidal' if conversion == 'BNS' else 'IMRPhenomPv2')
        assert init_source.frequency_domain_source_model.__name__ == ('lal_binary_neutron_star'
                                                                      if conversion == 'BNS'
                                                                      else 'lal_binary_black_hole')
        assert init_source.parameter_conversion.__name__ == ('convert_to_lal_binary_neutron_star_parameters'
                                                             if conversion == 'BNS'
                                                             else 'convert_to_lal_binary_black_hole_parameters')
        wfg = init_source.waveform_generator(param)
        assert len(wfg.time_domain_strain()[key]) == init_source.num_t
        assert len(wfg.frequency_domain_strain()[key]) == init_source.num_f
