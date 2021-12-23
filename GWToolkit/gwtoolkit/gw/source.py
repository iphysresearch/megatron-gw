"""
GW source
"""
import bilby
from .base import Waveform
# import pycbc
bilby.core.utils.setup_logger(log_level=100)


class Source(Waveform):
    """
    Source
    """
    def __init__(self, base, conversion, sampling_frequency, duration):
        """
        Parameters
        ==========
        base: str, bilby/pycbc
        conversion: str, BNH/BNS
        sampling_frequency: float, optional
            The sampling frequency
        duration: float, optional
            Time duration of data
        """
        super().__init__(sampling_frequency, duration)
        self.base = None
        self.waveform_approximant = None
        self.parameter_conversion = None
        self.frequency_domain_source_model = None
        self.time_domain_source_model = None
        self._is_base_valid(base)
        self._is_conversion_bbh_or_bns(conversion)

    def waveform_generator(self, parameters, **kwargs):
        """A waveform generator

        Parameters
        ==========
        parameters: dict, optional
            Initial values for the parameters
        waveform_arguments: dict, optional
            A dictionary of fixed keyword arguments to pass to either
            `frequency_domain_source_model` or `time_domain_source_model`.
            Note: the arguments of frequency_domain_source_model (except the first,
            which is the frequencies at which to compute the strain) will be added to
            the WaveformGenerator object and initialised to `None`.
            - waveform_approximant  ('IMRPhenomPv2' for BBH, 'IMRPhenomPv2_NRTidal' for BNS)
                Details: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html
            - reference_frequency   (50 Hz)
            - minimum_frequency     (20 Hz)
            - maximum_frequency     (sampling_frequency/2)
            - catch_waveform_errors (False)
            - pn_spin_order         (-1)
            - pn_tidal_order        (-1)
            - pn_phase_order        (-1)
            - pn_amplitude_order    (0)
            - start_time            (0)
            - mode_array:           (None)
                Activate a specific mode array and evaluate the model using those
                modes only.  e.g. waveform_arguments =
                dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
                returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
                specify modes that are included in that particular model.  e.g.
                waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
                mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
                55 modes are not included in this model.  Be aware that some models
                only take positive modes and return the positive and the negative
                mode together, while others need to call both.  e.g.
                waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
                mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
                However, waveform_arguments =
                dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
                returns the 22 and 4-4 of IMRPhenomXHM.
        Return
            bilby.gw.WaveformGenerator
        """
        # Fixed arguments passed into the source model
        waveform_arguments = dict(
            waveform_approximant=self.waveform_approximant,
            reference_frequency=50.0,
            minimum_frequency=20.0,
            maximum_frequency=self.f_max,
            catch_waveform_errors=False,
            pn_spin_order=-1,
            pn_tidal_order=-1,
            pn_phase_order=-1,
            pn_amplitude_order=0,
            start_time=0,
        )
        waveform_arguments.update(kwargs)
        # Create the waveform_generator using a LAL BinaryBlackHole source function
        # the generator will convert all the parameters
        return bilby.gw.WaveformGenerator(
            self.duration, self.sampling_frequency, waveform_arguments['start_time'],
            self.frequency_domain_source_model,
            self.time_domain_source_model,
            parameters, self.parameter_conversion,
            waveform_arguments)

    def _is_base_valid(self, base):
        self.base = base if base in ['bilby', 'pycbc'] else None

    def _is_conversion_bbh_or_bns(self, conversion):
        """
        Return
        parameter_conversion: func, optional
            Function to convert from sampled parameters to parameters of the
            waveform generator. Default value is the identity, i.e. it leaves
            the parameters unaffected.
        """
        if conversion == 'BBH':
            self.parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
            self.frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
            self.waveform_approximant = 'IMRPhenomPv2'
        elif conversion == 'BNS':
            self.parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
            self.frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star
            self.waveform_approximant = 'IMRPhenomPv2_NRTidal'
        else:
            print('Conversion input not understood, it should be BBH or BNS. Now as default for BBH.')
            self.parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
            self.frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
            self.waveform_approximant = 'IMRPhenomPv2'
