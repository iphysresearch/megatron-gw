"""
prior
"""
from pathlib import Path
import bilby
from bilby.core.prior import PriorDict


class Priors(PriorDict):
    """
    Prior
    """
    def __init__(self, filename=None, conversion=None):
        self._is_conversion_bbh_or_bns(conversion)
        super().__init__(conversion_function=self.conversion_function)

        if isinstance(filename, (str, Path)):
            self.from_file(filename=filename)
            # print('Using priors in', filename)
        else:
            # print("No prior given, using default BBH priors.")
            self.update(bilby.gw.prior.BBHPriorDict())

    def _is_conversion_bbh_or_bns(self, conversion):
        if conversion == 'BBH':
            self.conversion_function = bilby.gw.conversion.generate_all_bbh_parameters
        elif conversion == 'BNS':
            self.conversion_function = bilby.gw.conversion.generate_all_bns_parameters
        else:
            self.conversion_function = None

    def update_prior_range(self, **kwargs):
        """ Update the exist prior's range

        Eg: self.update_prior_range(mass_1=(10, 11))
        """
        for key, value in self.items():
            value.minimum, value.maximum = kwargs.get(key, (value.minimum, value.maximum))

    def append_uniform_prior(self, name, minimum, maximum, **kwargs):
        """append/create/cover an Uniform prior

        Eg: self.append_uniform_prior('geocent_time', 100, 101,
                                      latex_label='$t_c$', unit='$s$')
        Eg: self.append_uniform_prior('chirp_mass', 10, 101,
                                      latex_label='$\\mathcal{M}$', unit=None, boundary=None)
        Eg: self.append_uniform_prior('ra', 0, 6.283185307179586,
                                      latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic')
        Eg: self.append_uniform_prior('psi', 0, 3.141592653589793,
                                      latex_label='$\\psi$', unit=None, boundary='periodic')
        """
        self[name] = bilby.core.prior.Uniform(minimum, maximum, name,
                                              kwargs.get('latex_label', None),
                                              kwargs.get('unit', None),
                                              kwargs.get('boundary', None))

    def append_sine_prior(self, name, minimum, maximum, **kwargs):
        """append/create/cover an Sine prior

        Eg: self.append_sine_prior('tilt_1', 0, 3.141592653589793,
                                   latex_label='$\\theta_1$', unit=None, boundary=None)
        """
        self[name] = bilby.core.prior.Sine(minimum, maximum, name,
                                           kwargs.get('latex_label', None),
                                           kwargs.get('unit', None),
                                           kwargs.get('boundary', None))

    def append_cosine_prior(self, name, minimum, maximum, **kwargs):
        """append/create/cover an Cosine prior

        Eg: self.append_cosine_prior('dec', -1.5707963267948966, 1.5707963267948966,
                                      latex_label='$\\mathrm{DEC}$', unit=None, boundary=None)
        """
        self[name] = bilby.core.prior.Cosine(minimum, maximum, name,
                                             kwargs.get('latex_label', None),
                                             kwargs.get('unit', None),
                                             kwargs.get('boundary', None))

    def append_powerlaw_prior(self, name, alpha, minimum, maximum, **kwargs):
        """append/create/cover a PowerLaw prior, p(x) ~ x^{alpha}

        Note: alpha=0 is a uniform distribution, alpha=-1 is uniform-in-log
        Eg: self.append_PowerLaw_prior(name='luminosity_distance', alpha=2, minimum=50, maximum=2000,
                                       unit='Mpc', latex_label='$d_L$')
        """
        self[name] = bilby.core.prior.PowerLaw(alpha, minimum, maximum, name,
                                               kwargs.get('latex_label', None),
                                               kwargs.get('unit', None),
                                               kwargs.get('boundary', None))

    def append_uniformsourceframe_prior(self, name, minimum, maximum, cosmology=None, **kwargs):
        """append/create/cover an UniformSourceFrame prior

        Eg: self.append_UniformSourceFrame_prior(name='luminosity_distance', minimum=1e2, maximum=5e3)
        """
        self[name] = bilby.gw.prior.UniformSourceFrame(minimum, maximum, cosmology, name,
                                                       kwargs.get('latex_label', None),
                                                       kwargs.get('unit', None),
                                                       kwargs.get('boundary', None))

    def _show_priors(self):
        """Print current priors
        """
        print('),\n'.join(str(self).split('),')))

    def sampling(self, subset=None, size=None):
        """
        Draw samples from the prior set (or the subset)

        Parameters
        ==========
        keys: list (optional)
            List of prior keys to draw samples from
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the drawn samples
        """
        if subset:
            return self.sample_subset(subset, size)

        return self.sample(size)
