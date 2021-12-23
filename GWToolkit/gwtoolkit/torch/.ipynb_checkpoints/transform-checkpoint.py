"""
Transforms
"""
import numpy
import torch
from sklearn import preprocessing


class Normalize_params():
    """Standardize for parameters
    """

    def __init__(self, kind, **kwargs):
        if kind == 'minmax':
            self.standardize = lambda X: self.minmaxscaler(X, **self.kwargs)
            self.standardize_inv = lambda X: self.minmaxscaler_inverse(X, **self.kwargs)
            assert 'wfd' in kwargs
            assert 'labels' in kwargs
            self.kwargs = kwargs
            print(f"\tStandardize by '{kind}' for {len(kwargs['labels'])} parameters.")

    def __call__(self, sample):
        # Self check
        # assert numpy.allclose(samples, self.minmaxscaler_inverse(self.minmaxscaler(sample, wfd, labels), wfd, labels))
        sample = self.standardize(sample)
        return sample

    def minmaxscaler(self, data, wfd, labels, feature_range=(0, 1)):
        """
        'minimax' method to standardizing
        """
        scale = preprocessing.MinMaxScaler(feature_range=feature_range)
        minimaximum = numpy.asarray([[wfd.prior[label].minimum, wfd.prior[label].maximum]
                                     for label in labels])
        scale.fit(minimaximum.T)
        return scale.transform(data)

    def minmaxscaler_inverse(self, data, wfd, labels, feature_range=(0, 1)):
        """
        'minimax' method to inversely standardizing
        """
        scale = preprocessing.MinMaxScaler(feature_range=feature_range)
        minimaximum = numpy.asarray([[wfd.prior[label].minimum, wfd.prior[label].maximum]
                                     for label in labels])
        scale.fit(minimaximum.T)
        return scale.inverse_transform(data)


class Patching_data():
    """Patching for strain
    """

    def __init__(self, patch_size, overlap, sampling_frequency):
        """
        patch_size, sec
        overlap, %
        """
        self.nperseg = int(patch_size * sampling_frequency)  # sec
        # noverlap must be less than nperseg.
        self.noverlap = int(overlap * self.nperseg)  # [%]
        # nstep = nperseg - noverlap
        print(f'\tPatching with patch size={patch_size}s and overlap={overlap*100.0}%.')

    def __call__(self, data):
        # ! data.shape = (num, dets, duration*sample_rate)
        # ! return shape = (num, dets * (duration / patch_size * (1 - overlap)) - 1)
        shape = data.shape
        # Created strided array of data segments
        if self.nperseg == 1 and self.noverlap == 0:
            return data[..., numpy.newaxis]
        # https://stackoverflow.com/a/5568169  also
        # https://iphysresearch.github.io/blog/post/signal_processing/spectral_analysis_scipy/#_fft_helper
        nstep = self.nperseg - self.noverlap
        shape = shape[:-1]+((shape[-1]-self.noverlap)//nstep, self.nperseg)
        strides = data.strides[:-1]+(nstep*data.strides[-1], data.strides[-1])
        return numpy.lib.stride_tricks.as_strided(data, shape=shape,
                                                  strides=strides).reshape(shape[0],
                                                                           -1,
                                                                           self.nperseg)


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
