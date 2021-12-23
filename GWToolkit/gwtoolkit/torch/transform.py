"""
Transforms
"""
import sys
import numpy
import torch
from sklearn import preprocessing


class Normalize_strain():
    """Standardize for strains
    """
    def __init__(self, feature_range=(-1, 1), verbose=True, **kwargs):
        # Print debug messages or not?
        self.verbose = verbose
        self.feat_min, self.feat_max = feature_range

    def transform_data(self, X, Xmin=None, Xmax=None):
        """'minimax' method to standardizing
        Input
            X: ndarray
            Xmin:
            Xmax:

        Return
            X_scaled: same shape of X
            Xmin[...,0,0]: array with length of X
            Xmax[...,0,0]: array with length of X
        """
        if Xmin is None:
            Xmin = X.min(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
        if Xmax is None:
            Xmax = X.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
        X_std = (X - Xmin) / (Xmax - Xmin)
        X_scaled = X_std * (self.feat_max - self.feat_min) + self.feat_min
        return X_scaled, Xmin, Xmax

    def transform_signalornoise(self, X, Xmin=None, Xmax=None):
        """'minimax' method to standardizing
        Input
            X: ndarray
            Xmin:
            Xmax:

        Return
            X_scaled: same shape of X
            Xmin[...,0,0]: array with length of X
            Xmax[...,0,0]: array with length of X
        """
        if Xmin is None:
            Xmin = X.min(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
        if Xmax is None:
            Xmax = X.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
        X_std = (X - Xmin/2) / (Xmax - Xmin)
        X_scaled = X_std * (self.feat_max - self.feat_min) + self.feat_min/2
        return X_scaled

    def inverse_transform_data(self, X_scaled, Xmin, Xmax):
        """'minimax' method to inverse standardizing
        Input:
            X_scaled: ndarray
            maxx: array with length of X
            minn: array with length of X
        Return
            :same shape of X
        """
        # Xmax = maxx[...,np.newaxis,np.newaxis]
        # Xmin = minn[...,np.newaxis,np.newaxis]
        return (X_scaled - self.feat_min) * (Xmax - Xmin) / (self.feat_max - self.feat_min) + Xmin

    def inverse_transform_signalornoise(self, X_scaled, Xmin, Xmax):
        """'minimax' method to inverse standardizing
        Input:
            X_scaled: ndarray
            maxx: array with length of X
            minn: array with length of X
        Return
            :same shape of X
        """
        # Xmax = maxx[...,np.newaxis,np.newaxis]
        # Xmin = minn[...,np.newaxis,np.newaxis]
        return (X_scaled - self.feat_min/2) * (Xmax - Xmin) / (self.feat_max - self.feat_min) + Xmin/2

    def vprint(self, string, *args, **kwargs):
        """
        Verbose printing: Wrapper around `print()` to only call it if
        `self.verbose` is set to true.

        Args:
            string (str): String to be printed if `self.verbose`
                is `True`.
            *args: Arguments passed to `print()`.
            **kwargs: Keyword arguments passed to `print()`.
        """

        if self.verbose:
            print(string, *args, **kwargs)
            sys.stdout.flush()


class Normalize_params():
    """Standardize for parameters
    """

    def __init__(self, kind, verbose=True, **kwargs):
        # Print debug messages or not?
        self.verbose = verbose

        if kind == 'minmax':
            self.standardize = lambda X: self.minmaxscaler(X, **self.kwargs)
            self.standardize_inv = lambda X: self.minmaxscaler_inverse(X, **self.kwargs)
            assert 'wfd' in kwargs
            assert 'labels' in kwargs
            self.kwargs = kwargs
            self.vprint(f"\tStandardize by '{kind}' for {len(kwargs['labels'])} parameters.")

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

    def vprint(self, string, *args, **kwargs):
        """
        Verbose printing: Wrapper around `print()` to only call it if
        `self.verbose` is set to true.

        Args:
            string (str): String to be printed if `self.verbose`
                is `True`.
            *args: Arguments passed to `print()`.
            **kwargs: Keyword arguments passed to `print()`.
        """

        if self.verbose:
            print(string, *args, **kwargs)
            sys.stdout.flush()


class Patching_data():
    """Patching for strain
    """

    def __init__(self, patch_size, overlap, sampling_frequency, duration=None, verbose=True):
        """
        patch_size, sec
        overlap, %
        """
        # Print debug messages or not?
        self.verbose = verbose

        self.nperseg = int(patch_size * sampling_frequency)  # sec
        # noverlap must be less than nperseg.
        self.noverlap = int(overlap * self.nperseg)  # [%]
        # nstep = nperseg - noverlap
        self.vprint(f'\tPatching with patch size={patch_size}s and overlap={overlap*100.0}%.')
        if duration:
            self.output_shape = self.__call__(numpy.empty(shape=(1, duration * sampling_frequency))).shape

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

    def inverse(self, data):
        # data.shape = (..., num_tokens, nperseg)
        # return shape = (..., nperseg + noverlap * (num_tokens-1) )
        shape = data.shape
        return numpy.concatenate([
            data[..., 0, :self.noverlap],
            numpy.reshape(data[..., -self.noverlap:],
                          newshape=(*shape[:-2], -1))
        ],
            axis=-1)

    def vprint(self, string, *args, **kwargs):
        """
        Verbose printing: Wrapper around `print()` to only call it if
        `self.verbose` is set to true.

        Args:
            string (str): String to be printed if `self.verbose`
                is `True`.
            *args: Arguments passed to `print()`.
            **kwargs: Keyword arguments passed to `print()`.
        """

        if self.verbose:
            print(string, *args, **kwargs)
            sys.stdout.flush()


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
