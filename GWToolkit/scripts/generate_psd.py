from logging import disable
import sys
sys.path.append('..')
from gwtoolkit.gw import readligo as rl
from pycbc.types.timeseries import TimeSeries
from lal import LIGOTimeGPS
from pycbc.psd import interpolate
import numpy as np
# import matplotlib.pyplot as plt
# from utils import Data_utils, Glitch_Generator
# from astropy import constants as const
from gwtoolkit.gw import hdffiles as hd
from tqdm import tqdm


class PSD_Sampler(object):
    def __init__(self, datadir='/workspace/zhaoty/O1_data/',
                 random_seed=42, noise_interval=4096, signal_length=8,
                 target_sampling_rate=4096, 
                 dq_bits=(0, 1, 2, 3),
                 inj_bits=(0, 1, 2, 4)):
        self.datadir = datadir
        self.random_seed = random_seed
        self.filelist = rl.FileList(directory=self.datadir)
        self.noise_timeline = hd.NoiseTimeline(background_data_directory=self.datadir,
                                               random_seed=self.random_seed)
        self.noise_interval = noise_interval
        self.signal_length = signal_length
        
        self.dq_bits = dq_bits
        self.inj_bits = inj_bits

        self.original_sampling_rate = 16384
        self.target_sampling_rate = target_sampling_rate
        self.Nf = int(self.target_sampling_rate/2*signal_length+1)
        self.sample_freq = np.linspace(0.0, target_sampling_rate/2.0,
                           num=self.Nf, endpoint=True,
                           dtype=np.float32)
        self.PSDs = None
        self.ra_PSDs = None

    def gps_time_sampler(self,):

        delta_t = int(self.noise_interval/2)
        '''
        half of Width of the background noise interval (in seconds) around the event_time,
        ;    which is used to make the injection. Should be larger than (see below):
        ;   sample_length = seconds_before_event + seconds_after_event
        ; because we need to crop off the edges that are corrupted by the whitening.
        '''
        # sample is interval middle point
        noise_times = self.noise_timeline.sample(delta_t=delta_t, dq_bits=self.dq_bits,
                                                 inj_bits=self.inj_bits, return_paths=False)
        return noise_times - delta_t

    def generate_psd(self, start, ifo='H1', signal_length=8):

        sampling_factor = int(self.original_sampling_rate / self.target_sampling_rate)
        stop = start + self.noise_interval

        strain, time, dqmask, injmask = rl.getstrain_cvmfs(start, stop, ifo, self.filelist)

        strain = strain[::sampling_factor]

        timeseries = TimeSeries(initial_array=strain,
                                delta_t=1.0/self.target_sampling_rate,
                                epoch=LIGOTimeGPS(start))
        psd = timeseries.psd(4)
        psd = interpolate(psd, delta_f=1 / signal_length)
        
        return psd

    def generate_psds(self, num, ifo='H1', signal_length=8,):
        psds = np.zeros([num, self.Nf])
        for i in tqdm(range(num),disable=disable):
            start = self.gps_time_sampler()
            psd = self.generate_psd(start, ifo=ifo, signal_length=signal_length)
            psds[i, :] = np.sqrt(psd.data)
        
        # self.PSDs = psds
        #print('Generate {} PSDs from real LIGO noise done !!!'.format(num))
        return psds
    
    def rand_average_psds(self, num_per_psd, psd_num):
        ra_psds = np.zeros([psd_num, self.Nf])
        for i in tqdm(range(psd_num)):
            psds = self.generate_psds(num_per_psd)
            ra_psds[i, :] = np.mean(psds,axis=0)
        print('Generate {} random average PSDs from real LIGO noise done !!!'.format(psd_num))
        return ra_psds


    def save_txt(self, idx, psd, dir='./'):
        with open('{}psd-{}'.format(dir, idx), 'w') as f:
            for i in range(psd.shape[0]):
                f.write('{:.16e} {:.16e}\n'.format(self.sample_freq[i], psd[i]))

    def save_psds(self, psds, dir):
        for i in range(psds.shape[0]):
            self.save_txt(i, psds[i, :],dir)


class Glitch_Sampler(object):
    def __init__(self):
        pass



def main():
    # work with readligo.py and hdffiles.py
    psd_sampler = PSD_Sampler('/workspace/zhaoty/dataset/ligo_H1_test/',noise_interval=1024)
    psd_sampler.PSDs = psd_sampler.generate_psds(10)
    # np.save('psds.npy', psd_sampler.PSDs)
    psd_sampler.ra_PSDs = psd_sampler.rand_average_psds(5,10)
    psd_sampler.save_psds(psd_sampler.ra_PSDs, 'test/')
    
    return 0


if __name__ == '__main__':
    main()
