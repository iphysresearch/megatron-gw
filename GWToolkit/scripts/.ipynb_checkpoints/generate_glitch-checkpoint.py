import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from gwtoolkit.gw.readligo import  getstrain_cvmfs
from gwtoolkit.gw.gwosc_cvmfs import GWOSC


class Glitch_Sampler(GWOSC):
    def __init__(self, glitch_dir = '../tests/gw/trainingset_v1d1_metadata.csv', signal_length=8):
        super().__init__(ifo='H1', data_dir='/workspace/zhaoty/dataset/O1_H1_All/',
                         sampling_frequency=4096, noise_interval=1024,
                 dq_bits=(0, 1, 2, 3), inj_bits=(0, 1, 2, 4))
        self.glitch_meta_data = pd.read_csv(glitch_dir)
        self.peak_time = self.glitch_meta_data.peak_time.astype('float64') + self.glitch_meta_data.peak_time_ns.astype('float64') * 1e-9
        self.start_time = self.glitch_meta_data.start_time.astype('float64') + self.glitch_meta_data.start_time_ns.astype('float64') * 1e-9
        self.duration = self.glitch_meta_data.duration
        self.signal_length = signal_length
    
    def get_start_time(self):
        idx = np.random.randint(0,high=len(self.glitch_meta_data))
        if self.duration[idx] > self.signal_length:
            return int(self.start_time[idx] - np.random.rand(1) * (self.duration[idx] - self.signal_length +0.5))
        else:
            return int(self.start_time[idx] + np.random.rand(1) * (self.signal_length - self.duration[idx] +0.5))
    
    def get_strain(self,):
        strain = None
        while strain is None:
            try:
                start_time = self.get_start_time()
                strain, time = getstrain_cvmfs(start_time, start_time+self.signal_length,
                                                 self.ifo, self.filelist, inj_dq_cache=0)
            except:
                # no data exist case resampling start_time
                pass
        sampling_factor = int(self.original_sampling_rate / self.sampling_frequency)
        strain = strain[::sampling_factor]
        return strain
    
    
def main():
    glitch_sampler = Glitch_Sampler()
    # the csv file pd.DataFrame
    glitch_meta_data = glitch_sampler.glitch_meta_data
    glitch_sampler.get_start_time()
    # get start time read strain from file and down sampling
    # 1.random time 
    # 2.random start time (before glitch) for glitch duration < 8s
    # 3.random start time (during glitch) for glitch duration > 8s
    glitch_sampler.get_strain()
    return 0


if __name__ == '__main__':
    main()