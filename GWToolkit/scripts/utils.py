import numpy as np
from scipy import constants as C
from astropy import constants as const
import functools
import pycbc.psd
from random import random


class Data_utils(object):
    def __init__(self,time_duration,sampling_rate):
        self.time_duration = time_duration
        self.sampling_rate =sampling_rate
        self.f_min = 20

        self.psd = self.get_psd_2()

    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_rate / 2.0

    @f_max.setter
    def f_max(self, f_max):
        self.sampling_rate = 2.0 * f_max

    @property
    def delta_t(self):
        return 1.0 / self.sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self.sampling_rate = 1.0 / delta_t

    @property
    def delta_f(self):
        return 1.0 / self.time_duration

    @delta_f.setter
    def delta_f(self, delta_f):
        self.time_duration = 1.0 / delta_f

    @property
    def Nt(self):
        return int(self.time_duration * self.sampling_rate)

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f) + 1
    
    @property
    def sample_times(self):
        """Array of times at which waveforms are sampled."""
        return np.linspace(0.0, self.time_duration,
                           num=self.Nt,
                           endpoint=False,
                           dtype=np.float32)

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return np.linspace(0.0, self.f_max,
                           num=self.Nf, endpoint=True,
                           dtype=np.float32)



    def tukey(self,M,alpha=0.5):
        """
        Tukey window code copied from scipy
        """
        n = np.arange(0, M)
        width = int(np.floor(alpha*(M-1)/2.0))
        n1 = n[:width+1]
        n2 = n[width+1:M-width-1]
        n3 = n[M-width-1:]

        w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
        w2 = np.ones(n2.shape)
        w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
        w = np.concatenate((w1, w2, w3))

        return np.array(w[:M])

    def get_psd_2(self):
        psd = pycbc.psd.analytical.aLIGOaLIGODesignSensitivityT1800044(self.Nf, self.delta_f, self.f_min)
        return np.sqrt(psd.data) 


    def get_snr(self,data):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min
        
        N = int(T_obs*fs)
        df = 1.0/T_obs
        dt = 1.0/fs
        fidx = int(fmin/df)

        win = self.tukey(N,alpha=1.0/8.0)
        idx = np.argwhere(psd>0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0/psd[idx]

        xf = np.fft.rfft(data*win)*dt
        SNRsq = 4.0*np.sum((np.abs(xf[fidx:])**2)*invpsd[fidx:])*df
        return np.sqrt(SNRsq)
    
    def get_inner_product(self,data1,data2):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min
        
        N = int(T_obs*fs)
        df = 1.0/T_obs
        dt = 1.0/fs
        fidx = int(fmin/df)

        win = self.tukey(N,alpha=1.0/8.0)
        idx = np.argwhere(psd>0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0/psd[idx]

        xf1 = np.fft.rfft(data1*win)*dt
        xf2 = np.fft.rfft(data2*win)*dt
        SNRsq = 2.0*np.sum((xf1[fidx:] * np.conjugate(xf2[fidx:]) + np.conjugate(xf1[fidx:])*xf2[fidx:]) *invpsd[fidx:])*df
        return np.sqrt(SNRsq).real
    
    def get_overlap(self,data1,data2):
        #normalize data
        data_1 = data1 / self.get_snr(data1)
        data_2 = data2 / self.get_snr(data2)
        return self.get_inner_product(data_1,data_2)

class Glitch_Generator(Data_utils):
    def __init__(self, time_duration, sampling_rate):
        super().__init__(time_duration, sampling_rate)
        self.t0_range = [5,7] 

    def liner_random(self,a,b):
        return np.abs(a-b) * random() + min(a,b)

    def Gaussian(self,t0,tau,snr):
        tmp1 = ((self.sample_times - t0)**2) / (2*tau**2)
        tmp2 = np.exp(- tmp1)
        return snr * tmp2 / self.get_snr(tmp2)

    def Sine_Gaussian(self,f0,t0,Q,snr):
        tau = Q / (np.sqrt(2) * np.pi * f0)
        tmp1 = ((self.sample_times - t0)**2) / (2*tau**2)
        tmp2 = np.sin(2*np.pi*f0*(self.sample_times - t0)) * np.exp(- tmp1)
        return snr * tmp2 / self.get_snr(tmp2)
    
    def Ring_Down(self,f0,t0,Q,snr):
        index = 0
        for idx in range(len(self.sample_times)):
            if self.sample_times[idx] >= t0:
                index = idx
                break
        tmp1 = self.Sine_Gaussian(f0,t0,Q,snr)
        tmp1[:index] = 0
        return snr * tmp1 / self.get_snr(tmp1)
    
    def Chrip_like(self,m1,m2,t0,snr):
        m_c = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        tau = self.sample_times - t0
        phi = -2 * (5 * C.G * m_c / C.c**3)**(-5/8) * tau**(5/8)
        tmp1 = tau**(-1/4)*np.sin(phi)
        return snr * tmp1 / self.get_snr(tmp1)

    def Scattered_light_like(self,f0,tau,t0,K,snr):
        tau_c = self.sample_times - t0
        phi_SL = 2 * np.pi * f0 * tau_c * (1 - K * tau_c**2)
        tmp1 = (tau_c**2) / (2*tau**2)
        tmp2 = np.sin(phi_SL) * np.exp(- tmp1)
        return snr * tmp2 / self.get_snr(tmp2)

    def generate_glitchs(self,num,type):
        '''
        type = [Gaussian,SG,RD,CL,SL]
        bug in CL
        '''
        snr = 1
        t0_range = self.t0_range
        t0 = self.liner_random(t0_range[0],t0_range[1])

        if type == 'Gaussian':
            tau_range = [4e-4,4e-3]
            tau = self.liner_random(tau_range[0],tau_range[1])
    
            return self.Gaussian(t0,tau,snr)
        
        if type == 'SG':
            f0_range = [50,1500]
            Q_range = [2,20]
            f0 = self.liner_random(f0_range[0],f0_range[1])
            Q = self.liner_random(Q_range[0],Q_range[1])
    
            return self.Sine_Gaussian(f0,t0,Q,snr)
        
        if type == 'RD':
            f0_range = [50,1500]
            Q_range = [2,20]
            f0 = self.liner_random(f0_range[0],f0_range[1])
            Q = self.liner_random(Q_range[0],Q_range[1])
    
            return self.Ring_Down(f0,t0,Q,snr)
        
        if type == 'CL':
            M_sun = float(const.M_sun.base)
            m1_range = [1.4*M_sun,30.0*M_sun]
            m2_range = m1_range
            m1 = self.liner_random(m1_range[0],m1_range[1])
            m2 = self.liner_random(m2_range[0],m2_range[1])
    
            return self.Chrip_like(m1,m2,t0,snr)

        if type == 'SL':
            f0_range = [32.0,64.0]
            tau_range = [0.2,0.5]
            f0 = self.liner_random(f0_range[0],f0_range[1])
            tau = self.liner_random(tau_range[1],tau_range[1])
            K = 0.5
    
            return self.Scattered_light_like(f0,tau,t0,K,snr)
              








