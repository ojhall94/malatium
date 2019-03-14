#O. J. Hall 2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import lightkurve
from scipy.special import legendre as P
from scipy.misc import factorial as fct

from omnitool import literature_values as lv
plt.style.use(lightkurve.MPLSTYLE)

class star():
    def __init__(self, freqs, nyquist, numax, dnu, d02, nus, i):
        '''A class model that stores the basic stellar properties'''
        self.freqs = freqs
        self.nyquist = nyquist
        self.numax = numax
        self.dnu = dnu
        self.d02 = d02
        self.epsilon = 0.601 + 0.632*np.log(self.dnu)  #from Vrard et al. 2015 (for RGB)
        self.nmax = self.numax/self.dnu - self.epsilon #from Vrard et al. 2015
        self.lmax = 3     #Don't care about higher order
        self.Gamma = 1.   #Depends on the mode lifetimes (which I don't know)
        self.nus = .4     #Depends on rotation & coriolis force (which I don't understand yet)
        self.i = np.pi/4 #Determines the mode height

    def get_height(self, nunlm, hmax=50.):
        #I modulate the mode height based on a fudged estimate of the FWHM
        #No physics has gone into this
        fwhm = 0.66*self.numax**0.88 * 0.5 #This FWHM relation is for RGB only
        H = hmax * np.exp(-0.5 * (nunlm - self.numax)**2 / fwhm**2)
        return H

    def lorentzian(self, nunlm, l, m):
        #We set all mode heights to 1 to start with
        H = self.get_height(nunlm) * self.get_Epsilon(self.i, l, m)
        model = H / (1 + (4/self.Gamma**2)*(self.freqs - nunlm)**2)
        return model

    def get_Epsilon(self, i, l, m):
        #I use the prescriptions from Gizon & Solank 2003 and Handberg & Campante 2012
        if l == 0:
            return 1
        if l == 1:
            if m == 0:
                return np.cos(i)**2
            if np.abs(m) == 1:
                return 0.4 * np.sin(i)**2
        if l == 2:
            if m == 0:
                return 0.25 * (3 * np.cos(i)**2 - 1)**2
            if np.abs(m) ==1:
                return (3/8)*np.sin(2*i)**2
            if np.abs(m) == 2:
                return (3/8) * np.sin(i)**4
        if l == 3:
            if m == 0:
                return (1/64)*(5*np.cos(3*i) + 3*np.cos(i))**2
            if np.abs(m) == 1:
                return (3/64)*(5*np.cos(2*i) + 3)**2 * np.sin(i)**2
            if np.abs(m) == 2:
                return (15/8) * np.cos(i)**2 * np.sin(i)**4
            if np.abs(m) == 3:
                return (5/16)*np.sin(i)**6

    def asymodelocs(self, n, l, m):
        #d00, d01, d02, d03
        dnu0 = [0., 0., self.d02, self.d02]
        return self.dnu * (n + l/2 + self.epsilon) - dnu0[l] + m * self.nus

    def get_apodization(self):
        return np.sinc(self.freqs / 2.0 / self.nyquist)**2

    def harvey_guesses(self):
        ak, ae = 3.3, -0.48
        bk, be = -0.43, 0.86
        ck, ce = 3.59, -0.59
        dk, de = 0.02, 0.96
        wk, we = -0.82, 1.03
        hk, he = 6.95, -2.18
        a = 10**(ak + np.log10(self.numax)*ae)
        b = 10**(bk + np.log10(self.numax)*be)
        c = 10**(ck + np.log10(self.numax)*ce)
        d = 10**(dk + np.log10(self.numax)*de)
        h = a * 0.5
        j = b / 40.0
        width = 10**(wk + np.log10(self.numax)*we)
        height = 10**(hk + np.log10(self.numax)*he)
        scale = 1.0
        return a, b, c, d, h, j

    def harvey(self, a, b, c=4.0):
        #I need to find and include a harvey profile myself still
        return 0.9*a**2/b/(1 + (self.freqs/b)**c)

    def get_background(self, scale=0.2):
        a, b, c, d, h, j = self.harvey_guesses()
        return scale*(self.harvey(a, b) + self.harvey(c, d) + self.harvey(h, j, 2.0))

    def get_noise(self):
        return np.random.chisquare(2, size=len(self.freqs))

    def get_model(self):
        nn = np.arange(np.floor(self.nmax-10.), np.floor(self.nmax+10.), 1)
        model = np.ones(len(self.freqs))
        locs = np.ones([len(nn), self.lmax+1])
        for idx, n in enumerate(nn):
            for l in np.arange(self.lmax+1):
                locs[idx, l] = self.asymodelocs(n, l, 0.)
                if l == 0:
                    loc = self.asymodelocs(n, l, 0.)
                    model += self.lorentzian(locs[idx, l], l, 0.)
                else:
                    for m in np.arange(-l, l+1):
                        loc = self.asymodelocs(n, l, m)
                        model += self.lorentzian(loc, l, m) #change height of multiplet
        apod = self.get_apodization()
        background = 0. #self.get_background()
        return (model + background + self.get_noise()) * apod, locs

    def plot_model(self):
        model, locs = self.get_model()
        l0s = np.ones(locs.shape[0])*.81 * np.max(model)
        l1s = np.ones(locs.shape[0])*.82 * np.max(model)
        l2s = np.ones(locs.shape[0])*.83 * np.max(model)
        l3s = np.ones(locs.shape[0])*.84 * np.max(model)

        fig = plt.figure()
        plt.plot(self.freqs, model)
        plt.scatter(locs[:,0],l0s, marker=',',s=10,label='l=0')
        plt.scatter(locs[:,1],l1s, marker='*',s=10,label='l=1')
        plt.scatter(locs[:,2],l2s, marker='^',s=10,label='l=2')
        plt.scatter(locs[:,3],l3s, marker='o',s=10,label='l=3')
        plt.legend(fontsize=20)
        plt.show()
        return locs

if __name__ == '__main__':
    nyquist = 0.5 * (1./58.6) * u.hertz
    nyquist = nyquist.to(u.microhertz)
    fs = 1./(4*365) * (1/u.day)
    fs = fs.to(u.microhertz)

    #Parameters for 16 Cyg A
    nus = 0.411
    i = np.deg2rad(56.)
    d02 = 6.8
    Dnu = 102.
    numax = 2200.

    freqs = np.arange(fs.value, numax*2, fs.value)

    locs = star(freqs, nyquist, numax, Dnu, d02, nus, i).plot_model()
