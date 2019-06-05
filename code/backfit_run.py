#!/usr/bin/env python3
#O. J. Hall 2019

import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import lightkurve as lk
import pandas as pd
import pystan
import astropy.units as u
import sys
from astropy.units import cds
import corner
import glob
import time
import lightkurve as lk
from astropy.io import ascii
timestr = time.strftime("%m%d-%H%M")

import argparse
parser = argparse.ArgumentParser(description='Run our PyStan model')
parser.add_argument('iters', type=int, help='Number of MCMC iterations in PyStan.')
parser.add_argument('idx',type=int,help='Index on the kiclist')
args = parser.parse_args()

__iter__ = args.iters

def create_model(overwrite=True):
    backfit = '''
    functions {
        real harvey(real f, real a, real b, real c){
            return 0.9*a^2/b/(1.0 + (f/b)^c);
        }
        real apod(real f, real nyq){
            real x = 3.14 / 2.0 * f / nyq;
            return (sin(x) / x)^2;
        }
    }
    data {
        int N;
        vector[N] f;
        vector[N] p;
        real white_est;
        real nyq_est;
        real numax_est;
        real scale_spread;
    }
    parameters {
        real loga;
        real logb;
        real logc;
        real logd;
        real logj;
        real logk;
        real lognumax;
        real<lower=0> white;
        real<lower=0> nyq;
        real scale;
    }
    transformed parameters {
        real numax;
        real logac = loga - logc;
        real logdb = logd - logb;

        numax = 10^lognumax;
    }
    model {
        real a = 10^loga;
        real b = 10^logb;
        real c = 10^logc;
        real d = 10^logd;
        real j = 10^logj;
        real k = 10^logk;
        real beta[N];

        for (i in 1:N){
            beta[i] = 1. / (apod(f[i], nyq) * scale
                    * (harvey(f[i], a, b, 4.0)
                    + harvey(f[i], c, d, 4.0)
                    + harvey(f[i], j, k, 2.0))
                    + white);
            }
        p ~ gamma(1., beta);

        numax_est ~ normal(numax, numax_est*0.1);
        white ~ normal(white_est, white_est*0.3);
        nyq ~ normal(nyq_est, nyq_est*0.01);
        scale ~ normal(1, scale_spread);

        loga ~ normal(3.4 + lognumax *.48, 0.3);
        logb ~ normal(-0.43 + lognumax * 0.86, 0.3);
        logc ~ normal(3.59 + lognumax * -0.59, 0.3);
        logd ~ normal(0.02 + lognumax * 0.96, 0.3);
        logj ~ normal(loga-1, 1.2);
        logk ~ normal(logb-1, 0.2);

        logac ~ lognormal(1., 1.);
        logdb ~ lognormal(1., 1.);
    }
    '''
    model_path = 'backfit.pkl'
    if overwrite:
        print('Updating Stan model')
        sm = pystan.StanModel(model_code = backfit, model_name='backfit')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()
    if os.path.isfile(model_path):
        print('Reading in Stan model')
        sm = pickle.load(open(model_path, 'rb'))
    else:
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = backfit, model_name='backfit')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()
        model_path = 'backfit.pkl'

def first_guess(numax):
    ak, ae = 3.3, -0.48
    bk, be = -0.43, 0.86
    ck, ce = 3.59, -0.59
    dk, de = 0.02, 0.96

    a = 10**(ak + np.log10(numax)*ae)
    b = 10**(bk + np.log10(numax)*be)
    c = 10**(ck + np.log10(numax)*ce)
    d = 10**(dk + np.log10(numax)*de)
    j = a * 0.5
    k = b / 40.0
    scale = 1.0

    return [np.log10(a), np.log10(b),
            np.log10(c), np.log10(d),
            np.log10(j), np.log10(k),
            np.log10(numax)]

def rebin(f, p, binsize=10):
    m = int(len(p)/binsize)

    bin_f = f[:m*binsize].reshape((m, binsize)).mean(1)
    bin_p = p[:m*binsize].reshape((m, binsize)).mean(1)

    return bin_f, bin_p

class run_stan:
    def __init__(self, data, init, dir):
        '''Core PyStan class.
        Input __init__:
        dat (dict): Dictionary of the data in pystan format.

        init (dict): Dictionary of initial guesses in pystan format.
        '''
        self.data = data
        self.init = init
        self.dir = dir

    def read_stan(self):
        '''Reads the existing stanmodel'''
        model_path = 'backfit.pkl'
        if os.path.isfile(model_path):
            sm = pickle.load(open(model_path, 'rb'))
        else:
            print('No stan model found')
            create_model(overwrite=True)
            sm = pickle.load(open(model_path, 'rb'))
        return sm

    def run_stan(self):
        '''Runs PyStan'''
        sm = self.read_stan()

        fit = sm.sampling(data = self.data,
                    iter= __iter__, chains=4,
                    init = [self.init, self.init, self.init, self.init])

        return fit

    def out_corner(self, fit):
        labels=['loga','logb','logc','logd','logj','logk',
                'white','numax','scale','nyq']
        verbose=[r'$\log_{10}a$',r'$\log_{10}b$',
                r'$\log_{10}c$',r'$\log_{10}d$',
                r'$\log_{10}j$',r'$\log_{10}k$',
                'white',r'$\nu_{\rm max}$','scale',r'$\nu_{\rm nyq}$']

        chain = np.array([fit[label] for label in labels])

        corner.corner(chain.T, labels=verbose, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True)

        plt.savefig(self.dir+'corner.png')
        plt.close('all')

    def out_stanplot(self, fit):
        fit.plot()
        plt.savefig(self.dir+'stanplot.png')
        plt.close('all')

    def harvey(self, f, a, b, c):
        harvey = 0.9*a**2/b/(1.0 + (f/b)**c);
        return harvey

    def get_apodization(self, freqs, nyquist):
        x = (np.pi * freqs) / (2 * nyquist)
        return (np.sin(x)/x)**2

    def get_background(self, f, a, b, c, d, j, k, white, numax, scale, nyq):
        background = np.zeros(len(f))
        background += self.get_apodization(f, nyq) * scale\
                        * (self.harvey(f, a, b, 4.) + self.harvey(f, c, d, 4.) + self.harvey(f, j, k, 2.))\
                        + white
        return background

    def out_modelplot(self, fit):
        labels=['loga','logb','logc','logd','logj','logk','white','numax','scale','nyq']
        res = np.array([np.median(fit[label]) for label in labels])
        res[0:6] = 10**res[0:6]
        model = self.get_background(f, *res)

        pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.25, label='Data', scale='log')
        ax.plot(f, model, label='Model')
        ax.plot(f, self.harvey(f, res[0],res[1], 4.), label='Harvey 1', ls=':')
        ax.plot(f, self.harvey(f, res[2],res[3], 4.), label='Harvey 2', ls=':')
        ax.plot(f, self.harvey(f, res[4],res[5], 2.), label='Harvey 3', ls=':')
        ax.plot(f, self.get_apodization(f, f[-1]), label='Apod', ls='--')
        ax.plot(f, res[-4]*np.ones_like(f), label='white',ls='-.')
        plt.legend(fontsize=10)

        plt.savefig(self.dir+'modelplot.png')
        plt.close('all')

    def out_pickle(self, fit):
        path = self.dir+'fit.pkl'
        with open(path, 'wb') as f:
            pickle.dump(fit.extract(), f)

    def __call__(self):
        fit = self.run_stan()
        self.out_pickle(fit)
        self.out_corner(fit)
        self.out_stanplot(fit)
        self.out_modelplot(fit)

        with open(self.dir+'summary.txt', "w") as text_file:
            print(fit.stansummary(),file=text_file)

        print('Run complete!')
        return fit

def get_folder(kic):
    fol = '/rds/projects/2018/daviesgr-asteroseismic-computation/ojh251/malatium/backfit' + '/'+str(kic)
    if not os.path.exists(fol):
        os.makedirs(fol)
    return fol + '/' + timestr +'_idx'+str(idx)+'_'+str(kic)+'_backfit_'

if __name__ == '__main__':
    idx = int(args.idx)

    # Get the star data
    mal = pd.read_csv('../data/malatium.csv', index_col=0)
    star = mal.loc[idx]
    kic = star.KIC
    numax = star.numax
    dnu = star.dnu

    #Get the output director
    dir = get_folder(kic)

    # Get the power spectrum
    # Col1 = frequency in microHz, Col2 = psd
    sfile = glob.glob('../data/*{}*.pow'.format(kic))
    data = ascii.read(sfile[0]).to_pandas()

    # Read in the mode locs
    cop = pd.read_csv('../data/copper.csv',index_col=0)
    locs = cop[cop.KIC == str(kic)].Freq.values
    lo = locs.min() - .1*dnu
    hi = locs.max() + .1*dnu

    # Make the frequency range selection
    ff, pp = data['col1'],data['col2']
    sel = (ff > lo) & (ff < hi)
    tf = ff[~sel].values
    tp = pp[~sel].values

    #Make extra accomodations for KIC 347720
    if kic == 3427720:
        sel = (tf > 90.) & (tf < 400.)
        tf = tf[~sel]
        tp = tp[~sel]

    #Rebin the frequencies
    f, p = rebin(tf, tp, binsize=10)

    # Initiate the first guesses
    white = 1.
    p0 = first_guess(numax)

    data = {'N': len(f),
            'f': f, 'p': p,
            'numax_est': numax, 'white_est': white,
            'nyq_est': np.max(f),
            'scale_spread': 0.01}

    init = {'loga': p0[0], 'logb': p0[1],
            'logc': p0[2], 'logd': p0[3],
            'logj': p0[4], 'logk': p0[5],
            'lognumax': p0[6],
            'white': white, 'nyq': np.max(f),
            'scale': 1.}

    #Instead lets quickly plot the data for a test
    # pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
    # ax = pg.plot(scale='log')
    # plt.savefig(dir+'test.png')

    # Run stan
    run = run_stan(data, init, dir)
    fit = run()
