#!/usr/bin/env python3
#O. J. Hall 2019

import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import lightkurve as lk
import pystan
import astropy.units as u
import sys
from astropy.units import cds
import corner
import time
timestr = time.strftime("%m%d-%H%M")

import argparse
parser = argparse.ArgumentParser(description='Run our PyStan model')
parser.add_argument('iters', type=int, help='Number of MCMC iterations in PyStan.')
args = parser.parse_args()

# __outdir__ = 'output_fmr/'+timestr+'_'
__outdir__ = timestr+'_backfit_'
__iter__ = args.iters

def create_model(overwrite=True):
    overwrite = False
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
        real p[N];
        real white_est;
        real nyq_est;
        real numax_est;
        real scale_spread;
    }
    parameters {
        real<lower = 0> a;
        real<lower = 0> b;
        real<lower = 0> c;
        real<lower = 0> d;
        real<lower = 0> j;
        real<lower = 0> k;
        real<lower = 0> numax;
        real<lower = 0> white;
        real<lower = 0> nyq;
        real scale;
    }
    transformed parameters {
        real loga;
        real logb;
        real logc;
        real logd;
        real logj;
        real logk;
        real log_ac;
        real lognumax;

        loga = log10(a);
        logb = log10(b);
        logc = log10(c);
        logd = log10(d);
        logj = log10(j);
        logk = log10(k);
        log_ac = loga - logc;

        lognumax = log10(numax);

    }
    model {
        real beta[N];
        for (i in 1:N){
            beta[i] = 1. / (apod(f[i], nyq) * scale
                    * (harvey(f[i], a, b, 4.0)
                    + harvey(f[i], c, d, 4.0)
                    + harvey(f[i], j, k, 2.0))
                    + white);
            }
        p ~ gamma(1., beta);

        numax ~ normal(numax_est, numax_est*0.1);
        white ~ normal(white_est, white_est*0.3);

        loga ~ normal(3.4 + lognumax * -0.48, 0.3);
        logb ~ normal(-0.43 + lognumax * 0.86, 0.3);
        logc ~ normal(3.59 + lognumax * -0.59, 0.3);
        logd ~ normal(0.02 + lognumax * 0.96, 0.3);
        logj ~ normal(loga-1, 1.2);
        logk ~ normal(logb-1, 0.2);
        nyq ~ normal(nyq_est, nyq_est*0.01);
        scale ~ normal(1, scale_spread);
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

    return [a, b, c, d, j, k, numax]

class run_stan:
    def __init__(self, data, init):
        '''Core PyStan class.
        Input __init__:
        dat (dict): Dictionary of the data in pystan format.

        init (dict): Dictionary of initial guesses in pystan format.
        '''
        self.data = data
        self.init = init

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
                    iter= __iter__, chains=4, seed=1895,
                    init = [self.init, self.init, self.init, self.init])

        return fit

    def out_corner(self, fit):
        labels=['a','b','c','d','j','k','white','scale','nyq']
        truths = [init['a'], init['b'], init['c'], init['d'], init['j'],
                init['k'], init['white'], init['scale'], init['nyq']]
        verbose=[r'$a$',r'$b$',r'$c$',r'$d$',r'$j$',r'$k$','white','scale',
        r'$\nu_{\rm nyq}$']

        chain = np.array([fit[label] for label in labels])

        corner.corner(chain.T, labels=verbose, quantiles=[0.16, 0.5, 0.84],
                    truths=truths,show_titles=True)

        plt.savefig(__outdir__+'corner.png')
        plt.close('all')

    def out_stanplot(self, fit):
        fit.plot()
        plt.savefig(__outdir__+'stanplot.png')
        plt.close('all')

    def harvey(self, f, a, b, c):
        harvey = 0.9*a**2/b/(1.0 + (f/b)**c);
        return harvey

    def get_apodization(self, freqs, nyquist):
        x = (np.pi * freqs) / (2 * nyquist)
        return (np.sin(x)/x)**2

    def get_background(self, f, a, b, c, d, j, k, white, scale, nyq):
        background = np.zeros(len(f))
        background += get_apodization(f, nyq)**2 * scale\
                        * (harvey(f, a, b, 4.) + harvey(f, c, d, 4.) + harvey(f, j, k, 2.))\
                        + white
        return background

    def out_modelplot(self, fit):
        labels=['a','b','c','d','j','k','white','scale','nyq']
        res = [np.median(fit[label]) for label in labels]
        model = get_background(f, *res)

        pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.25, label='Data', scale='log')
        ax.plot(f, model, label='Model')
        ax.plot(f, harvey(f, *res[0:2], 4.), label='Harvey 1', ls=':')
        ax.plot(f, harvey(f, *res[2:4], 4.), label='Harvey 2', ls=':')
        ax.plot(f, harvey(f, *res[4:6], 2.), label='Harvey 3', ls=':')
        ax.plot(f, get_apodization(f, f[-1]), label='Apod', ls='--')
        ax.plot(f, res[-3]*np.ones_like(f), label='white',ls='-.')
        plt.legend(fontsize=10)

        plt.savefig(__outdir__+'modelplot.png')
        plt.close('all')

    def __call__(self):
        fit = self.run_stan()
        self.out_corner(fit)
        self.out_stanplot(fit)
        self.out_modelplot(fit)

        with open(__outdir__+'_summary.txt', "w") as text_file:
            print(fit.stansummary(),file=text_file)

        print('Run complete!')
        return fit

if __name__ == '__main__':
    # Get the locs
    locs = np.genfromtxt('../locs.txt')
    lo = locs.flatten().min() - 50.
    hi = locs.flatten().max() + 50.

    # Get the data range
    ff = np.genfromtxt('../freqs.txt')
    pp = np.genfromtxt('../model.txt')
    sel = (ff > lo) & (ff < hi)
    f = ff[~sel]
    p = pp[~sel]

    numax = 2200
    white = 1.
    p0 = first_guess(numax)

    data = {'N': len(f),
            'f': f,
            'p': p,
            'numax_est': numax,
            'white_est': white,
            'nyq_est': np.max(f),
            'scale_spread': 0.01}

    init = {'a': p0[0], 'b': p0[1],
            'c': p0[2], 'd': p0[3],
            'j': p0[4], 'k': p0[5],
            'numax': p0[6],
            'white': white,
            'nyq': np.max(f),
            'scale': 0.7}

    # Run stan
    run = run_stan(data, init)
    fit = run()
