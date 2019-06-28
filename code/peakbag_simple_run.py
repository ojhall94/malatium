#!/usr/bin/env python3
#O. J. Hall 2019

import numpy as np
import matplotlib
import pandas as pd

import os
import pwd
import sys
import pickle

os.getlogin = lambda: pwd.getpwuid(os.getuid())[0]
if os.getlogin() != 'oliver':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import lightkurve as lk

import pymc3 as pm
import corner

import glob
import time
import lightkurve as lk
import astropy.units as u
from astropy.units import cds
from astropy.io import ascii
timestr = time.strftime("%m%d-%H%M")

import argparse
parser = argparse.ArgumentParser(description='Run our PyMC3 model')
parser.add_argument('iter', type=int, help='Number of iterations steps')
parser.add_argument('idx',type=int,help='Index on the kiclist')
args = parser.parse_args()

class model():
    def __init__(self, f, f0_, f1_, f2_):
        self.f = f
        self.npts = len(f)
        self.M = [len(f0_), len(f1_), len(f2_)]


    def epsilon(self, i, l, m):
    #I use the prescriptions from Gizon & Solank 2003 and Handberg & Campante 2012
        if l == 0:
            return 1
        if l == 1:
            if m == 0:
                return np.cos(i)**2
            if np.abs(m) == 1:
                return 0.5 * np.sin(i)**2
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

    def lor(self, freq, h, w):
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)

    def mode(self, l, freqs, hs, ws, i, split=0):
        for idx in range(self.M[l]):
            for m in range(-l, l+1, 1):
                self.mod += self.lor(freqs[idx] + (m*split),
                                     hs[idx] * self.epsilon(i, l, m),
                                     ws[idx])

    def model(self, p):
        b, f0, f1, f2, g0, g1, g2, h0, h1, h2, split, i = p
        self.mod = np.ones(self.npts) * b
        self.mode(0, f0, h0, g0, i)
        self.mode(1, f1, h1, g1, i, split)
        self.mode(2, f2, h2, g2, i, split)
        return self.mod

    def __call__(self, p):
        return self.model(p)

class run_pymc3:
    def __init__(self, mod, p, init, fs, fes, dir):
        '''Core PyStan class.
        Input __init__:
        mod : model class (with frequencies loaded in)

        p : observed power

        init : initial guesses

        fs : mode frequencies

        fes : error on mode frequencies

        dir : output directory
        '''
        self.mod = mod
        self.p = p
        self.init = init
        self.fs = fs
        self.fes = fes
        self.dir = dir

    def build_model(self):
        print('Building the model')
        self.pm_model = pm.Model()

        with self.pm_model:
            b = pm.HalfNormal('b', sigma=2.0, testval=self.init[0])

            f0 = pm.Normal('f0', mu=self.fs[0], sigma=self.fes[0], testval=self.fs[0], shape=len(self.fs[0]))
            f1 = pm.Normal('f1', mu=self.fs[1], sigma=self.fes[1], testval=self.fs[1], shape=len(self.fs[1]))
            f2 = pm.Normal('f2', mu=self.fs[2], sigma=self.fes[2], testval=self.fs[2], shape=len(self.fs[2]))

            g0 = pm.HalfNormal('g0', sigma=2.0, testval=self.init[4], shape=len(self.init[4]))
            g1 = pm.HalfNormal('g1', sigma=2.0, testval=self.init[5], shape=len(self.init[5]))
            g2 = pm.HalfNormal('g2', sigma=2.0, testval=self.init[6], shape=len(self.init[6]))

            a0 = pm.HalfNormal('a0', sigma=20., testval=self.init[7], shape=len(self.init[7]))
            a1 = pm.HalfNormal('a1', sigma=20., testval=self.init[8], shape=len(self.init[8]))
            a2 = pm.HalfNormal('a2', sigma=20., testval=self.init[9], shape=len(self.init[9]))

            h0 = pm.Deterministic('h0', 2*a0**2/np.pi/g0)
            h1 = pm.Deterministic('h1', 2*a1**2/np.pi/g1)
            h2 = pm.Deterministic('h2', 2*a2**2/np.pi/g2)

            xsplit = pm.HalfNormal('xsplit', sigma=2.0, testval=self.init[10])
            cosi = pm.Uniform('cosi', 0., 1.)

            i = pm.Deterministic('i', np.arccos(cosi))
            split = pm.Deterministic('split', xsplit/pm.math.sin(i))

            fit = self.mod([b, f0, f1, f2, g0, g1, g2, h0, h1, h2, split, i])

            like = pm.Gamma('like', alpha=1, beta=1.0/fit, observed=self.p)

    def run_pymc3(self):
        '''Runs PyMC3'''
        print('Running the model')
        with self.pm_model:
            self.trace = pm.sample(tune=int(args.iter/2),
                                    draws=int(args.iter/2),
                                    chains=4)

    def out_corner(self):
        labels=['xsplit','cosi','b','i','split']
        chain = np.array([self.trace[label] for label in labels])
        verbose = [r'$\delta\nu_s^*$',r'$\cos(i)$',r'$b$',r'$i$',r'$\delta\nu_{\rm s}$']
        corner.corner(chain.T, labels=verbose, quantiles=[0.16, 0.5, 0.84]
                      ,show_titles=True)
        plt.savefig(self.dir+'corner.png')
        plt.close('all')

    def out_traceplot(self):
        pm.traceplot(self.trace,
                    var_names=['b','xsplit','cosi',
                                'i','split',
                                'g0','g1','g2',
                                'a0','a1','a2'])
        plt.savefig(self.dir+'stanplot.png')
        plt.close('all')

    def out_modelplot(self):
        labels=['b','f0','f1','f2','g0','g1','g2','h0','h1','h2','split','i']
        res = np.array([np.median(self.trace[label],axis=0) for label in labels])

        pg = lk.Periodogram(self.mod.f*u.microhertz, self.p*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.5, label='Data')
        plt.plot(self.mod.f, self.mod(res), linewidth=2, label='Model')
        plt.legend()

        plt.savefig(self.dir+'modelplot.png')
        plt.close('all')

    def out_csv(self):
        df = pm.backends.tracetab.trace_to_dataframe(self.trace)
        df.to_csv(self.dir+'chains.csv')

    def __call__(self):
        self.build_model()
        self.run_pymc3()
        self.out_csv()
        self.out_corner()
        self.out_traceplot()
        self.out_modelplot()
        pm.summary(self.trace).to_csv(self.dir+'summary.csv')

        print('Run complete!')

def harvey(f, a, b, c):
    harvey = 0.9*a**2/b/(1.0 + (f/b)**c);
    return harvey

def get_apodization(freqs, nyquist):
    x = (np.pi * freqs) / (2 * nyquist)
    return (np.sin(x)/x)**2

def get_background(f, a, b, c, d, j, k, white, scale, nyq):
    background = np.zeros(len(f))
    background += get_apodization(f, nyq) * scale\
                    * (harvey(f, a, b, 4.) + harvey(f, c, d, 4.) + harvey(f, j, k, 2.))\
                    + white
    return background

def gaussian(locs, l, numax, Hmax0):
    fwhm = 0.25 * numax
    std = fwhm/2.355

    Vl = [1.0, 1.22, 0.71, 0.14]

    return Hmax0 * Vl[l] * np.exp(-0.5 * (locs - numax)**2 / std**2)

def get_folder(kic):
    fol = '/rds/projects/2018/daviesgr-asteroseismic-computation/ojh251/malatium/peakbag/'+str(kic)
    if not os.path.exists(fol):
        os.makedirs(fol)
    return fol + '/' + timestr +'_idx'+str(idx)+'_'+str(kic)+'_peakbag_'

if __name__ == '__main__':
    idx = int(args.idx)

    #Get the star data
    mal = pd.read_csv('../data/malatium.csv', index_col=0)
    star = mal.loc[idx]
    kic = star.KIC
    numax = star.numax
    dnu = star.dnu

    #Get the output director
    if os.getlogin() == 'oliver':
        dir = 'tests/output_fmr/'
    else:
        dir = get_folder(kic)

    # Get the power spectrum
    # Col1 = frequency in microHz, Col2 = psd
    sfile = glob.glob('../data/*{}*.pow'.format(kic))
    data = ascii.read(sfile[0]).to_pandas()

    # Read in the mode locs
    cop = pd.read_csv('../data/copper.csv',index_col=0)
    cop = cop[cop.l != 3]
    locs = cop[cop.KIC == str(kic)].Freq.values#[27:33]
    elocs = cop[cop.KIC == str(kic)].e_Freq.values#[27:33]
    modeids = cop[cop.KIC == str(kic)].l.values#[27:33]

    lo = locs.min() - .25*dnu
    hi = locs.max() + .25*dnu

    # Make the frequency range selection
    ff, pp = data['col1'],data['col2']
    sel = (ff > lo) & (ff < hi)
    f = ff[sel].values
    pf = pp[sel].values

    #Divide out the background
    try:
        if os.getlogin() == 'oliver':
            backdir = glob.glob('/home/oliver/PhD/mnt/RDS/malatium/backfit/'
                                +str(kic)+'/*_fit.pkl')[0]
        else:
            backdir = glob.glob('/rds/projects/2018/daviesgr-asteroseismic-computation/ojh251/malatium/backfit/'
                                +str(kic)+'/*_fit.pkl')[0]
        with open(backdir, 'rb') as file:
            backfit = pickle.load(file)

        labels=['loga','logb','logc','logd','logj','logk','white','scale','nyq']
        res = np.array([np.median(backfit[label]) for label in labels])
        res[0:6] = 10**res[0:6]

        bgmodel = get_background(f, *res)
        p = pf / bgmodel
    except IndexError:
        pg = lk.periodogram.SNRPeriodogram(f*u.microhertz, pf*(cds.ppm**2/u.microhertz))
        p = pg.flatten().power.value * 2

    # Set up the data
    f0_ = locs[modeids==0]
    f1_ = locs[modeids==1]
    f2_ = locs[modeids==2]
    f0_e = elocs[modeids==0]
    f1_e = elocs[modeids==1]
    f2_e = elocs[modeids==2]

    init = [1.,                          # Background
           f0_,                         # l0 modes
           f1_,                         # l1 modes
           f2_,                         # l2 modes
           np.ones(len(f0_)) * 2.0,     # l0 widths
           np.ones(len(f1_)) * 2.0,     # l1 widths
           np.ones(len(f2_)) * 2.0,     # l2 widths
           np.sqrt(gaussian(f0_, 0, numax, 15.) * 2.0 * np.pi / 2.0), # l0 amps
           np.sqrt(gaussian(f1_, 1, numax, 15.) * 2.0 * np.pi / 2.0), # l1 amps
           np.sqrt(gaussian(f2_, 2, numax, 15.) * 2.0 * np.pi / 2.0), # l2 amps
           1.0 * np.sin(np.pi/2),       # projected splitting
           np.pi/2.]

    mod = model(f, f0_, f1_, f2_)

    #Plot the data
    pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
    ax = pg.plot(alpha=.5)
    ax.scatter(locs, [15]*len(locs),c=modeids, s=20, edgecolor='k')
    ax.plot(f, mod(init), lw=2)
    plt.savefig(dir+'dataplot.png')
    plt.close()

    # Run stan
    print('About to go into Pymc3')
    run = run_pymc3(mod, p, init,
                    [f0_, f1_, f2_], [f0_e, f1_e, f2_e],
                    dir)
    run()
