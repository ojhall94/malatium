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
    overwrite = True
    malatium = '''
    functions{
        vector lorentzian(real loc, int l, int m, vector f, real eps, real H, real w, real nus){
            return (eps * H) ./ (1 + (4/w^2) * square(f - loc + m*nus));
        }
        real harvey(real f, real a, real b, real c){
            return 0.9*a^2/b/(1.0 + (f/b)^c);
        }
        real apod(real f, real nyq){
            real x = 3.14 / 2.0 * f / nyq;
            return (sin(x) / x)^2;
        }
        real background(real f, real a, real b, real c, real d, real j, real k,
                        real numax, real white, real nyq, real scale){
            return (apod(f, nyq) * scale
                    * (harvey(f, a, b, 4.0)
                    + harvey(f, c, d, 4.0)
                    + harvey(f, j, k, 2.0))
                    + white);
        }
    }
    data{
        int N;                   // Number of data points
        int M;                   // Number of modes
        vector[N] f;             // Frequency
        vector[N] p;             // Power
        real pr_locs[M];         // Mode locations (this will have to change for multiple n modes)
        real e_locs[M];          // Uncertainty on the mode locations
        int ids[M];              // The ID's of the modes
        real rho;                // The length scale of the GP Gamma prior
        vector[10] pr_phi;         // The initial guesses for the background parameters
        cov_matrix[10] sigphi;   // The covariance of the background parameters
    }
    transformed data{
        matrix[10,10] L_sigphi = cholesky_decompose(sigphi);
    }
    parameters{
        real logAmp[M];          // Mode amplitude in log space
        vector[M] logGamma;      // Mode linewidth in log space
        real locs[M];            // True mode locations
        real<lower=0> vsini;     //  Sin of angle of inclination x rotational splitting
        real<lower=0> vcosi;     //  Cos of angle of inclination x rotational splitting
        real<lower=0.> alpha;    // Spread on the squared exponential kernel
        vector[10] phi;            // The background parameters
    }
    transformed parameters{
        real numax = 10^phi[7];                // Background parameters
        real<lower=0> logac = phi[1] - phi[3]; // Background parameters
        real<lower=0> logdb = phi[4] - phi[2]; // Background parameters
        real H[M];                             // Mode height
        real w[M];                             // Mode linewidth
        real i;                                // Angle of inclination (rad)
        real<lower=0> nus;                     // Rotational frequency splitting
        matrix[M, M] gpG = cov_exp_quad(pr_locs, alpha, rho)
                            +diag_matrix(rep_vector(1e-10, M));
        matrix[M, M] LgpG = cholesky_decompose(gpG);


        nus = sqrt(vsini^2 + vcosi^2);         //Calculate the splitting
        i = acos(vcosi / nus);                 // Calculate the inclination

        for (m in 1:M){
            w[m] = 10^logGamma[m];             // Transform log linewidth to linewidth
            H[m] = 10^logAmp[m] / pi() / w[m]; // Transform mode amplitude to mode height
        }
    }
    model{
        real a = 10^phi[1];          // Caculate the linear background parameters
        real b = 10^phi[2];
        real c = 10^phi[3];
        real d = 10^phi[4];
        real j = 10^phi[5];
        real k = 10^phi[6];
        vector[N] modes;             // Our Model
        matrix[4,4] eps;             // Matrix of legendre polynomials
        int l;                       // The radial degree
        real nus_mu = 0.5;            // Circumventing a Stan problem

        eps = rep_matrix(1., 4, 4);  // Calculate all the legendre polynomials for this i
        eps[0+1,0+1] = 1.;
        eps[1+1,0+1] = cos(i)^2;
        eps[1+1,1+1] = 0.5 * sin(i)^2;
        eps[2+1,0+1] = 0.25 * (3. * cos(i)^2 - 1.)^2;
        eps[2+1,1+1] = (3./8.)*sin(2*i)^2;
        eps[2+1,2+1] = (3./8.) * sin(i)^4;
        eps[3+1,0+1] = (1./64.)*(5.*cos(3.*i) + 3.*cos(i))^2;
        eps[3+1,1+1] = (3./64.)*(5.*cos(2.*i) + 3.)^2 * sin(i)^2;
        eps[3+1,2+1] = (15./8.)*cos(i)^2 * sin(i)^4;
        eps[3+1,3+1] = (5./16.)*sin(i)^6;


        // Generating our model
        for (n in 1:N){
            modes[n] = background(f[n], a, b, c, d, j, k, numax, phi[8], phi[9], phi[10]);
        }

        for (mode in 1:M){        // Iterate over all modes passed in
            l = ids[mode];        // Identify the Mode ID
            for (m in -l:l){      // Iterate over all m in a given l
                modes += lorentzian(locs[mode], l, m, f, eps[l+1,abs(m)+1], H[mode], w[mode], nus);
            }
        }

        // Model drawn from a gamma distribution scaled to the model (Anderson+1990)
        p ~ gamma(1., 1../modes);

        //priors on the parameters
        logAmp ~ normal(1.5, 1.);
        locs ~ normal(pr_locs, e_locs);
        nus_mu ~ normal(nus, 1.);
        vsini ~ uniform(0,nus);

        alpha ~ normal(0.3, .5);
        logGamma ~ multi_normal_cholesky(rep_vector(0., M), LgpG);

        phi ~ multi_normal_cholesky(pr_phi, L_sigphi);
        logac ~ lognormal(1., 1.);
        logdb ~ lognormal(1., 1.);
    }
    '''
    model_path = 'malatium.pkl'
    if overwrite:
        print('Updating Stan model')
        sm = pystan.StanModel(model_code = malatium, model_name='malatium')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()
    if os.path.isfile(model_path):
        print('Reading in Stan model')
        sm = pickle.load(open(model_path, 'rb'))
    else:
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = malatium, model_name='malatium')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()

class run_stan:
    def __init__(self, data, init):
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
        model_path = 'gpgamma.pkl'
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
        labels=['vsini','vcosi','i','nus', 'alpha']
        verbose = [r'$\nu_{\rm s}\sin(i)$',r'$\nu_{\rm s}\cos(i)$',r'$i$',
                    r'$\nu_{\rm s}$', r'$\alpha$']

        chain = np.array([fit[label] for label in labels])
        corner.corner(chain.T, labels=verbose, quantiles=[0.16, 0.5, 0.84],
                    truths=truths,show_titles=True)

        plt.savefig(self.dir+'corner.png')
        plt.close('all')

    def out_stanplot(self, fit):
        fit.plot(pars=['vsini','vcosi','i','nus','H','logAmp','logGamma','alpha'])
        plt.savefig(self.dir+'stanplot.png')
        plt.close('all')

    def _get_epsilon(self, i, l, m):
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

    def _lorentzian(self, f, l, m, loc, i, H, w, nus):
        eps = self._get_epsilon(i,l,m)
        model = eps * H / (1 + (4/w**2)*(f - loc + m * nus)**2)
        return model

    def out_modelplot(self, fit):
        model = np.ones(len(self.data['f']))
        nus = np.median(fit['nus'])
        for mode in range(len(self.data['ids'])):
            l = self.data['ids'][mode]
            for m in range(-l, l+1):
                loc = np.median(fit['locs'].T[mode])
                H = np.median(fit['H'].T[mode])
                w = np.median(fit['w'].T[mode])
                model += self._lorentzian(f, l, m, loc, i, H, w, nus)
        fitlocs = np.median(fit['locs'],axis=0)

        pg = lk.Periodogram(data['f']*u.microhertz, data['p']*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.5, label='Data')
        plt.scatter(fitlocs, [15]*len(fitlocs),c='k',s=25, label='fit locs')
        plt.scatter(data['pr_locs'], [15]*len(data['pr_locs']),c='r',s=5, label='true locs')
        plt.plot(data['f'], model, linewidth=1, label='Model')
        plt.legend()

        plt.savefig(self.dir+'modelplot.png')
        plt.close('all')

    def _kernel(self, x, y, p):
        return p[0]**2 * np.exp(-0.5 * np.subtract.outer(x, y)**2 / p[1]**2)

    def _predict(self, t_2, t_1, theta, a, c, y_1, y_v):
        B = self._kernel(t_1, t_2, theta).T
        A = self._kernel(t_1, t_1, theta).T + np.diag(y_v)
        C = self._kernel(t_2, t_2, theta).T

        y = c + np.dot(np.dot(B, np.linalg.inv(A)), (y_1 - a))
        Sigma = C - np.dot(np.dot(B, np.linalg.inv(A)),B.T)

        y_pred = y
        sigma_new = np.sqrt(np.diagonal(Sigma))
        return y_pred, sigma_new

    def _plot_GP(self, ax, t_1, t_2, y_1, s, y_pred, sigmas, label='Observation'):
        ax.fill_between(t_2, y_pred-sigmas, y_pred+sigmas, alpha=.5, color='#8d44ad')
        ax.plot(t_2, y_pred, c='k')
        ax.errorbar(t_1, y_1, yerr=s, fmt='o', capsize=0, label=label)
        ax.legend(fontsize=15)
        ax.set_ylabel(r'Linewidth [$\mu Hz$]', fontsize=20)
        ax.set_xlabel(r'Frequency [$\mu Hz$]', fontsize=20)
        ax.legend(fontsize=20)
        return ax

    def out_gpplot(self, fit):
        ws = np.median(fit['logGamma'], axis=0)
        ws_std = np.std(fit['logGamma'],axis=0)
        flocs = np.median(fit['locs'], axis=0)
        alpha = np.median(fit['alpha'])
        rho = data['rho']

        truths = np.genfromtxt('../scripts/lws.txt')

        npts = 500

        a = np.zeros(len(flocs))
        c = np.zeros(npts)

        flocs2 = np.linspace(np.min(flocs), np.max(flocs), npts)

        theta = [alpha, rho]
        ws_pred, sigmas = predict(flocs2, flocs, kernel, theta, a, c, ws, ws_std**2)

        fig, ax = plt.subplots(figsize=(12,8))

        ax = plot_GP(ax, flocs, flocs2, ws, ws_std, ws_pred, sigmas)
        ax.plot(locs.flatten(), np.log10(truths.flatten()), c='r', alpha=.5,lw=5, label='Truth')
        ax.set_xlim(flocs.min()-5*d02, flocs.max()+5*d02)
        plt.savefig(self.dir+'gpplot.png')
        plt.close('all')

    def _harvey(self, f, a, b, c):
        harvey = 0.9*a**2/b/(1.0 + (f/b)**c);
        return harvey

    def _get_apodization(self, freqs, nyquist):
        x = (np.pi * freqs) / (2 * nyquist)
        return (np.sin(x)/x)**2

    def _get_background(self, f, a, b, c, d, j, k, white, numax, scale, nyq):
        background = np.zeros(len(f))
        background += self._get_apodization(f, nyq) * scale\
                        * (self._harvey(f, a, b, 4.) + self._harvey(f, c, d, 4.) + self._harvey(f, j, k, 2.))\
                        + white
        return background

    def out_backplot(self, fit):
        res = np.median(fit['phi'],axis=0)
        res[0:6] = 10**res[0:6]
        model = self._get_background(f, *res)

        pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.25, label='Data', scale='log')
        ax.plot(f, model, label='Model')
        ax.plot(f, self._harvey(f, res[0],res[1], 4.), label='Harvey 1', ls=':')
        ax.plot(f, self._harvey(f, res[2],res[3], 4.), label='Harvey 2', ls=':')
        ax.plot(f, self._harvey(f, res[4],res[5], 2.), label='Harvey 3', ls=':')
        ax.plot(f, self._get_apodization(f, f[-1]), label='Apod', ls='--')
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
        self.out_gpplot(fit)
        self.out_backplot(fit)

        with open(self.dir+'_summary.txt', "w") as text_file:
            print(fit.stansummary(),file=text_file)

        print('Run complete!')
        return fit

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
    d02 = star.d02

    #Get the output director
    dir = get_folder(kic)

    # Get the power spectrum
    # Col1 = frequency in microHz, Col2 = psd
    sfile = glob.glob('../data/*{}*.pow'.format(kic))
    data = ascii.read(sfile[0]).to_pandas()

    # Read in the mode locs
    cop = pd.read_csv('../data/copper.csv',index_col=0)
    locs = cop[cop.KIC == str(kic)].Freq.values
    elocs = cop[cop.KIC == str(kic)].e_Freq.values
    modeids =
    lo = locs.min() - .1*dnu
    hi = locs.max() + .1*dnu

    # Make the frequency range selection
    ff, pp = data['col1'],data['col2']
    sel = (ff > lo) & (ff < hi)
    f = ff[~sel].values
    p = pp[~sel].values

    #Read in backfit information
    backdir = glob.glob('/rds/projects/2018/daviesgr-asteroseismic-computation/ojh251/malatium/backfit/'
                        +str(kic)+'/*idx'+str(idx)+'*.pkl')[0]
    with open(backdir, 'rb') as file:
        backfit = pickle.load(file)

    labels=['loga','logb','logc','logd','logj','logk','lognumax','white','nyq','scale']
    pr_phi = np.array([np.median(backfit[label]) for label in labels])
    bf = pd.DataFrame(backfit)[labels]
    sigphi = np.cov(bf.T)

    data = {'N':len(f),
            'M': len(locs),
            'f':f,
            'p':p,
            'pr_locs':locs,
            'e_locs':elocs,
            'ids':modeids,
            'rho':d02 * 5,
            'pr_phi':pr_phi,
            'sigphi':sigphi}

    init = {'logAmp' :   np.ones(len(modelocs))*1.5,
            'logGamma': np.zeros(len(modelocs)),
            'vsini' : nus*np.sin(i),
            'vcosi' : nus*np.cos(i),
            'i' : 45.,
            'nus': .5,
            'locs' : modelocs,
            'alpha':.3,
            'phi':pr_phi}

    # Run stan
    run = run_stan(data, init)
    fit = run()
