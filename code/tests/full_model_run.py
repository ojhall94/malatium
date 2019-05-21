#!/usr/bin/env python3
#O. J. Hall 2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import lightkurve as lk
import os
import astropy.units as u
from astropy.units import cds
import corner
import time
timestr = time.strftime("%m%d-%H%M")

import argparse
parser = argparse.ArgumentParser(description='Run our PyStan model')
parser.add_argument('-f','--full', action='store_const',
                    const=True, default=False, help='Run on all modes.')
args = parser.parse_args()

__outdir__ = 'output_fmr/'+timestr+'_'
__iter__ = 1000

def create_model(overwrite=True):
    overwrite = True
    gpgamma = '''
    functions{
        vector lorentzian(real loc, int l, int m, vector f, real eps, real H, real w, real nus){
            return (eps * H) ./ (1 + (4/w^2) * square(f - loc + m*nus));
        }
    }
    data{
        int N;            // Number of data points
        int M;            // Number of modes
        vector[N] f;      // Frequency
        vector[N] p;      // Power
        real pr_locs[M]; // Mode locations (this will have to change for multiple n modes)
        int ids[M];   // The ID's of the modes
    }
    parameters{
        real logAmp[M];         // Mode amplitude in log space
        real logGamma[M];  // Mode linewidth in log space
        real locs[M];           // True mode locations
        real<lower=0> vsini;    //  Sin of angle of inclination x rotational splitting
        real<lower=0> vcosi;    //  Cos of angle of inclination x rotational splitting
        real<lower=0.1> b;      // Background
        real<lower=0.> alpha;   // Spread on the squared exponential kernel
        real<lower=0.> rho;     // The length scale of the GP Gamma prior

    }
    transformed parameters{
        real H[M];                // Mode height
        vector[M] w;     // Mode linewidth
        real i;          // Angle of inclination (rad)
        real<lower=0> nus;     // Rotational frequency splitting

        nus = sqrt(vsini^2 + vcosi^2); //Calculate the splitting
        i = acos(vcosi / nus);         // Calculate the inclination

        for (m in 1:M){
            w[m] = 10^logGamma[m];             // Transform log linewidth to linewidth
            H[m] = 10^logAmp[m] / pi() / w[m]; // Transform mode amplitude to mode height
        }
    }
    model{
        vector[N] modes; // Our Model
        matrix[4,4] eps; // Matrix of legendre polynomials
        matrix[M,M] gpw; // Covariance Matrix of the linewidths
        int l;           // The radial degree

        // First we'll calculate all the legendre polynomials for this i
        eps = rep_matrix(1., 4, 4);
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


        // Generating the GP prior on linewidth
        gpw = cov_exp_quad(locs, alpha, rho) + diag_matrix(rep_vector(1e-6, M));

        // Generating our model
        modes = rep_vector(b, N);
        for (mode in 1:M){        // Iterate over all modes passed in
            l = ids[mode];    // Identify the Mode ID
            for (m in -l:l){      // Iterate over all m in a given l
                modes += lorentzian(locs[mode], l, m, f, eps[l+1,abs(m)+1], H[mode], w[mode], nus);
            }
        }

        // Model drawn from a gamma distribution scaled to the model (Anderson+1990)
        p ~ gamma(1., 1../modes);

        //priors on the parameters
        logAmp ~ normal(1.5, 1.);
        locs ~ normal(pr_locs, 1);
        nus ~ normal(0.5, 1.);
        vsini ~ uniform(0,nus);


        alpha ~ normal(2.5, 1.);
        rho ~ normal(250., 50.);
        w ~ multi_normal(rep_vector(1.,M), gpw);

        b ~ normal(1.,.1);
    }
    '''
    model_path = 'gpgamma.pkl'
    if overwrite:
        print('Updating Stan model')
        sm = pystan.StanModel(model_code = gpgamma, model_name='gpgamma')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()
    if os.path.isfile(model_path):
        print('Reading in Stan model')
        sm = pickle.load(open(model_path, 'rb'))
    else:
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = gpgamma, model_name='gpgamma')
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

    def read_stan(self):
        '''Reads the existing stanmodel'''
        model_path = 'gpgamma.pkl'
        if os.path.isfile(model_path):
            sm = pickle.load(open(model_path, 'rb'))
        else:
            print('No stan model found')
            create_asterostan(overwrite=True)
            sys.exit()
        return sm

    def run_stan(self):
        '''Runs PyStan'''
        sm = self.read_stan()

        fit = sm.sampling(data = self.data,
                    iter= __iter__, chains=4, seed=1895,
                    init = [self.init, self.init, self.init, self.init])

        return fit

    def out_corner(self, fit):
        truths= [init['vsini'],init['vcosi'],init['i'],init['nus'],
                    np.nan, np.nan,np.nan]
        labels=['vsini','vcosi','i','nus', 'b', 'alpha', 'rho']
        chain = np.array([fit[label] for label in labels])

        verbose = [r'$\nu_{\rm s}\sin(i)$',r'$\nu_{\rm s}\cos(i)$',r'$i$',
                    r'$\nu_{\rm s}$', r'$b$', r'$\alpha$', r'$\rho$']
        corner.corner(chain.T, labels=verbose, quantiles=[0.16, 0.5, 0.84],
                    truths=truths,show_titles=True)

        plt.savefig(__outdir__+'corner.png')
        plt.close('all')

    def out_stanplot(self, fit):
        fit.plot(pars=['vsini','vcosi','i','nus','H','logAmp','w','alpha','rho'])
        plt.savefig(__outdir__+'stanplot.png')
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

        plt.savefig(__outdir__+'modelplot.png')
        plt.close('all')

    def _kernel(self, x, y, p):
        ''' Returns a sqaured exponetial covariance matrix '''
        # p[0] = sigma
        # p[1] = length scale
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
        ws = np.median(fit['w'], axis=0)
        ws_std = np.std(fit['w'],axis=0)
        flocs = np.median(fit['locs'], axis=0)
        alpha = np.median(fit['alpha'])
        rho = np.median(fit['rho'])

        npts = 500

        a = np.ones(len(flocs))
        c = np.ones(npts)

        flocs2 = np.linspace(np.min(flocs), np.max(flocs), npts)

        theta = [alpha, rho]
        ws_pred, sigmas = self._predict(flocs2, flocs, theta, a, c, ws, ws_std**2)

        fig, ax = plt.subplots(figsize=(12,8))
        ax = self._plot_GP(ax, flocs, flocs2, ws, ws_std, ws_pred, sigmas)
        plt.savefig(__outdir__+'gpplot.png')
        plt.close('all')

    def __call__(self):
        fit = self.run_stan()
        self.out_corner(fit)
        self.out_stanplot(fit)
        self.out_modelplot(fit)
        self.out_gpplot(fit)

        with open(__outdir__+'_summary.txt', "w") as text_file:
            print(fit.stansummary(),file=text_file)

        print('Run complete!')
        return fit

if __name__ == '__main__':
    # Get the locs
    locs = np.genfromtxt('locs.txt')
    mid = int(np.floor(len(locs)/2))
    l0s = locs[mid:mid+1,0]
    l2s = locs[mid-1:mid,2]
    modelocs = np.append(l0s, l2s)
    modeids = [0]*len(l0s)  + [2]*len(l2s)

    # Get ALL the locs
    if args.full:
        l0s = locs[:,0]
        l1s = locs[:,1]
        l2s = locs[:,2]
        modelocs = np.array([l0s, l1s, l2s]).flatten()
        modeids = [0]*len(l0s)  + [1]*len(l1s) + [2]*len(l2s)

    # Get the data range
    ff = np.genfromtxt('freqs.txt')
    pp = np.genfromtxt('model.txt')
    sel = [(ff >= np.min(modelocs)-25) & (ff <= np.max(modelocs+25))]
    f = ff[tuple(sel)]
    p = pp[tuple(sel)]

    # Set up the prior data
    nus = 0.411 #uHz
    i = np.deg2rad(56.) #rad
    dnu = 102. #uHz

    data = {'N':len(f),
            'M': len(modelocs),
            'f':f,
            'p':p,
            'pr_locs':modelocs,
            'ids':modeids}

    init = {'logAmp' :   np.ones(len(modelocs))*1.5,
            'logGamma': np.zeros(len(modelocs)),
            'w' : np.ones(len(modelocs)),
            'vsini' : nus*np.sin(i),
            'vcosi' : nus*np.cos(i),
            'i' : i,
            'nus': nus,
            'locs' : modelocs,
            'alpha':2.5,
            'rho':250.,
            'b':1.}

    # Run stan
    run = run_stan(data, init)
    fit = run()
