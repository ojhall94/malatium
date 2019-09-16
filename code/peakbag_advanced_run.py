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
import pymc3 as pm
from pymc3.gp.util import plot_gp_dist
import theano.tensor as tt
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
    def __init__(self, f, n0_, n1_, n2_, deltanu_):
        self.f = f
        self.n0 = n0_
        self.n1 = n1_
        self.n2 = n2_
        self.npts = len(f)
        self.M = [len(n0_), len(n1_), len(n2_)]
        self.deltanu = deltanu_

    def epsilon(self, i):
        eps = tt.zeros((3,3))
        eps0 = tt.set_subtensor(eps[0][0], 1.)
        eps1 = tt.set_subtensor(eps[1][0], tt.sqr(tt.cos(i)))
        eps1 = tt.set_subtensor(eps1[1], 0.5 * tt.sqr(tt.sin(i)))
        eps2 = tt.set_subtensor(eps[2][0], 0.25 * tt.sqr((3. * tt.sqr(tt.cos(i)) - 1.)))
        eps2 = tt.set_subtensor(eps2[1], (3./8.) * tt.sqr(tt.sin(2*i)))
        eps2 = tt.set_subtensor(eps2[2], (3./8.) * tt.sin(i)**4)

        eps = tt.set_subtensor(eps[0], eps0)
        eps = tt.set_subtensor(eps[1], eps1)
        eps = tt.set_subtensor(eps[2], eps2)

        return eps

    def lor(self, freq, h, w):
        return h / (1.0 + 4.0/tt.sqr(w)*tt.sqr((self.f - freq)))

    def mode(self, l, freqs, hs, ws, eps, split=0):
        for idx in range(self.M[l]):
            for m in range(-l, l+1, 1):
                self.modes += self.lor(freqs[idx] + (m*split),
                                     hs[idx] * eps[l,abs(m)],
                                     ws[idx])

    def model(self, p, theano=True):
        f0, f1, f2, g0, g1, g2, h0, h1, h2, split, i, phi = p

        # Unpack background parameters
        loga = phi[0]
        logb = phi[1]
        logc = phi[2]
        logd = phi[3]
        logj = phi[4]
        logk = phi[5]
        white = phi[6]
        scale = phi[7]
        nyq = phi[8]

        # Calculate the modes
        eps = self.epsilon(i)
        self.modes = np.zeros(self.npts)
        self.mode(0, f0, h0, g0, eps)
        self.mode(1, f1, h1, g1, eps, split)
        self.mode(2, f2, h2, g2, eps, split)
        self.modes *= self.get_apodization(nyq)

        #Calculate the background
        self.back = self.get_background(loga, logb, logc, logd, logj, logk,
                                       white, scale, nyq)

        #Create the model
        self.mod = self.modes + self.back
        if theano:
            return self.mod
        else:
            return self.mod.eval()

    # Small separations are fractional
    def asymptotic(self, n, numax, alpha, epsilon, d=0.):
        nmax = (numax / self.deltanu) - epsilon
        curve = (alpha/2.)*(n-nmax)*(n-nmax)
        return (n + epsilon + d + curve) * self.deltanu

    def f0(self, p):
        numax, alpha, epsilon, d01, d02 = p

        return self.asymptotic(self.n0, numax, alpha, epsilon, 0.)

    def f1(self, p):
        numax, alpha, epsilon, d01, d02 = p

        return self.asymptotic(self.n1, numax, alpha, epsilon, d01)

    def f2(self, p):
        numax, alpha, epsilon, d01, d02 = p

        return self.asymptotic(self.n2+1, numax, alpha, epsilon, -d02)

    def gaussian(self, freq, numax, w, A):
        return A * tt.exp(-0.5 * tt.sqr((freq - numax)) / tt.sqr(w))

    def A0(self, f, p, theano=True):
        numax, w, A, V1, V2 = p
        height = self.gaussian(f, numax, w, A)
        if theano:
            return height
        else:
            return height.eval()

    def A1(self, f, p, theano=True):
        numax, w, A, V1, V2 = p
        height = self.gaussian(f, numax, w, A)*V1
        if theano:
            return height
        else:
            return height.eval()

    def A2(self, f, p, theano=True):
        numax, w, A, V1, V2 = p
        height = self.gaussian(f, numax, w, A)*V2
        if theano:
            return height
        else:
            return height.eval()

    def harvey(self, a, b, c):
        harvey = 0.9*tt.sqr(a)/b/(1.0 + tt.pow((self.f/b), c))
        return harvey

    def get_apodization(self, nyquist):
        x = (np.pi * self.f) / (2 * nyquist)
        return tt.sqr((tt.sin(x)/x))

    def get_background(self, loga, logb, logc, logd, logj, logk, white, scale, nyq):
        background = np.zeros(len(self.f))
        background += self.get_apodization(nyq) * scale                          * (self.harvey(tt.pow(10, loga), tt.pow(10, logb), 4.)                         +  self.harvey(tt.pow(10, logc), tt.pow(10, logd), 4.)                         +  self.harvey(tt.pow(10, logj), tt.pow(10, logk), 2.))                        +  white
        return background


class run_pymc3:
    def __init__(self, mod, p, nf_, kic, phi_, phi_cholesky, dir):
        '''Core PyStan class.
        Input __init__:
        mod : model class (with frequencies loaded in)

        p : observed power

        nf_ : rescaled frequency for the GP

        dir : output directory
        '''
        self.mod = mod
        self.p = p
        self.nf_ = nf_
        self.kic = kic
        self.phi_ = phi_
        self.phi_cholesky = phi_cholesky
        self.dir = dir

    def get_first_guesses(self):
        """ First guesses go here"""
        mal = pd.read_csv('../data/malatium.csv', index_col=0)
        cad = pd.read_csv('../data/cadmium.csv', index_col=0)
        ben = pd.read_csv('../data/bendalloy.csv',index_col=0)
        cop = pd.read_csv('../data/copper.csv', index_col=0)

        cop = cop.loc[cop['KIC'] == str(self.kic)]
        kic = self.kic

        self.init = {}
        self.init['numax'] = mal.loc[mal['KIC'] == kic].numax.values[0]
        self.init['alpha'] = cad.loc[cad['KIC'] == kic].alpha.values[0]
        self.init['epsilon'] = cad.loc[cad['KIC'] == kic].epsilon.values[0]
        self.init['d01'] = cad.loc[cad['KIC'] == kic].d01.values[0]
        self.init['d02'] = cad.loc[cad['KIC'] == kic].d02.values[0]
        self.init['sigma0'] = cop.loc[cop.l == 0, 'e_Freq'].mean()
        self.init['sigma1'] = cop.loc[cop.l == 1, 'e_Freq'].mean()
        self.init['sigma2'] = cop.loc[cop.l == 2, 'e_Freq'].mean()

        self.init['m'] = ben.loc[ben.KIC == kic].m.values[0]
        self.init['c'] = ben.loc[ben.KIC == kic].c.values[0]
        self.init['rho'] = np.abs(ben.loc[ben.KIC == kic].rho.values[0])
        self.init['L'] = np.abs(ben.loc[ben.KIC == kic].L.values[0])

        self.init['w'] = (0.25 * mal.loc[mal['KIC'] == int(kic)].numax.values[0])/2.355
        self.init['A'] = np.sqrt(np.pi* np.nanmax(self.p) / 2)
        self.init['V1'] = 1.2
        self.init['V2'] = 0.7
        self.init['sigmaA'] = 0.5

        self.init['xsplit'] = 1.0 * np.sin(np.pi/4)
        self.init['cosi'] = np.cos(np.pi/4)

    def build_model(self):
        print('Building the model')
        self.pm_model = pm.Model()

        with self.pm_model:
            # Mode locations
            numax =  pm.Normal('numax', self.init['numax'], 10., testval = self.init['numax'])
            alpha =  pm.Normal('alpha', self.init['alpha'], 0.01, testval = self.init['alpha'])
            epsilon = pm.Normal('epsilon', self.init['epsilon'], 1., testval = self.init['epsilon'])
            d01     = pm.Normal('d01', self.init['d01'], 0.1, testval = self.init['d01'])
            d02     = pm.Normal('d02', self.init['d02'], 0.1, testval = self.init['d02'])

            sigma0 = pm.HalfCauchy('sigma0', 2., testval = self.init['sigma0'])
            sigma1 = pm.HalfCauchy('sigma1', 2., testval = self.init['sigma1'])
            sigma2 = pm.HalfCauchy('sigma2', 2., testval = self.init['sigma2'])

            f0 = pm.Normal('f0', mod.f0([numax, alpha, epsilon, d01, d02]), sigma0, shape=len(f0_))
            f1 = pm.Normal('f1', mod.f1([numax, alpha, epsilon, d01, d02]), sigma1, shape=len(f1_))
            f2 = pm.Normal('f2', mod.f2([numax, alpha, epsilon, d01, d02]), sigma2, shape=len(f2_))

            # Mode Linewidths
            m = pm.Normal('m', self.init['m'], 1., testval = self.init['m'])
            c = pm.Normal('c', self.init['c'], 1., testval = self.init['c'])
            rho = pm.Normal('rho', self.init['rho'], 0.1, testval = self.init['rho'])
            ls = pm.Normal('ls', self.init['L'], 0.1)

            mu = pm.gp.mean.Linear(coeffs=m, intercept=c)
            cov = tt.sqr(rho) * pm.gp.cov.ExpQuad(1, ls=ls)

            gp = pm.gp.Latent(cov_func = cov, mean_func=mu)
            lng = gp.prior('lng', X=self.nf_)

            g0 = pm.Deterministic('g0', tt.exp(lng)[0:len(f0_)])
            g1 = pm.Deterministic('g1', tt.exp(lng)[len(f0_):len(f0_)+len(f1_)])
            g2 = pm.Deterministic('g2', tt.exp(lng)[len(f0_)+len(f1_):])

            # Mode Amplitude & Height
            w = pm.Normal('w', self.init['w'], 10., testval=self.init['w'])
            A = pm.Normal('A', self.init['A'], 1., testval=self.init['A'])
            V1 = pm.Normal('V1', self.init['V1'], 0.1, testval=self.init['V1'])
            V2 = pm.Normal('V2', self.init['V2'], 0.1, testval=self.init['V2'])

            sigmaA = pm.HalfCauchy('sigmaA', 1., testval = self.init['sigmaA'])
            Da0 = pm.Normal('Da0',0, 1, shape=len(f0_))
            Da1 = pm.Normal('Da1',0, 1, shape=len(f1_))
            Da2 = pm.Normal('Da2',0, 1, shape=len(f2_))

            a0 = pm.Deterministic('a0', sigmaA * Da0 + mod.A0(f0_, [numax, w, A, V1, V2]))
            a1 = pm.Deterministic('a1', sigmaA * Da1 + mod.A1(f1_, [numax, w, A, V1, V2]))
            a2 = pm.Deterministic('a2', sigmaA * Da2 + mod.A2(f2_, [numax, w, A, V1, V2]))

            h0 = pm.Deterministic('h0', 2*tt.sqr(a0)/np.pi/g0)
            h1 = pm.Deterministic('h1', 2*tt.sqr(a1)/np.pi/g1)
            h2 = pm.Deterministic('h2', 2*tt.sqr(a2)/np.pi/g2)

            # Mode splitting
            xsplit = pm.HalfNormal('xsplit', sigma=2.0, testval = self.init['xsplit'])
            cosi = pm.Uniform('cosi', 0., 1., testval = self.init['cosi'])

            i = pm.Deterministic('i', tt.arccos(cosi))
            split = pm.Deterministic('split', xsplit/tt.sin(i))

            # Background treatment
            phi = pm.MvNormal('phi', mu=self.phi_, chol=self.phi_cholesky, testval=self.phi_, shape=len(self.phi_))

            # Construct model
            fit = mod.model([f0, f1, f2, g0, g1, g2, h0, h1, h2, split, i, phi])

            like = pm.Gamma('like', alpha=1., beta=1./fit, observed=self.p)


    def run_pymc3(self):
        '''Runs PyMC3'''
        print('Running the model')
        with self.pm_model:
            self.trace = pm.sample(tune = int(args.iter/2),
                                    draws = int(args.iter/2),
                                    chains = 4,
                                    init = 'adapt_diag',
                                    start = self.init,
                                    target_accept = 0.99)

    def out_corner(self):
        labels = ['numax','alpha','epsilon','d01','d02',    # Mode frequencies
                    'sigma0','sigma1','sigma2',             # Mode frequencies
                    'm','c','rho','ls',                     # Mode width
                    'w','A','V1','V2','sigmaA',             # Mode amplitude
                    'xsplit','cosi','split','i'             # Mode splitting
                    ]

        chain = np.array([self.trace[label] for label in labels])
        verbose = [r'$\nu_{\rm max}$', r'$\alpha$',r'$\epsilon$',r'$\delta_{01}$',r'$\delta{02}$',
                    r'$\sigma_0$', r'$\sigma_1$', r'$\sigma_2$',
                    r'$m$', r'$c$', r'$\rho$', r'$L$',
                    r'$w$', r'$A$', r'$V_1$', r'$V_2$', r'$\sigma_A$',
                    r'$\delta\nu_{\rm s}$', r'$\cos(i)$', r'$\nu_{\rm s}$', r'$i$']
        corner.corner(chain, labels=verbose, quantiles=[0.16, 0.5, 0.84]
                      ,show_titles=True)
        plt.savefig(self.dir+'corner.png')
        plt.close('all')

    def out_modelplot(self):
        labels=['f0','f1','f2',
                'g0','g1','g2',
                'h0','h1','h2',
                'split','i',
                'phi']
        res_m = np.array([np.median(self.trace[label],axis=0) for label in labels])

        pg = lk.Periodogram(self.mod.f*u.microhertz, self.p*(cds.ppm**2/u.microhertz))
        ax = pg.plot(alpha=.5, label='Data')
        plt.plot(self.mod.f, self.mod.model(res_m, theano=False), lw=3, label='Model')
        plt.legend()

        plt.savefig(self.dir+'modelplot.png')
        plt.close('all')

    def out_diagnostic_plots(self):
        with plt.style.context(lk.MPLSTYLE):
            fig, ax = plt.subplots()
            res = [np.median(self.trace[label]) for label in ['numax', 'w', 'A', 'V1','V2']]
            resls = [np.median(self.trace[label],axis=0) for label in ['a0','a1','a2']]
            resfs = [np.median(self.trace[label],axis=0) for label in ['f0', 'f1', 'f2']]

            ax.plot(resfs[0], self.mod.A0(f0_, res,theano=False), label='0 Trend',lw=2, zorder=1)
            ax.plot(resfs[1], self.mod.A1(f1_, res,theano=False), label='1 Trend',lw=2, zorder=1)
            ax.plot(resfs[2], self.mod.A2(f2_, res,theano=False), label='2 Trend',lw=2, zorder=1)

            ax.scatter(resfs[0], resls[0], marker='^',label='0 mod', s=10, zorder=3)
            ax.scatter(resfs[1], resls[1], marker='*',label='1 mod', s=10, zorder=3)
            ax.scatter(resfs[2], resls[2], marker='o',label='2 mod', s=10, zorder=3)

            ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.3))
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Amplitude')

            plt.savefig(self.dir + 'amplitudefit.png')
            plt.close()

            fig, ax = plt.subplots()
            res = [np.median(self.trace[label]) for label in ['numax', 'alpha', 'epsilon','d01','d02']]
            resls = [np.median(self.trace[label],axis=0) for label in ['f0','f1','f2']]

            ax.plot(self.mod.f0(res)%self.mod.deltanu, self.mod.n0, label='0 Trend',lw=2, zorder=1)
            ax.plot(self.mod.f1(res)%self.mod.deltanu, self.mod.n1, label='1 Trend',lw=2, zorder=1)
            ax.plot(self.mod.f2(res)%self.mod.deltanu, self.mod.n2, label='2 Trend',lw=2, zorder=1)

            ax.scatter(resls[0]%self.mod.deltanu, self.mod.n0, marker='^',label='0 mod', s=10, zorder=3)
            ax.scatter(resls[1]%self.mod.deltanu, self.mod.n1, marker='*',label='1 mod', s=10, zorder=3)
            ax.scatter(resls[2]%self.mod.deltanu, self.mod.n2, marker='o',label='2 mod', s=10, zorder=3)

            ax.set_xlabel(r'Frequency mod $\Delta\nu$')
            ax.set_ylabel('Overtone order n')
            ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.3))

            plt.savefig(self.dir + 'frequencyfit.png')
            plt.close()

            fig, ax = plt.subplots()
            resls = [np.median(self.trace[label],axis=0) for label in ['f0','f1','f2']]
            nflin = np.linspace(self.nf_.min(), self.nf_.max(), 100)

            plot_gp_dist(ax, self.trace['g0'], resls[0], palette='viridis', fill_alpha=.05)

            ax.scatter(resls[0], np.median(self.trace['g0'],axis=0), marker='^', label='mod', s=10,zorder=5)
            ax.scatter(resls[1], np.median(self.trace['g1'],axis=0), marker='*', label='mod 1', s=10,zorder=5)
            ax.scatter(resls[2], np.median(self.trace['g2'],axis=0), marker='o', label='mod 2', s=10,zorder=5)

            ax.errorbar(resls[0], np.median(self.trace['g0'],axis=0), yerr=np.std(self.trace['g0'],axis=0), fmt='|', c='k', lw=3, alpha=.5)
            ax.errorbar(resls[1], np.median(self.trace['g1'],axis=0), yerr=np.std(self.trace['g1'],axis=0), fmt='|', c='k', lw=3, alpha=.5)
            ax.errorbar(resls[2], np.median(self.trace['g2'],axis=0), yerr=np.std(self.trace['g2'],axis=0), fmt='|', c='k', lw=3, alpha=.5)

            ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))

            plt.savefig(self.dir + 'widthfit.png')
            plt.close()

    def out_csv(self):
        df = pm.backends.tracetab.trace_to_dataframe(self.trace)
        df.to_csv(self.dir+'chains.csv')

    def __call__(self):
        self.get_first_guesses()
        self.build_model()
        self.run_pymc3()
        self.out_csv()
        pm.summary(self.trace).to_csv(self.dir+'summary.csv')
        self.out_corner()
        self.out_diagnostic_plots()        
        self.out_modelplot()
        print('Run complete!')


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
    numax_ = star.numax
    deltanu_ = star.dnu

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
    locs = cop[cop.KIC == str(kic)].Freq.values
    elocs = cop[cop.KIC == str(kic)].e_Freq.values
    modeids = cop[cop.KIC == str(kic)].l.values

    lo = locs.min() - .25*deltanu_
    hi = locs.max() + .25*deltanu_

    # Make the frequency range selection
    ff, pp = data['col1'],data['col2']
    sel = (ff > lo) & (ff < hi)
    f = ff[sel].values
    p = pp[sel].values

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
        phi_ = np.array([np.median(backfit[label]) for label in labels])
        phi_sigma = pd.DataFrame(backfit)[labels].cov()
        phi_cholesky = np.linalg.cholesky(phi_sigma)

    except IndexError:
        print("Can't read in the background for some reason")

    # Build the first guesses
    f0_ = locs[modeids==0]
    f1_ = locs[modeids==1]
    f2_ = locs[modeids==2]
    n0_ = cop.loc[cop.KIC == str(kic)].loc[cop.l == 0].n.values
    n1_ = cop.loc[cop.KIC == str(kic)].loc[cop.l == 1].n.values
    n2_ = cop.loc[cop.KIC == str(kic)].loc[cop.l == 2].n.values
    fs = np.concatenate((f0_, f1_, f2_))
    fs -= fs.min()
    nf = fs/fs.max()
    nf_ = nf[:,None]

    init = [
           f0_,                         # l0 modes
           f1_,                         # l1 modes
           f2_,                         # l2 modes
           np.ones(len(f0_)) * 2.0,     # l0 widths
           np.ones(len(f1_)) * 2.0,     # l1 widths
           np.ones(len(f2_)) * 2.0,     # l2 widths
           np.ones(len(f0_)) * 15. * 2.0 / np.pi / 2.0, # l0 amps
           np.ones(len(f1_)) * 15. * 2.0 / np.pi / 2.0, # l1 amps
           np.ones(len(f2_)) * 15. * 2.0 / np.pi / 2.0, # l2 amps
           1.0 ,                        # projected splitting
           np.pi/2.,                    # inclination angle
           phi_                         # background terms
           ]

    mod = model(f, n0_, n1_, n2_, deltanu_)

    #Plot the data
    pg = lk.Periodogram(f*u.microhertz, p*(cds.ppm**2/u.microhertz))
    ax = pg.plot(alpha=.5)
    ax.scatter(locs, [15]*len(locs),c=modeids, s=20, edgecolor='k')
    ax.plot(f, mod.model(init, theano=False), lw=2)
    plt.savefig(dir+'dataplot.png')
    # plt.show()
    plt.close()

    # Run stan
    print('About to go into Pymc3')
    run = run_pymc3(mod, p, nf_, kic, phi_, phi_cholesky, dir)
    run()
