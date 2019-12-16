import numpy as np
import lightkurve as lk
import pandas as pd
import astropy.units as u
from tqdm import tqdm
import math
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import ascii
from astropy.convolution import convolve, Box1DKernel
import corner
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
sns.set_context('poster')
sns.set_palette('colorblind')


class model():
    def __init__(self, f, n0_, n1_, n2_, deltanu_):
        self.f = f
        self.n0 = n0_
        self.n1 = n1_
        self.n2 = n2_
        self.npts = len(f)
        self.M = [len(n0_), len(n1_), len(n2_)]
        self.deltanu = deltanu_

    def epsilon(self, i, l, m):
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
                
    def lor(self, freq, h, w):
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)

    def mode(self, l, freqs, hs, ws, i, split=0):
        for idx in range(self.M[l]):
            for m in range(-l, l+1, 1):
                self.modes += self.lor(freqs[idx] + (m*split),
                                     hs[idx] * self.epsilon(i, l, abs(m)),
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
        self.modes = np.zeros(self.npts)
        self.mode(0, f0, h0, g0, i)
        self.mode(1, f1, h1, g1, i, split)
        self.mode(2, f2, h2, g2, i, split)
        self.modes *= self.get_apodization(nyq)

        #Calculate the background
        self.back = self.get_background(loga, logb, logc, logd, logj, logk,
                                       white, scale, nyq)

        #Create the model
        self.mod = self.modes + self.back
        return self.mod

    def harvey(self, a, b, c):
        harvey = 0.9*a**2/b/(1.0 + (self.f/b)**c)
        return harvey

    def get_apodization(self, nyquist):
        x = (np.pi * self.f) / (2 * nyquist)
        return (np.sin(x)/x)**2

    def get_background(self, loga, logb, logc, logd, logj, logk, white, scale, nyq):
        background = np.zeros(len(self.f))
        background += self.get_apodization(nyq) * scale  \
                        * (self.harvey(10**loga, 10**logb, 4.) \
                        +  self.harvey(10**logc, 10**logd, 4.) \
                        +  self.harvey(10**logj, 10**logk, 2.))\
                        +  white
        return background  

def save_corner(chains, kic, idx):
    labels = ['xsplit', 'cosi', 'i', 'split', 'P']
    nus = u.Quantity(chains['split'].values, u.microhertz)
    chains['P'] = 1./nus.to(1./u.day).value

    chain = np.array([chains[label] for label in labels])

    fig = corner.corner(chain.T, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.suptitle(f'KIC {kic}, ID {idx}')
    plt.savefig(f'/home/oliver/PhD/mnt/RDS/malatium/peakbag/byeye/{kic}_corner.png')
    plt.close()

def get_models(chains, kic, idx, ati):
    bro = pd.read_csv('../../data/bronze.csv', index_col=0)
    star = bro[bro.KIC == str(kic)]
    res = [star.loc[star.l == 0].f.values,
            star.loc[star.l == 1].f.values,
            star.loc[star.l == 2].f.values,
            star.loc[star.l == 0].g.values,
            star.loc[star.l == 1].g.values,
            star.loc[star.l == 2].g.values,
            star.loc[star.l == 0].H.values,
            star.loc[star.l == 1].H.values,
            star.loc[star.l == 2].H.values,
            ati.loc[idx].nus,
            ati.loc[idx].i,
            [np.median(chains[f'phi__{j}']) for j in range(9)]]

    # Read in the data
    sfile = glob.glob(f'../../data/*{kic}*.pow')
    data = ascii.read(sfile[0]).to_pandas()
    ff, pp = data['col1'], data['col2']

    # Select the range
    deltanu = ati.loc[idx].dnu
    lo = star.f.min() - .25*deltanu
    hi = star.f.max() + .25*deltanu
    
    sel = (ff > lo) & (ff < hi)
    f = ff[sel].values
    p = pp[sel].values

    # Set up the model
    n0 = star.loc[star.l == 0].n.values
    n1 = star.loc[star.l == 1].n.values
    n2 = star.loc[star.l == 2].n.values
    mod = model(f, n0, n1, n2, deltanu)    

    full_plot(mod, p, res, kic, idx)

    ns = np.unique(star.n.values)
    N = len(ns)

    fig, ax = plt.subplots(N, figsize=[10,2*N])
    plt.subplots_adjust(hspace=0.0)
    for i, n in enumerate(ns):
        s = star.loc[star.n == n].loc[star.l != 2] 
        lo = s.f.min() - 0.25 * deltanu
        hi = s.f.max() + 0.25 * deltanu
        ref = f[(f > lo) & (f < hi)]        
        rep = p[(f > lo) & (f < hi)]
        smp = smooth(ref, rep, filter_width=0.1)

        mod = model(ref, n0, n1, n2, deltanu)
        M = mod.model(res, theano=False)

        ax[i].plot(ref, rep, c='k', lw=1, alpha=.5)
        ax[i].plot(ref, smp, c='k', lw=1, alpha=1.)
        ax[i].plot(ref, M, c='r', lw=1)

        markers = ['o',',','^']
        sc = star.copy()
        sc.loc[star.l == 2, 'n'] += 1
        modes = sc.loc[sc.n == n]
        for j, l in enumerate(modes.l):
            ax[i].scatter(modes.loc[modes.l == l].f, 10,
                        marker=markers[l], s=25, c='r', zorder=10)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)   
    fig.suptitle(f'KIC {kic}, ID {idx}')

    plt.savefig(f'/home/oliver/PhD/mnt/RDS/malatium/peakbag/byeye/{kic}_echelle.png', dpi=450)            
    plt.close()

def smooth(f, p, filter_width=35):
    fs = np.mean(np.diff(f))

    box_kernel = Box1DKernel(math.ceil((filter_width/fs)))
    smooth_power = convolve(p, box_kernel)
    return smooth_power


def full_plot(mod, p, res, kic, idx):
    pg = lk.Periodogram(mod.f*u.microhertz, p*(u.cds.ppm**2/u.microhertz))
    ax = pg.plot(alpha=.5, label='Data')
    plt.plot(mod.f, mod.model(res, theano=False), lw=3, label='Model')
    ax.set_title(f'KIC {kic}, ID {idx}')
    plt.savefig(f'/home/oliver/PhD/mnt/RDS/malatium/peakbag/byeye/{kic}_full.png')
    plt.close()

if __name__ == "__main__":
    ati = pd.read_csv('../../data/atium.csv', index_col=0)

    #Visual inspection of the corner plots
    for idx in tqdm(range(2)):
        kic = ati.loc[idx].KIC

        print('#############################')
        print(f'RUNNING KIC {kic}, IDX {idx}')
        print('#############################')

        files = glob.glob('/home/oliver/PhD/mnt/RDS/malatium/peakbag/{}/*chains.csv'.format(str(kic)))

        try:
            chains = pd.read_csv(files[0], index_col=0)
        except IndexError:
            continue

        save_corner(chains, kic, idx)
        get_models(chains, kic, idx, ati)
    


