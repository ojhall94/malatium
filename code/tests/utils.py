import glob
from astropy.io import ascii
import pandas as pd
import pickle
import numpy as np

def read_data(target, lower_index, upper_index):
    mal = pd.read_csv('../../data/malatium.csv', index_col=0)
    idx = np.where(mal.KIC == target)[0][0]
    star = mal.loc[idx]
    kic = star.KIC
    dnu_ = star.dnu

    sfile = glob.glob('../../data/*{}*.pow'.format(kic))
    data = ascii.read(sfile[0]).to_pandas()
    ff, pp = data['col1'].values, data['col2'].values

    # Read in the mode locs
    cop = pd.read_csv('../../data/copper.csv',index_col=0)
    cop = cop[cop.l != 3]
    modelocs = cop[cop.KIC == str(kic)].Freq.values[lower_index:upper_index]
    elocs = cop[cop.KIC == str(kic)].e_Freq.values[lower_index:upper_index]
    modeids = cop[cop.KIC == str(kic)].l.values[lower_index:upper_index]
    overtones = cop[cop.KIC == str(kic)].n.values[lower_index:upper_index]

    lo = modelocs.min() - .25*dnu_
    hi = modelocs.max() + .25*dnu_

    sel = (ff > lo) & (ff < hi)
    f = ff[sel]
    p = pp[sel]

    return f, p, ff, pp, star, modelocs, elocs, modeids, overtones

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

def divide_background(kic, f, p, ff, pp):
    backdir = glob.glob('/home/oliver/PhD/mnt/RDS/malatium/backfit/'
                    +str(kic)+'/*_fit.pkl')[0]
    with open(backdir, 'rb') as file:
        backfit = pickle.load(file)

    labels=['loga','logb','logc','logd','logj','logk','white','scale','nyq']
    res = np.array([np.median(backfit[label]) for label in labels])
    res[0:6] = 10**res[0:6]

    phi_ = np.array([np.median(backfit[label]) for label in labels])
    phi_sigma = pd.DataFrame(backfit)[labels].cov()
    phi_cholesky = np.linalg.cholesky(phi_sigma)

    model = get_background(ff, *res)
    m = get_background(f, *res)
    p /= m
    pp /= model

    return p, pp

def rebin(f, p, binsize):

    print('Length: {}'.format(len(f)))
    m = int(len(p)/binsize)

    bin_f = f[:m*binsize].reshape((m, binsize)).mean(1)
    bin_p = p[:m*binsize].reshape((m, binsize)).mean(1)
    print('Length: {}'.format(len(bin_f)))
    return bin_f, bin_p

def gaussian(locs, l, numax, Hmax0):
    fwhm = 0.25 * numax
    std = fwhm/2.355

    Vl = [1.0, 1.22, 0.71, 0.14]

    return Hmax0 * Vl[l] * np.exp(-0.5 * (locs - numax)**2 / std**2)
