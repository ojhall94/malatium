import glob
from astropy.io import ascii
import pandas as pd
import pickle
import numpy as np

def pairplot_divergence(x, y, trace, ax=None, divergence=True, color='C3', divergence_color='C2'):
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y, 'o', color=color, alpha=.5)
    if divergence:
        divergent = trace['diverging']
        ax.plot(x[divergent], y[divergent], 'o', color=divergence_color)
    return ax

def divergence_corner(trace, labels, entry=0):
    chain = np.array([trace[label] for label in labels])
    if len(chain.shape) > 2:
        chain = chain[:,:,entry]
        print('Only showing the entry [{}] for multi-parameter labels'.format(entry))
        titleadd = '[{}]'.format(entry)
    else:
        titleadd = ''
    
    K = len(chain)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(K, K, figsize=(dim, dim))

    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", 'k')
    for i, x in enumerate(chain):
        ax = axes[i,i]
        bins_1d = int(max(1, 20.))
        n, _, _ = ax.hist(x, bins=bins_1d, histtype='step')       


        title = "{}{}".format(labels[i], titleadd)
        ax.set_title(title)    

        for j, y in enumerate(chain):
            ax = axes[i, j]

            if j > i:    
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                ax.set_xticks([])
                ax.set_yticks([])            
                continue    

            ax = pairplot_divergence(y, x, trace, ax=ax)

            if i < K - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])   
                
# plot the estimate for the mean of log(τ) cumulating mean
def biasplot(label, true):
    logtau = np.log(trace[label])
    mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
    plt.figure(figsize=(15, 4))
    plt.axhline(np.log(true), lw=2.5, color='gray')
    plt.plot(mlogtau, lw=2.5)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of log({})'.format(label))
    plt.title('MCMC estimation of log({})'.format(label));                

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
