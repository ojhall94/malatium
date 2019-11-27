import numpy as np
import lightkurve as lk
import pandas as pd
import fnmatch as fnm
import seaborn as sns
import astropy.units as u
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt



if __name___ == "__main__":
    ati = pd.read_csv('../../data/atium.csv', index_col=0)

    #Visual inspection of the corner plots
    for idx in range(95):
        kic = mal.loc[idx].kicfiles = glob.glob('/home/oliver/PhD/mnt/RDS/malatium/peakbag/{}/*chains.csv'.format(str(kic)))

        try:
            chains = pd.read_csv(files[0], index_col=0)
        except IndexError:
            continue

        labels = ['xsplit', 'cosi', 'i', 'split', 'P']
        nus = u.Quantity(chains['split'].values, u.microhertz)
        chains['P'] = 1./nus.to(1./u.day).value
        
        chain = np.array([chains[label] for label in labels])

        corner.corner(chain.T, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.show()
        
