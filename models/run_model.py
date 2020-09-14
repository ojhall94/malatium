#!/usr/bin/env python3
#O. J. Hall 2019

from model import mix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser(description='Run our emcee model')
parser.add_argument('idx',type=int,help='Index on the kiclist')
args = parser.parse_args()

if __name__ == "__main__":
    # Read in the data
    dfile = "../data/atium.csv"
    df = pd.read_csv(dfile, index_col=0)

    ## Below: systematic increase blocks
    # Increase uncertainties on age and mass
    # df['upage'][df.source == 'K'] += 0.06 * df['age'][df.source == 'K'] 
    # df['loage'][df.source == 'K'] += 0.06 * df['age'][df.source == 'K']
    # df['upage'][df.source == 'L'] += 0.08 * df['age'][df.source == 'L'] 
    # df['loage'][df.source == 'L'] += 0.08 * df['age'][df.source == 'L']

    # df['upmodmass'][df.source == 'K'] += 0.02 * df['modmass'][df.source == 'K'] 
    # df['lomodmass'][df.source == 'K'] += 0.02 * df['modmass'][df.source == 'K']
    # df['upmodmass'][df.source == 'L'] += 0.016 * df['modmass'][df.source == 'L'] 
    # df['lomodmass'][df.source == 'L'] += 0.016 * df['modmass'][df.source == 'L']    

    # Build the input list
    df['logage'] = np.log(df.age)
    df['uplogage'] = np.log(df.age + df.upage) - df.logage
    df['lologage'] = df.logage - np.log(df.age - df.loage)

    df['logP'] = np.log(df.P)
    df['uplogP'] = np.log(df.P + df.u_P) - df.logP
    df['lologP'] = df.logP - np.log(df.P - df.l_P)

    stars = []
    for idx, row in df.iterrows():
        mass = [row.modmass, max([row.upmodmass, row.lomodmass])]
        teff = [row.Teff, row.eTeff]
        logage = [row.logage, max([row.uplogage, row.lologage])]
        logprot = [row.logP, max([row.uplogP, row.lologP])]
        stars.append({'ID': str(row.KIC), 
                    'mass': mass, 
                    'teff': teff, 
                    'logage':  logage, 
                    'logprot': logprot})

    # Select the star we want
    star = stars[args.idx]
    print('################################')
    print(f"RUNNING ON KIC {star['ID']} | IDX {args.idx}")
    print(star)
    print('################################')

    # Run the model
    print('Running emcee')
    
    mix = mix()
    mix.run_one_star(star)
    print('Run complete!')
