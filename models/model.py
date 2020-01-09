import numpy as np
import emcee
import corner
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import sys

class mix():
    ''' Holder for the mixture model on JVS rotation models '''
    def __init__(self):
        self.rocrit_file = '../data/jvs_models/rocrit_model.csv'
        self.standard_file = '../data/jvs_models/standard_model.csv'
        self.d = '/rds/projects/2018/daviesgr-asteroseismic-computation/ojh251/malatium/models'

        self.mapper = mapper = {'# Teff(K)': 'Teff',
                                '#Teff(K)': 'Teff',
                                ' Prot(days)': 'Prot',
                                ' Age(Gyr)': 'Age',
                                ' Mass(Msun)': 'Mass'}
        self.bw = np.array([0.02, 10.0, 0.01, 0.01])
        self.mass = [0, 0]
        self.teff = [0, 0]
        self.age = [0,0]
        self.prot = [0, 0]
        self.cols = ['Mass', 'Teff', 'Age', 'Prot', 'P_A']
        self.latexcols = [r'$M$', r'$T_{\rm eff}$', r'$\log{\tau}$', r'$\log{P}$', r'$P_A$']

        self.get_data()

    def get_data(self):
        ''' load in the data and fix the column titles '''
        self.df_rocrit = pd.read_csv(self.rocrit_file)
        self.df_stan = pd.read_csv(self.standard_file)
        self.df_rocrit.rename(columns=self.mapper, inplace=True)
        self.df_stan.rename(columns=self.mapper, inplace=True)
        self.df_rocrit['Age'] = np.log(self.df_rocrit.Age)
        self.df_stan['Age'] = np.log(self.df_stan.Age)
        self.df_rocrit['Prot'] = np.log(self.df_rocrit.Prot)
        self.df_stan['Prot'] = np.log(self.df_stan.Prot)

    def print_len(self):
        ''' Print the length of the full data set '''
        print(f'Length of dataset rocrit : {len(self.df_rocrit)}')
        print(f'Length of dataset standard : {len(self.df_stan)}')

    def select_down(self, mass=[1.0, 0.1], teff=[5777.0, 100.0],
                          age=[np.log(5.5), 0.4]):
        ''' Select only a subset of stars within the models '''
        self.sel_rocrit = self.df_rocrit.loc[np.abs(self.df_rocrit.Mass - mass[0]) < mass[1]]
        self.sel_rocrit = self.sel_rocrit.loc[np.abs(self.sel_rocrit.Teff - teff[0]) < teff[1]]
        self.sel_rocrit = self.sel_rocrit.loc[np.abs(self.sel_rocrit.Age - age[0]) < age[1]]

        self.sel_stan = self.df_stan.loc[np.abs(self.df_stan.Mass - mass[0]) < mass[1]]
        self.sel_stan = self.sel_stan.loc[np.abs(self.sel_stan.Teff - teff[0]) < teff[1]]
        self.sel_stan = self.sel_stan.loc[np.abs(self.sel_stan.Age - age[0]) < age[1]]

    def make_kde(self):
        ''' Make a KDE with a preselected bin width '''
        try:
            self.dens_rocrit = sm.nonparametric.KDEMultivariate(
                    data=self.sel_rocrit[['Mass', 'Teff', 'Age', 'Prot']].sample(frac=1.0).values,
                                                var_type='cccc', bw=self.bw)
            self.dens_stan = sm.nonparametric.KDEMultivariate(
                    data=self.sel_stan[['Mass', 'Teff', 'Age', 'Prot']].sample(frac=1.0).values,
                                                var_type='cccc', bw=self.bw)
        except ValueError:
            print('Star out of range of one of the KDEs.')
            np.savetxt(f'{self.d}/{self.ID}_out_of_range.txt', [1])
            sys.exit()

    def plot_kde_example(self, age=np.log(5.5), npts=100):
        ''' Make an example plot to check the KDE is smooth '''
        prot = np.linspace(20.0, 30.0, npts)
        solar_d_ro = [self.dens_rocrit.pdf([1.0, 5777.0, age, n]) for n in np.log(prot)]
        solar_d_stan = [self.dens_stan.pdf([1.0, 5777.0, age, n]) for n in np.log(prot)]
        fig, ax = plt.subplots()
        ax.plot(prot, solar_d_ro / np.max(solar_d_ro))
        ax.plot(prot, solar_d_stan / np.max(solar_d_stan))

    def prior_standard(self, p):
        return self.dens_stan.pdf(p[:-1])

    def prior_rocrit(self, p):
        return self.dens_rocrit.pdf(p[:-1])

    def ln_normal(self, x, mu, sigma):
        return -0.5 * np.abs(x - mu)**2 / sigma**2

    def likelihood(self, p):
        ''' The likelihood function

        A small number is added to the probability from the models
        to stop log(0) from happening
        '''
        if (p[-1] > 1.0) or (p[-1] < 0.0):
            return -np.inf
        like_mix = np.log(1e-30 + p[-1] * self.prior_standard(p) + (1 - p[-1]) * self.prior_rocrit(p))
        like_mix += self.ln_normal(p[0], self.mass[0], self.mass[1])
        like_mix += self.ln_normal(p[1], self.teff[0], self.teff[1])
        like_mix += self.ln_normal(p[2], self.age[0], self.age[1])
        like_mix += self.ln_normal(p[3], self.prot[0], self.prot[1])
        return like_mix

    def set_obs(self, ID, mass, teff, age, prot):
        ''' Set observables '''
        self.ID = ID
        self.mass = mass
        self.teff = teff
        self.age = age
        self.prot = prot

    def mcmc(self, nwalkers=32, burnin=1000, sample=2000):
        ''' Run the mcmc part '''
        ndim = 5
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)
        start =  [self.mass[0], self.teff[0], self.age[0], self.prot[0], 0.01]
        p0 =  [start + np.random.rand(ndim) * [0.001, 100, 0.5, 1.0, 0.98] for n in range(nwalkers)]
        state = self.sampler.run_mcmc(p0, burnin)
        self.sampler.reset()
        self.sampler.run_mcmc(state, sample)

        frac_acc = np.mean(self.sampler.acceptance_fraction)
        if  frac_acc < 0.2:
            warnings.warn('Sampler acceptance fraction is low : {frac_acc}')

    def corner(self, input):
        ''' Plot a corner plot '''
        samples = self.sampler.get_chain(flat=True)
        hall = [input['mass'][0], input['teff'][0],\
                input['logage'][0], input['logprot'][0], np.nan]
        corner.corner(samples, truths=hall, labels=self.latexcols)
        plt.savefig(f'{self.d}/{self.ID}_corner.png')
        plt.close('all')

    def save_samples(self):
        ''' Save the samples to csv '''
        samples = self.sampler.get_chain(flat=True)
        output = pd.DataFrame(data=samples, columns=self.cols)
        output.to_csv(f'{self.d}/{self.ID}_samples.csv')

    def run_one_star(self, input):
        ''' run the whole thing for a single star

        Inputs
        ------

        input: dict
            Dictionary that contains ID, mass, teff, age, prot as
            list of length 2 with [value, uncertainty].

        '''
        # Run a Prot check
        if np.isnan(input['logprot'][0]):
            print('No results for rotation.')
            np.savetxt(f'{self.d}/{self.ID}_incomplete.txt', [0])

        self.select_down(mass=[input['mass'][0], input['mass'][1]*3],
                         teff=[input['teff'][0], input['teff'][1]*3],
                         age=[input['logage'][0], input['logage'][1]*3])
        self.set_obs(ID=input['ID'],
                     mass=input['mass'],
                     teff=input['teff'],
                     age=input['logage'],
                     prot=input['logprot'])
        self.make_kde()

        self.mcmc()
        self.corner(input)
        self.save_samples()

    def __call__(self):
        pass
