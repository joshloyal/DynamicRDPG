import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit

from os.path import join



for dens_type in ['decreasing', 'increasing', 'logistic']:
    res_dir = f'output/{dens_type}/rw2_n400_T50'
    data_rw2 = pd.read_csv(join(res_dir, 'results.csv'))

    res_dir = f'output/{dens_type}/rw1_n400_T50'
    data_rw1 = pd.read_csv(join(res_dir, 'results.csv'))
            

    fontsize = 18
    titlesize =20 
    lw = 3
    ms = 10

    rw2_mse = data_rw2.iloc[:, 3:53]
    rw2_mse.columns = np.arange(50)
    rw2_mse.loc[:, 'model'] = 'GB-DASE (RW(2) Prior)'

    rw1_mse = data_rw1.iloc[:, 3:53]
    rw1_mse.columns = np.arange(50)
    rw1_mse.loc[:, 'model'] = 'GB-DASE (RW(1) Prior)'

    ase_mse = data_rw2.iloc[:, 54:104]
    ase_mse.columns = np.arange(50)
    ase_mse.loc[:, 'model'] = 'ASE'

    omni_mse = data_rw2.iloc[:, 104:154]
    omni_mse.columns = np.arange(50)
    omni_mse.loc[:, 'model'] = 'OMNI'

    data = pd.concat([rw2_mse, rw1_mse, ase_mse, omni_mse])
    data = pd.melt(data, id_vars='model')

    fig, ax = plt.subplots(figsize=(17, 5), ncols=2, sharex=True)

    n_time_steps = 50
    density_max = 0.5
    density_min = 0.1
    if dens_type == 'increasing':
        def density(t):
            return (density_max - density_min) * np.sin(np.pi * t / n_time_steps) + density_min 
    elif dens_type == 'decreasing':
        def density(t):
            return -(density_max - density_min) * np.sin(np.pi * t / n_time_steps) + density_max
    else:
        def density(t):
            return (density_max - density_min) * expit((t - n_time_steps/2.) / 3.) + density_min

    ts = np.arange(n_time_steps)
    ax[0].plot(ts, density(ts), 'k-')
    sns.lineplot(x='variable', y='value', hue='model', data=data, ax=ax[1])
    ax[1].legend(ncol=2, fontsize=14)

    if dens_type == 'increasing':
        ax[0].set_title(r"$f_1(t) = 0.4 \sin\left(\pi t/m\right) + 0.1$", fontsize=fontsize)
    elif dens_type == 'decreasing':
        ax[0].set_title(r"$f_2(t) = -0.4 \sin\left(\pi t/m\right) + 0.5$", fontsize=fontsize)
    else:
        ax[0].set_title(r"$f_3(t) = 0.5 \left[1 + \exp\left\{-(t - m/2)/3\right\}\right]^{-1} + 0.1$", fontsize=fontsize)
    ax[0].set_ylabel(r'Expected Edge Density', fontsize=fontsize)
    ax[1].set_ylabel(r'$\text{RMSE}_{\mathbf{X}}$', fontsize=fontsize)

    ax[0].set_xlabel(r'Time ($t$)', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].set_xlabel(r'Time ($t$)', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

#
    fig.savefig(f'{dens_type}_density.pdf', dpi=300, bbox_inches='tight')
