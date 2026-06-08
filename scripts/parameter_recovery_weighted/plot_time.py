import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


family = 'laplace'
n_nodes = 50
time_steps = [10, 25, 50, 100, 200]
snrs = [2.0, 1.0, 0.5]
nus = [2.5, 0.5] 

data = []
data_rw1 = []
for nu in nus:
    for snr in snrs:
        for n_time_steps in time_steps:
            res_dir = f'output/{family}/rw2_s{snr}_nu{nu}/n{n_nodes}_T{n_time_steps}_s{snr}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['snr'] = str(snr)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data.append(df)
            
            res_dir = f'output/{family}/rw1_s{snr}_nu{nu}/n{n_nodes}_T{n_time_steps}_s{snr}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['snr'] = str(snr)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data_rw1.append(df)
            

data = pd.concat(data)
data_rw1 = pd.concat(data_rw1)

#data['rw1_rdpg_mse'] = np.sqrt(data['rw1_rdpg_mse'])
#data['rw2_rdpg_mse'] = np.sqrt(data['rw2_rdpg_mse'])
#data['ase_mse'] = np.sqrt(data['ase_mse'])
#data['omni_mse'] = np.sqrt(data['omni_mse'])

fontsize = 26
titlesize = 18
lw = 3
ms = 10

fig, ax = plt.subplots(figsize=(30, 12), nrows=2, ncols=3, sharey=True, sharex=True)

for l, snr in enumerate(snrs):
    for k, nu in enumerate(nus):
        data_subset = data.query(f"snr == '{snr}'").query(f"nu == '{nu}'")
        g = sns.lineplot(x='n_time_steps', y='rpdg_mse', label='GB-DASE (RW(2) Prior)',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', lw=lw, markersize=ms)
        if k == 0:
            ax[k, l].set_title(f"SNR = {snr}", fontsize=fontsize)
        
        ax[k, l].set_ylabel(r'$\nu =$' + f"{nu}\n\n"+ r'$\text{RMSE}_{\mathbf{X}}$', fontsize=28)
        ax[k, l].set_xlabel('Number of Time Points ($m$)', fontsize=fontsize)
        ax[k, l].tick_params(axis='both', which='major', labelsize=22)
        ax[k, l].set_xticks(time_steps)
        
        data_subset_rw = data_rw1.query(f"snr == '{snr}'").query(f"nu == '{nu}'")
        sns.lineplot(x='n_time_steps', y='rpdg_mse', label='GB-DASE (RW(1) Prior)',
                data = data_subset_rw, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='--', lw=lw, markersize=ms)

        sns.lineplot(x='n_time_steps', y='ase_mse', label='ASE',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle=':', lw=lw, markersize=ms)

        sns.lineplot(x='n_time_steps', y='omni_mse', label='OMNI',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='-.', lw=lw, markersize=ms)

        ax[k, l].set(yscale='log')
        ax[k, l].legend(ncol=2, fontsize=16)
        #if k == 0:
        #    ax[k, l].set_ylim(0, 0.38)

label = 'Poisson' if family == 'poisson' else 'Laplace'
fig.suptitle(label + ' Noise', fontsize=30)
fig.savefig(f'recovery_time_{family}.pdf', dpi=300, bbox_inches='tight')

