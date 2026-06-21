import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


family = 'laplace'
node_sizes = [25, 50, 100, 200, 400]
n_time_steps = 50
snrs = [0.5, 1.0, 2.0]
nus = [2.5, 0.5] 

data = []
data_rw1 = []
for nu in nus:
    for snr in snrs:
        for n_nodes in node_sizes:
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

fontsize = 26
titlesize = 18
lw = 3
ms = 10

fig, ax = plt.subplots(figsize=(30, 12), nrows=2, ncols=3, sharey=True, sharex=True)

for l, snr in enumerate(snrs):
    for k, nu in enumerate(nus):
        data_subset = data.query(f"snr == '{snr}'").query(f"nu == '{nu}'")
        g = sns.lineplot(x='n_nodes', y='rpdg_mse', label='GB-DASE (RW(2) Prior)',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', lw=lw, markersize=ms)
        if k == 0:
            ax[k, l].set_title(f"SNR = {snr}", fontsize=fontsize)
        ax[k, l].set_ylabel(r'$\nu =$' + f"{nu}\n\n"+ r'$\text{RMSE}_{\mathbf{X}}$', fontsize=28)
        ax[k, l].set_xlabel('Number of Nodes ($n$)', fontsize=fontsize)
        ax[k, l].tick_params(axis='both', which='major', labelsize=22)
        ax[k, l].set_xticks(node_sizes)
        
        data_subset_rw = data_rw1.query(f"snr == '{snr}'").query(f"nu == '{nu}'")
        sns.lineplot(x='n_nodes', y='rpdg_mse', label='GB-DASE (RW(1) Prior)',
                data = data_subset_rw, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='--', lw=lw, markersize=ms)

        sns.lineplot(x='n_nodes', y='ase_mse', label='ASE',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle=':', lw=lw, markersize=ms)

        sns.lineplot(x='n_nodes', y='omni_mse', label='OMNI',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='-.', lw=lw, markersize=ms)
        ax[k, l].set(yscale='log')
        ax[k, l].legend(ncol=2, fontsize=16)

label = 'Poisson' if family == 'poisson' else 'Laplace'
fig.suptitle(label + ' Noise', fontsize=30)
fig.savefig(f'recovery_nodes_{family}.pdf', dpi=300, bbox_inches='tight')

