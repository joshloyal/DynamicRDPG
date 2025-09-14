import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


time_steps = [10, 25, 50, 100, 200]
n_nodes = 50
densities = [0.3, 0.2, 0.1]
nus = [0.5, 2.5] 

data_fase = []
data_rw1 = []
data = []
for nu in nus:
    for density in densities:
        for n_time_steps in time_steps:
            res_dir = f'output/d{density}_matern_nu{nu}/n{n_nodes}_T{n_time_steps}_d{density}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['density'] = str(density)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data.append(df)
             

data = pd.concat(data)

data['rw2_rdpg_mse'] = np.sqrt(data['rw2_rdpg_mse'])
data['rw1_rdpg_mse'] = np.sqrt(data['rw1_rdpg_mse'])
data['ase_mse'] = np.sqrt(data['ase_mse'])
data['omni_mse'] = np.sqrt(data['omni_mse'])


fontsize = 26
titlesize = 18
lw = 3
ms = 10

fig, ax = plt.subplots(figsize=(30, 16), nrows=2, ncols=3, sharey=True, sharex=True)

for l, density in enumerate(densities):
    for k, nu in enumerate(nus):
        data_subset = data.query(f"density == '{density}'").query(f"nu == '{nu}'")
        g = sns.lineplot(x='n_time_steps', y='rw2_rdpg_mse', label='GB-DASE (RW(2) Prior)',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', lw=lw, markersize=ms)
        if k == 0:
            ax[k, l].set_title(f"Expected Density = {density}", fontsize=fontsize)
        ax[k, l].set_ylabel(r'$\nu =$' + f"{nu}\n\n"+ r'$\text{RMSE}_{\mathbf{X}}$', fontsize=28)
        ax[k, l].set_xlabel('Number of Time Points ($m$)', fontsize=fontsize)
        ax[k, l].tick_params(axis='both', which='major', labelsize=22)
        ax[k, l].set_xticks(time_steps)

        sns.lineplot(x='n_time_steps', y='rw1_rdpg_mse', label='GB-DASE (RW(1) Prior)',
                data = data_subset_rw, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='--', lw=lw, markersize=ms)

        sns.lineplot(x='n_time_steps', y='ase_mse', label='ASE',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle=':', lw=lw, markersize=ms)

        sns.lineplot(x='n_time_steps', y='omni_mse', label='OMNI',
                data = data_subset, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='-.', lw=lw, markersize=ms)

        ax[k, l].legend(ncol=2, fontsize=16)
        if k == 0:
            ax[k, l].set_ylim(0, 0.43)

fig.savefig(f'plots/recovery_time.pdf', dpi=300, bbox_inches='tight')

