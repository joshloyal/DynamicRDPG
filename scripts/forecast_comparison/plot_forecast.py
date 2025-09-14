import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


n_nodes = 200
densities = [0.3, 0.2, 0.1]
k_steps = 6

fig, ax = plt.subplots(figsize=(20, 24), nrows=3, ncols=2, sharey='row', sharex=True)
for l, n_time_steps in enumerate([100, 150]):
    for k, density in enumerate(densities):
        res_dir = f'output/d{density}/n{n_nodes}_T{n_time_steps}_d{density}'
        data = pd.read_csv(join(res_dir, 'results.csv')) 

        forecast_data = []
        for kstep in range(1, k_steps):
            var_name = f'kstep_mse_{kstep}'
            row = {
                'rdpg_rw2': np.sqrt(data[f'rdpg_rw2_{var_name}']).mean(),
                'rdpg_rw2_std': np.std(np.sqrt(data[f'rdpg_rw2_{var_name}'])),
                'rdpg_rw1': np.sqrt(data[f'rdpg_rw1_{var_name}']).mean(),
                'rdpg_rw1_std': np.std(np.sqrt(data[f'rdpg_rw1_{var_name}'])),
                'ase': np.sqrt(data[f'ase_{var_name}']).mean(),
                'ase_std': np.std(np.sqrt(data[f'ase_{var_name}'])),
                'uase': np.sqrt(data[f'uase_{var_name}']).mean(),
                'uase_std': np.std(np.sqrt(data[f'uase_{var_name}'])),
                'omni': np.sqrt(data[f'omni_{var_name}']).mean(),
                'omni_std': np.std(np.sqrt(data[f'omni_{var_name}'])),
                'mase': np.sqrt(data[f'mase_{var_name}']).mean(),
                'mase_std': np.std(np.sqrt(data[f'mase_{var_name}'])),
                'k_step': kstep
            }
            forecast_data.append(row)

        forecast_data = pd.DataFrame(forecast_data)

        fontsize = 20
        titlesize = 18 
        lw = 3
        ms = 10


        sns.lineplot(x='k_step', y='rdpg_rw2', label='GB-DASE (RW(2) Prior)',
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', lw=lw, markersize=ms)
        up = forecast_data['rdpg_rw2'] + forecast_data['rdpg_rw2_std']
        down = forecast_data['rdpg_rw2'] - forecast_data['rdpg_rw2_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)

        if k == 0:
            ax[k, l].set_title(f"m = {n_time_steps}", fontsize=fontsize)
        if l == 0:
            ax[k, l].set_ylabel(f'Expected Density = {density}\n\nRMSE', fontsize=fontsize)

        if k == 2:
            ax[k, l].set_xlabel('K-Steps Ahead', fontsize=fontsize)
        ax[k, l].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[k, l].set_xticks(np.arange(1, 6))

        sns.lineplot(x='k_step', y=f'rdpg_rw1', label='GB-DASE (RW(1) Prior)', 
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', lw=lw, markersize=ms, linestyle='--') #
        up = forecast_data['rdpg_rw1'] + forecast_data['rdpg_rw1_std']
        down = forecast_data['rdpg_rw1'] - forecast_data['rdpg_rw1_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)

        sns.lineplot(x='k_step', y='ase', label='ASE',
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle=':', lw=lw, markersize=ms)
        up = forecast_data['ase'] + forecast_data['ase_std']
        down = forecast_data['ase'] - forecast_data['ase_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)

        sns.lineplot(x='k_step', y='omni', label='OMNI',
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='-.', lw=lw, markersize=ms)
        up = forecast_data['omni'] + forecast_data['omni_std']
        down = forecast_data['omni'] - forecast_data['omni_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)
        

        sns.lineplot(x='k_step', y='uase', label='UASE',
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle=(0, (3, 5, 1, 5, 1, 5)), lw=lw, markersize=ms)
        up = forecast_data['uase'] + forecast_data['uase_std']
        down = forecast_data['uase'] - forecast_data['uase_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)
        
        sns.lineplot(x='k_step', y='mase', label='MASE',
                data = forecast_data, marker='o', ax=ax[k, l],
                errorbar='sd', linestyle='-.', lw=lw, markersize=ms)
        up = forecast_data['mase'] + forecast_data['mase_std']
        down = forecast_data['mase'] - forecast_data['mase_std']
        ax[k, l].fill_between(np.arange(1, 6), up, down, alpha=0.25)

        ax[k, l].legend(loc = 'upper left', ncol=3, fontsize=16)
        
        if k == 0:
            ax[k, l].set_ylim(0, 0.42)
        if k == 1:
            ax[k, l].set_ylim(0, 0.30)
        if k == 2:
            ax[k, l].set_ylim(0, 0.15)

fig.savefig(f'forecasts.pdf', dpi=300, bbox_inches='tight')
