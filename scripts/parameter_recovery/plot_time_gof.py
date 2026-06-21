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
data_ase = []
for nu in nus:
    for density in densities:
        for n_time_steps in time_steps:
            res_dir = f'output/rw2_d{density}_matern_nu{nu}/n{n_nodes}_T{n_time_steps}_d{density}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['density'] = str(density)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data.append(df)
            
            res_dir = f'output/rw1_d{density}_matern_nu{nu}/n{n_nodes}_T{n_time_steps}_d{density}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['density'] = str(density)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data_rw1.append(df)
            
            res_dir = f'output_ase/d{density}_matern_nu{nu}/n{n_nodes}_T{n_time_steps}_d{density}'
            df = pd.read_csv(join(res_dir, 'results.csv'))
            df['n_nodes'] = n_nodes
            df['density'] = str(density)
            df['n_time_steps'] = n_time_steps
            df['nu'] = str(nu)
            data_ase.append(df)
        

data = pd.concat(data)
data_rw1 = pd.concat(data_rw1)
data_ase = pd.concat(data_ase)

metric_name = {'auc': 'In-Sample AUC', 'aupr': 'In-Sample AUPR'}
for metric in ['auc', 'aupr']:
    data[f'rdpg_{metric}'] = data[f'rdpg_{metric}']    
    data_rw1[f'rdpg_rw1_{metric}'] = data_rw1[f'rdpg_{metric}'] 
    data_ase[f'ase_{metric}'] = data_ase[f'ase_{metric}']
    data_ase[f'true_{metric}'] = data_ase[f'true_{metric}']
    data_ase[f'omni_{metric}'] = data_ase[f'omni_{metric}']

    fontsize = 26
    titlesize = 18
    lw = 3
    ms = 10

    fig, ax = plt.subplots(figsize=(30, 16), nrows=2, ncols=3, sharey=True, sharex=True)

    for l, density in enumerate(densities):
        for k, nu in enumerate(nus):
            data_subset = data.query(f"density == '{density}'").query(f"nu == '{nu}'")
            g = sns.lineplot(x='n_time_steps', y=f'rdpg_{metric}', label='GB-DASE (RW(2) Prior)',
                    data = data_subset, marker='o', ax=ax[k, l],
                    errorbar='sd', lw=lw, markersize=ms)
            if k == 0:
                ax[k, l].set_title(f"Expected Density = {density}", fontsize=fontsize)
            ax[k, l].set_ylabel(r'$\nu =$' + f"{nu}\n\n"+ metric_name[metric], fontsize=28)
            ax[k, l].set_xlabel('Number of Time Points ($m$)', fontsize=fontsize)
            ax[k, l].tick_params(axis='both', which='major', labelsize=22)
            ax[k, l].set_xticks(time_steps)
            if l != 0:
                ax[k, l].set_ylabel(' ')

            data_subset_rw = data_rw1.query(f"density == '{density}'").query(f"nu == '{nu}'")
            sns.lineplot(x='n_time_steps', y=f'rdpg_rw1_{metric}', label='GB-DASE (RW(1) Prior)',
                    data = data_subset_rw, marker='o', ax=ax[k, l],
                    errorbar='sd', linestyle='--', lw=lw, markersize=ms)

            data_subset = data_ase.query(f"density == '{density}'").query(f"nu == '{nu}'")
            sns.lineplot(x='n_time_steps', y=f'ase_{metric}', label='ASE',
                    data = data_subset, marker='o', ax=ax[k, l],
                    errorbar='sd', linestyle=':', lw=lw, markersize=ms)

            sns.lineplot(x='n_time_steps', y=f'omni_{metric}', label='OMNI',
                    data = data_subset, marker='o', ax=ax[k, l],
                    errorbar='sd', linestyle='-.', lw=lw, markersize=ms)

            sns.lineplot(x='n_time_steps', y=f'true_{metric}', label='Oracle',
                    data = data_subset, marker='o', ax=ax[k, l], color='black',
                    errorbar='sd', linestyle='-.', lw=lw, markersize=ms)
            
            ax[k, l].legend(ncol=2, fontsize=16)

    fig.savefig(f'recovery_time_{metric}.pdf', dpi=300, bbox_inches='tight')

