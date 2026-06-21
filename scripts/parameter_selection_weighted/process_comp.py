import pandas as pd
import numpy as np
import glob 

from os.path import join


for family in ['poisson', 'laplace']:
    for sim_rw_type in [1, 2]:
        for dims in [2, 4]:
            for snr in [1.0, 0.5, 0.05]:
                print(f'{family}, d_0 = {dims}, scenario = {1 if sim_rw_type == 2 else 2}, snr = {snr}')
                print('========================')
                data = []
                for rw_type in [1, 2]:
                    res_dir_name = f'output/{family}/sim_rw{sim_rw_type}/rw{rw_type}_n200_T50_s{snr}_dim{dims}'
                    for file_name in glob.glob(res_dir_name + "/*"):
                        seed = file_name.split('/')[-1].split('_')[-1].split('.')[0]
                        df = pd.read_csv(file_name)
                        df['rw_type'] = rw_type
                        df['seed'] = int(seed)
                        data.append(df)
                data = pd.concat(data)

                data_rw1 = data.query('rw_type == 1')
                data_rw2 = data.query('rw_type == 2')
                data = data_rw2.merge(data_rw1, on='seed', how='inner', suffixes=['_rw2', '_rw1'])
                
                loss = 'mse'

                # which selected model has the lowest MSE for latent space recovery
                q = np.argmin(data[['rdpg_mse_oracle_rw2', 'rdpg_mse_oracle_rw1']].values, axis = 1) 
                err_oracle = np.min(data[['rdpg_mse_oracle_rw2', 'rdpg_mse_oracle_rw1']].values, axis = 1)
                
                select_rw = np.argmin(data[[f'waic_{loss}_rw2', f'waic_{loss}_rw1']].values, axis=1)
                err_waic = data[[f'rdpg_mse_{loss}_rw2', f'rdpg_mse_{loss}_rw1']].values[np.arange(len(data)), select_rw]
               
              
                select_d = data[[f'n_features_{loss}_rw2', f'n_features_{loss}_rw1']].values[np.arange(len(data)), select_rw] 
                d_prop = select_d == dims
                print(f'Mean d: {np.mean(select_d)}')
                print(f'd-Prop: {np.mean(d_prop)}')
                
                q_prop = select_rw == q
                print(f'(d,r)-Prop: {np.mean(np.logical_and(d_prop, q_prop))}') 
                print(f'Ratio: {np.mean(err_waic/err_oracle)}\n') 
