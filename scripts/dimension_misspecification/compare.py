import pandas as pd
import numpy as np
import glob 

from os.path import join


res_dir_name = 'output'

for rw_order in [1, 2]:
    print('Order:', rw_order)
    cols = ['rdpg_x_mse', 'rdpg_proba_mse', 'rdpg_forecast_mse']
    data_true = np.mean(np.sqrt(pd.read_csv(f'output/d0.2/rw{rw_order}_d4_n200_T50_d0.2/results.csv')[cols]), axis=0)

    for d in [2, 3, 5, 6]:
        data = pd.read_csv(f'output/d0.2/rw{rw_order}_d{d}_n200_T50_d0.2/results.csv')
        print(f'd{d}: ', len(data))
        data = np.mean(np.sqrt(data[cols]), axis=0)
        print(np.abs(data) / np.abs(data_true))
