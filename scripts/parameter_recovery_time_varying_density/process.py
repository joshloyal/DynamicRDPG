import pandas as pd
import numpy as np
import glob 

from os.path import join


for dens_type in ['increasing', 'decreasing', 'logistic']:
    res_dir_name = f'output/{dens_type}'

    for file_name in glob.glob(res_dir_name + "/*"):
        try:
            print(file_name)
            data = []
            for sub_file_name in glob.glob(file_name + "/*"):
                print(sub_file_name)
                data.append(pd.read_csv(sub_file_name))
            data = pd.concat(data)
            data.to_csv(join(file_name, 'results.csv'), index=False)
        except:
            pass
