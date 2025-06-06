import joblib
import numpy as np
import pandas as pd

from datetime import datetime as dt
from os.path import dirname, join


__all__ = ['load_polecat']


def load_polecat(n_nodes=None, n_time_points=None):
    module_path = dirname(__file__)
    file_path = join(module_path, 'data')

    file_name = join(file_path, 'polecat', 
                     'polecat_weekly_subset.gz')
    # XXX first entry is from 2020 so discard it
    Y = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))[1:]
    
    # node names
    node_names = pd.read_csv(
            join(file_path, 'polecat', 'node_names.npy'), 
            header=None).values.ravel()
 
    #iso_codes = pd.read_csv(join(file_path, 'polecat', 'iso_codes.csv'))
    #iso_codes = pd.read_csv(join(file_path, 'polecat', 'iso_regions.csv'))
    #iso_map = {row['Country']: row['ISO3'] for (idx, row) in iso_codes.iterrows()} 
    #region_map = {row['Country']: row['Region'] for (idx, row) in iso_codes.iterrows()} 
    #iso_codes = np.array([iso_map[i] for i in node_names])
    #regions = np.array([region_map[i] for i in node_names])
 
    time_points = np.arange(Y.shape[0])
    #time_labels = []
    #years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    #years = [2021, 2022, 2023, 2024]
    #years = [2021, 2022, 2023, 2024]
    #for t in np.arange(Y.shape[0]+1):
    #    week = t % 52 + 1
    #    year = years[t // 52] 
    #    date = dt.fromisocalendar(year, week, 1)
    #    time_labels.append(date.strftime("%Y-%m-%W"))
    time_labels = pd.read_csv(
            join(file_path, 'polecat', 'map_week.csv'), header=None).values[1:, 0]
    
    # limit to n_nodes with highest overall degree
    if n_nodes is not None:
        degree = Y.sum(axis=(0, 1))
        order = np.argsort(degree)[::-1][:n_nodes]
        Y = Y[:, order][..., order]
        node_names = node_names[order]
        #iso_codes = iso_codes[order]
        #regions = regions[order]
    
    if n_time_points is not None:
        Y = Y[-n_time_points:]
        time_labels = time_labels[-n_time_points:]
    
    return Y, node_names, time_labels
