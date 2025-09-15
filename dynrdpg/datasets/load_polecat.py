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
    Y = np.ascontiguousarray(joblib.load(open(file_name, 'rb')))[1:]
    
    # node names
    node_names = pd.read_csv(
            join(file_path, 'polecat', 'node_names.npy'), 
            header=None).values.ravel()
  
    time_points = np.arange(Y.shape[0])
    time_labels = pd.read_csv(
            join(file_path, 'polecat', 'map_week.csv'), header=None).values[1:, 0]
    
    # limit to n_nodes with highest overall degree
    if n_nodes is not None:
        degree = Y.sum(axis=(0, 1))
        order = np.argsort(degree)[::-1][:n_nodes]
        Y = Y[:, order][..., order]
        node_names = node_names[order]
    
    if n_time_points is not None:
        Y = Y[-n_time_points:]
        time_labels = time_labels[-n_time_points:]
    
    return Y, node_names, time_labels
