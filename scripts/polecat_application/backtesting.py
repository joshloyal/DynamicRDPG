import plac
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from dynrdpg.datasets import load_polecat
from dynrdpg import DynamicRDPG
from dynrdpg.model_selection import backtest_selection



def backtest(rw_order=1):
    rw_order = int(rw_order)

    Y, node_names, time_labels = load_polecat(50)

    # index of first week of 2024
    k = -26
 
    models, criteria = backtest_selection(
        Y[:k], is_binary=True, rw_order=rw_order, min_features=2, max_features=5,
        n_heldout=4,
        n_burnin=5000, n_samples=5000)
    
    out_file = f'rw{rw_order}_results_backtest.csv'
    dir_base = 'output'
    if not os.path.exists(dir_base):
        os.makedirs(dir_base)

    criteria.to_csv(os.path.join(dir_base, out_file), index=False)


for r in [1, 2]:
    backtest(rw_order=r)
