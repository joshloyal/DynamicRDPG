import plac
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

from dynrdpg import DynamicRDPG
from dynrdpg.dynrdpg import calculate_auc, calculate_aupr, dynamic_adjacency_to_vec
from dynrdpg.datasets import simulate_network_bspline


def mse(X_true, X_pred):
    n_time_steps, n_nodes, n_features = X_true.shape
    d_max = max(X_pred.shape[-1], n_features)

    if d_max > n_features:
        n_padded = d_max - n_features
        X_true = np.concatenate([X_true, np.zeros((n_time_steps, n_nodes, n_padded))], axis=2)

    if d_max > X_pred.shape[-1]:
        n_padded = d_max - X_pred.shape[-1]
        X_pred = np.concatenate([X_pred, np.zeros((n_time_steps, n_nodes, n_padded))], axis=2)

    # calculate latent space recovery (with padding)
    for t in range(n_time_steps):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R

    return np.mean((X_pred - X_true) ** 2)


def simulation(seed, n_nodes=100, n_time_steps=100, density=0.2, rw_order=2, n_features=2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_steps = int(n_time_steps)
    density = float(density)
    rw_order = int(rw_order)

    k_steps = 5
    k_buffer = 10

    Y, X, probas_true = simulate_network_bspline(
            n_nodes=n_nodes, n_time_steps=n_time_steps, 
            n_features=4, density=density, df=5, k_buffer=k_buffer, 
            k_steps=k_steps, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)
    probas_true = dynamic_adjacency_to_vec(probas_true, is_binary=False)

    data = {}

    # Dynamic RDPG 
    rdpg = DynamicRDPG(n_features=n_features, rw_order=rw_order, random_state=42)
    rdpg.sample(Y[:-k_steps], n_burnin=2500, n_samples=2500)

    data[f'rdpg_proba_mse'] = np.mean((probas_true[:-k_steps] - rdpg.probas_) ** 2)
    data['rdpg_auc'] = rdpg.auc_
    data['rdpg_aupr'] = rdpg.auc_

    y_forecast = rdpg.forecast(k_steps=k_steps, return_subdiag=True)
    y_pred = y_forecast.mean(axis=0)
    data['rdpg_forecast_mse'] = np.mean((probas_true[-1] - y_pred[-1]) ** 2)
    data['rdpg_x_mse'] = mse(X[:-k_steps], rdpg.X_)

    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = f'output'
    dir_name = os.path.join(dir_base, f'd{density}', f"rw{rw_order}_d{n_features}_n{n_nodes}_T{n_time_steps}_d{density}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for rw_order in [1, 2]:
    for n_features in [2, 3, 4, 5, 6]
        for i in range(n_reps):
            simulation(i, n_nodes=200, n_time_steps=50, density=0.2, rw_order=rw_order, n_features=n_features)

