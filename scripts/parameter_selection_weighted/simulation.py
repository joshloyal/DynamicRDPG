import plac
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import TruncatedSVD

from dynrdpg import DynamicRDPG
from dynrdpg.dynrdpg import calculate_auc
from dynrdpg.datasets import simulate_network_gp_continuous
from dynrdpg.datasets.synthetic import simulate_network_gp_continuous_rw
from dynrdpg.model_selection import waic_selection, backtest_selection
from scipy.stats import mode

def simulation(seed, n_nodes=100, n_time_steps=100, family='laplace', snr=10., sim_rw_order=1, rw_order=2, n_features=2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_steps = int(n_time_steps)
    rw_order = int(rw_order)
    sim_rw_order = int(sim_rw_order)
    n_features = int(n_features)
    snr = float(snr)
    
    if sim_rw_order == 2:
        Y, X, means_true = simulate_network_gp_continuous(
                n_nodes=n_nodes, n_time_steps=n_time_steps, 
                snr=snr, gp_type='rbf',
                n_features=n_features, family=family, 
                length_scale=3, random_state=seed) 
    else:
        Y, X, means_true = simulate_network_gp_continuous_rw(
                n_nodes=n_nodes, n_time_steps=n_time_steps, 
                snr=snr, rho=0.95, rw_order=1,
                n_features=n_features, family=family, 
                random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)
    X = X[:-5]  # remove last five since held out

    data = {}
    
    models, criteria = backtest_selection(
        Y, rw_order=rw_order, is_binary=False, max_features=6, n_heldout=5,
        n_burnin=2500, n_samples=2500)
         
    n_padded = 6 - n_features
    X_padded = np.concatenate([X, np.zeros((X.shape[0], X.shape[1], n_padded))], axis=2)
    
    # selection based on mse
    best_idx = np.argmin(criteria['mse'].values)
    data['n_features_mse'] = best_idx + 1
    data['waic_mse'] = criteria['waic'].values[best_idx]
    
    rdpg = models[best_idx]
    n_padded = 6 - rdpg.n_features
    if n_padded > 0:
        X_pred_padded = np.concatenate([rdpg.X_, np.zeros((X.shape[0], X.shape[1], n_padded))], axis=2)
    else:
        X_pred_padded = rdpg.X_
    
    d_max = max(rdpg.n_features, n_features)
    X_true = X_padded[..., :d_max]
    X_pred = X_pred_padded[..., :d_max]
    
    # calculate latent space recovery (with padding)
    for t in range(len(Y)-5):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R

    # in-sample performance
    data['rdpg_loss_mse'] = criteria['mse']
    data['rdpg_mse_mse'] = np.mean((X_pred - X_true) ** 2)


    # also obtain errors for the model using the true value of n_features
    rdpg = models[n_features - 1]
    data['waic_oracle'] = criteria['waic'].values[n_features - 1]
    data['mse_oracle'] = criteria['mse'].values[n_features - 1]
    
    # calculate latent space recovery 
    X_true = X
    X_pred = rdpg.X_
    for t in range(len(Y)-5):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R
    
    data['rdpg_mse_oracle'] = np.mean((X_pred - X_true) ** 2)
    
    # also obtain errors for the model using the true value of n_features
    rdpg = models[n_features - 1]
    data['waic_oracle'] = criteria['waic'].values[n_features - 1]
    data['mse_oracle'] = criteria['mse'].values[n_features - 1]
    
    # calculate latent space recovery 
    X_true = X
    X_pred = rdpg.X_
    for t in range(len(Y)-5):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R
    
    data['rdpg_mse_oracle'] = np.mean((X_pred - X_true) ** 2)

    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = f'output/{family}'
    dir_name = os.path.join(dir_base, f'sim_rw{sim_rw_order}', f"rw{rw_order}_n{n_nodes}_T{n_time_steps}_s{snr}_dim{n_features}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for family in ['poisson', 'laplace']:
    for sim_rw_order in [1, 2]:
        for snr in [1.0, 0.5, 0.05]:
            for rw_order in [1, 2]:
                for n_features in [2, 4]:
                    for i in range(n_reps):
                        simulation(i, n_nodes=200 n_time_steps=50, family=family, snr=snr, sim_rw_order=sim_rw_order, rw_order=rw_order, n_features=n_features)
