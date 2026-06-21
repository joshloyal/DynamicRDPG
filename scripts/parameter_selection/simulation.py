import plac
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import TruncatedSVD

from graspologic.embed import AdjacencySpectralEmbed as ASE
from graspologic.embed import MultipleASE as MASE
from graspologic.embed import OmnibusEmbed as OMNI

from dynrdpg import DynamicRDPG
from dynrdpg.dynrdpg import calculate_auc
from dynrdpg.datasets import simulate_network_gp_mixture
from dynrdpg.model_selection import backtest_selection
from scipy.stats import mode

def simulation(seed, n_nodes=100, n_time_steps=100, density=0.2, gp_type='matern', nu=0.5, rw_order=2, n_features=2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_steps = int(n_time_steps)
    density = float(density)
    nu = float(nu)
    rw_order = int(rw_order)
    n_features = int(n_features)

    Y, X, probas_true, _ = simulate_network_gp_mixture(
            n_nodes=n_nodes, n_time_steps=n_time_steps, 
            density=density, gp_type=gp_type, nu=nu, 
            n_features=n_features,
            length_scale=3, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)
    X = X[:-5]  # remove last five since held out

    data = {}
    
    models, criteria = backtest_selection(
        Y, rw_order=rw_order, is_binary=True, max_features=6, n_heldout=5,
        n_burnin=2500, n_samples=2500)
         
   
    n_padded = 6 - n_features
    X_padded = np.concatenate([X, np.zeros((X.shape[0], X.shape[1], n_padded))], axis=2)
    
    # selection based on log-loss
    best_idx = np.argmin(criteria['logloss'].values)
    data['n_features_logloss'] = best_idx + 1
    data['waic_logloss'] = criteria['waic'].values[best_idx]
    
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
    data['rdpg_auc_logloss'] = rdpg.auc_
    data['rdpg_aupr_logloss'] = rdpg.aupr_
    data['rdpg_loss_logloss'] = criteria['logloss']
    data['rdpg_mse_logloss'] = np.mean((X_pred - X_true) ** 2)
    
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
    data['rdpg_auc_mse'] = rdpg.auc_
    data['rdpg_aupr_mse'] = rdpg.aupr_
    data['rdpg_loss_mse'] = criteria['mse']
    data['rdpg_mse_mse'] = np.mean((X_pred - X_true) ** 2)


    # also obtain errors for the model using the true value of n_features
    rdpg = models[n_features - 1]
    data['waic_oracle'] = criteria['waic'].values[n_features - 1]
    data['logloss_oracle'] = criteria['logloss'].values[n_features - 1]
    data['mse_oracle'] = criteria['mse'].values[n_features - 1]
    data['rdpg_auc_oracle'] = rdpg.auc_
    data['rdpg_aupr_oracle'] = rdpg.aupr_
    
    # calculate latent space recovery 
    X_true = X
    X_pred = rdpg.X_
    for t in range(len(Y)-5):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R
    
    data['rdpg_mse_oracle'] = np.mean((X_pred - X_true) ** 2)
    
    # selection based on aupr
    best_idx = np.argmin(criteria['aupr'].values)
    data['n_features_aupr'] = best_idx + 1
    data['waic_aupr'] = criteria['waic'].values[best_idx]
    
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
    data['rdpg_auc_aupr'] = rdpg.auc_
    data['rdpg_aupr_aupr'] = rdpg.aupr_
    data['rdpg_loss_aupr'] = criteria['mse']
    data['rdpg_mse_aupr'] = np.mean((X_pred - X_true) ** 2)


    # also obtain errors for the model using the true value of n_features
    rdpg = models[n_features - 1]
    data['waic_oracle'] = criteria['waic'].values[n_features - 1]
    data['logloss_oracle'] = criteria['logloss'].values[n_features - 1]
    data['mse_oracle'] = criteria['mse'].values[n_features - 1]
    data['rdpg_auc_oracle'] = rdpg.auc_
    data['rdpg_aupr_oracle'] = rdpg.aupr_
    
    # calculate latent space recovery 
    X_true = X
    X_pred = rdpg.X_
    for t in range(len(Y)-5):
        R, _ = orthogonal_procrustes(X_pred[t], X_true[t])
        X_pred[t] = X_pred[t] @ R
    
    data['rdpg_mse_oracle'] = np.mean((X_pred - X_true) ** 2)


    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output'
    dir_name = os.path.join(dir_base, f'rw{rw_order}_d{density}_{gp_type}_nu{nu}', f"n{n_nodes}_T{n_time_steps}_d{density}_dim{n_features}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for nu in [0.5, 2.5]:
    for density in [0.1, 0.2, 0.3]:
        for rw_order in [1, 2]:
            for n_features in [2, 4]:
                for i in range(n_reps):
                    simulation(i, n_nodes=200 n_time_steps=50, density=density, nu=nu, rw_order=rw_order, n_features=n_features)
