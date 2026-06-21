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

from dynrdpg.dynrdpg import dynamic_adjacency_to_vec
from dynrdpg.dynrdpg import calculate_auc
from dynrdpg.datasets import simulate_network_gp_continuous


def simulation(seed, n_nodes=100, n_time_steps=100, snr=1., family='poisson', nu=0.5, rw_order=2): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_steps = int(n_time_steps)
    snr = float(snr)
    nu = float(nu)
    rw_order = int(rw_order)

    Y, X, probas_true = simulate_network_gp_continuous(
            n_nodes=n_nodes, n_time_steps=n_time_steps, family=family,
            snr=snr, gp_type='matern', nu=nu, 
            length_scale=3, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)

    data = {}

    # Dynamic RDPG
    rdpg = DynamicRDPG(n_features=2, rw_order=rw_order, is_binary=False, random_state=42)
    rdpg.sample(Y, n_burnin=2500, n_samples=2500)

    X_pred = rdpg.X_.copy()
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R

    # in-sample performance
    data['rdpg_error'] = rdpg.rmse_
    data['rpdg_mse'] = np.mean((X_pred - X) ** 2)


    # ASE estimator
    mean_ase= []
    X_ase = []
    for t in range(len(Y)):
        ase = ASE(n_components=2)
        X_ase.append(ase.fit_transform(Y[t]))
        R, _ = orthogonal_procrustes(X_ase[t], X[t])
        X_ase[t] = X_ase[t] @ R
        mean_ase.append((X_ase[t] @ X_ase[t].T)[subdiag])
    X_ase = np.stack(X_ase)
    mean_ase = np.stack(mean_ase)
    
    # in-sample performance
    y_vec = dynamic_adjacency_to_vec(Y, sparse=False)
    data['ase_error'] = np.sqrt(np.mean((y_vec - mean_ase) ** 2))
    data['ase_mse'] = np.mean((X_ase - X) ** 2)

    # OMNI
    X_omni = OMNI(n_components=2).fit_transform(Y)
    mean_omni = []
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_omni[t], X[t])
        X_omni[t] = X_omni[t] @ R
        mean_omni.append((X_omni[t] @ X_omni[t].T)[subdiag])
    mean_omni = np.stack(mean_omni)
 
    data['omni_error'] = np.sqrt(np.mean((y_vec - mean_omni) ** 2))
    data['omni_mse'] = np.mean((X_omni - X) ** 2)

    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output'
    dir_name = os.path.join(dir_base, f'{family}', f'rw{rw_order}_s{snr}_nu{nu}', f"n{n_nodes}_T{n_time_steps}_s{snr}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for family in ['laplace', 'poisson']:
    for nu in [0.5, 2.5]:
        for n_nodes in [25, 50, 100, 200, 400]:
            for snr in [0.5, 1.0, 2.0]
                for i in range(n_reps):
                    for rw_order in [1, 2]:
                        simulation(i, n_nodes=n_nodes, n_time_steps=50, snr=snr, nu=nu, rw_order=rw_order)


for family in ['laplace', 'poisson']:
    for nu in [0.5, 2.5]:
        for n_time_steps in [10, 25, 50, 100, 200]:
            for snr in [0.5, 1.0, 2.0]
                for i in range(n_reps):
                    for rw_order in [1, 2]:
                        simulation(i, n_nodes=50, n_time_steps=n_time_steps, snr=snr, nu=nu, rw_order=rw_order)
