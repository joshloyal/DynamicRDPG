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
from dynrdpg.datasets.synthetic import simulate_network_gp_density


def simulation(seed, n_nodes=100, n_time_steps=100, rw_order=2, dens_type='increasing'): 
    seed = int(seed)
    n_nodes = int(n_nodes)
    n_time_steps = int(n_time_steps)
    rw_order = int(rw_order)

    Y, X, probas_true = simulate_network_gp_density(
            n_nodes=n_nodes, n_time_steps=n_time_steps, dens_type=dens_type,
            density_min=0.1, density_max=0.5, gp_type='matern', nu=2.5, 
            length_scale=3, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)

    data = {}

    # Dynamic RDPG
    rdpg = DynamicRDPG(n_features=2, rw_order=rw_order, random_state=42)
    rdpg.sample(Y, n_burnin=2500, n_samples=2500)

    X_pred = rdpg.X_.copy()
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R

    # in-sample performance
    data['rdpg_auc'] = rdpg.auc_
    data['rdpg_aupr'] = rdpg.aupr_
    data['rpdg_mse'] = np.mean((X_pred - X) ** 2)

    for t in range(len(Y)):
        data[f'rdpg_mse_{t}'] = np.mean((X_pred[t] - X[t]) ** 2)

    # ASE
    X_ase = []
    for t in range(len(Y)):
        X_ase.append(ASE(n_components=2).fit_transform(sp.csr_array(Y[t])))
        R, _ = orthogonal_procrustes(X_ase[t], X[t])
        X_ase[t] = X_ase[t] @ R
    X_ase = np.stack(X_ase)
    data['ase_mse'] = np.mean((X_ase - X) ** 2)

    for t in range(len(Y)):
        data[f'ase_mse_{t}'] = np.mean((X_ase[t] - X[t]) ** 2)

    # OMNI
    X_omni = OMNI(n_components=2).fit_transform(Y)
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_omni[t], X[t])
        X_omni[t] = X_omni[t] @ R
        data[f'omni_mse_{t}'] = np.mean((X_omni[t] - X[t]) ** 2)
    data['omni_mse'] = np.mean((X_omni - X) ** 2)

    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output'
    dir_name = os.path.join(dir_base, f"{dens_type}", f"rw{rw_order}_n{n_nodes}_T{n_time_steps}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for rw_order in [1, 2]:
    for dens_type in ['increasing', 'decreasing', 'logistic']:
        for i in range(n_reps):
            simulation(i, n_nodes=400, n_time_steps=50, rw_order=rw_order, dens_type=dens_type)
