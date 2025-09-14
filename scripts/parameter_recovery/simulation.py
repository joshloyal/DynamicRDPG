import plac
import os
import pandas as pd
import numpy as np

from scipy.linalg import orthogonal_procrustes

from graspologic.embed import AdjacencySpectralEmbed as ASE
from graspologic.embed import OmnibusEmbed as OMNI

from dynrdpg import DynamicRDPG
from dynrdpg.datasets import simulate_network_gp


def simulation(seed, n_nodes=100, n_time_steps=100, density=0.2, gp_type='matern', nu=0.5): 
    Y, X, probas_true = simulate_network_gp(
            n_nodes=n_nodes, n_time_steps=n_time_steps, 
            density=density, gp_type=gp_type, nu=nu, 
            length_scale=3, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)

    data = {}

    # GB-DASE [RW(1)]
    rdpg_rw1 = DynamicRDPG(n_features=2, rw_order=1, random_state=42)
    rdpg_rw1.sample(Y, n_burnin=2500, n_samples=2500)

    X_pred = rdpg_rw1.X_.copy()
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R
    data['rw1_rdpg_mse'] = np.mean((X_pred - X) ** 2)
    
    # GB-DASE [RW(2)]
    rdpg_rw2 = DynamicRDPG(n_features=2, rw_order=2, random_state=42)
    rdpg_rw2.sample(Y, n_burnin=2500, n_samples=2500)

    X_pred = rdpg_rw2.X_.copy()
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R
    data['rw2_rdpg_mse'] = np.mean((X_pred - X) ** 2)

    # ASE
    X_ase = []
    for t in range(len(Y)):
        ase = ASE(n_components=2)
        X_ase.append(ase.fit_transform(Y[t]))
        R, _ = orthogonal_procrustes(X_ase[t], X[t])
        X_ase[t] = X_ase[t] @ R
    X_ase = np.stack(X_ase)
    data['ase_mse'] = np.mean((X_ase - X) ** 2)

    # OMNI
    X_omni = OMNI(n_components=2).fit_transform(Y)
    proba_omni = []
    for t in range(len(Y)):
        R, _ = orthogonal_procrustes(X_omni[t], X[t])
        X_omni[t] = X_omni[t] @ R
        proba_omni.append(np.clip(X_omni[t] @ X_omni[t].T, 0, 1)[subdiag])
    data['omni_mse'] = np.mean((X_omni - X) ** 2)

    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = 'output'
    dir_name = os.path.join(dir_base, f'd{density}_{gp_type}_nu{nu}', f"n{n_nodes}_T{n_time_steps}_d{density}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for nu in [0.5, 2.5]:
    for n_nodes in [25, 50, 100, 200, 400]:
        for density in [0.1, 0.2, 0.3]:
            for i in range(n_reps):
                simulation(i, n_nodes=n_nodes, n_time_steps=50, density=density, nu=nu)

for nu in [0.5, 2.5]:
    for n_nodes in [10, 25, 50, 100, 200]:
        for density in [0.1, 0.2, 0.3]:
            for i in range(n_reps):
                simulation(i, n_nodes=50, n_time_steps=n_time_steps, density=density, nu=nu)
