import plac
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

from graspologic.embed import AdjacencySpectralEmbed as ASE
from graspologic.embed import MultipleASE as MASE
from graspologic.embed import OmnibusEmbed as OMNI

from dynrdpg import DynamicRDPG
from dynrdpg.dynrdpg import calculate_auc
from dynrdpg.datasets import simulate_network_bspline


def simulation(seed, n_nodes=100, n_time_steps=100, density=0.2): 
    k_steps = 5
    k_buffer = 10

    Y, X, probas_true = simulate_network_bspline(
            n_nodes=n_nodes, n_time_steps=n_time_steps, 
            density=density, df=5, k_buffer=k_buffer, 
            k_steps=k_steps, random_state=seed) 
    subdiag = np.tril_indices(Y[0].shape[0], k=-1)

    data = {}

    # GB-DASE [RW(2)] 
    rdpg = DynamicRDPG(n_features=2, rw_order=2, random_state=42)
    #rdpg.sample(Y[:-k_steps], n_burnin=2500, n_samples=2500)
    rdpg.sample(Y[:-k_steps], n_burnin=100, n_samples=100)

    X_pred = rdpg.X_.copy()
    for t in range(len(Y)-k_steps):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R

    # forecasting
    y_forecast = rdpg.forecast(k_steps=k_steps, return_subdiag=True)
    y_pred = y_forecast.mean(axis=0)
    for k in range(k_steps):
        y_pred_k = y_pred[k]
        y_true = Y[k-k_steps].toarray()[subdiag]
        probas_true_k = probas_true[k-k_steps][subdiag]
        data[f'rdpg_rw2_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    
    
    # GB-DASE [RW(1)] 
    rdpg = DynamicRDPG(n_features=2, rw_order=1, random_state=42)
    #rdpg.sample(Y[:-k_steps], n_burnin=2500, n_samples=2500)
    rdpg.sample(Y[:-k_steps], n_burnin=100, n_samples=100)

    X_pred = rdpg.X_.copy()
    for t in range(len(Y)-k_steps):
        R, _ = orthogonal_procrustes(X_pred[t], X[t])
        X_pred[t] = X_pred[t] @ R

    # forecasting
    y_forecast = rdpg.forecast(k_steps=k_steps, return_subdiag=True)
    y_pred = y_forecast.mean(axis=0)
    for k in range(k_steps):
        y_pred_k = y_pred[k]
        y_true = Y[k-k_steps].toarray()[subdiag]
        probas_true_k = probas_true[k-k_steps][subdiag]
        data[f'rdpg_rw1_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
 
    # ASE
    proba_ase= []
    X_ase = []
    for t in range(len(Y)-k_steps):
        ase = ASE(n_components=2)
        X_ase.append(ase.fit_transform(Y[t]))
        R, _ = orthogonal_procrustes(X_ase[t], X[t])
        X_ase[t] = X_ase[t] @ R
        proba_ase.append(np.clip((X_ase[t] @ X_ase[t].T)[subdiag], 0, 1))
    X_ase = np.stack(X_ase)
    proba_ase = np.stack(proba_ase)

    # forecasting
    for k in range(k_steps):
        y_pred_k = proba_ase[-1]
        y_true = Y[-k_steps + k].toarray()[subdiag]
        probas_true_k = probas_true[-k_steps + k][subdiag]
        data[f'ase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 


    # unfolded adjacency spectral embedding (UASE)
    n_nodes, _ = Y[0].shape
    A = sp.csr_array(np.hstack([Yt.toarray() for Yt in Y[:-k_steps]]))
    u, s, vh= sp.linalg.svds(A, k=2)
    s_sqrt = np.sqrt(s)[::-1]
    X_uase = (vh.T[:, ::-1] * s_sqrt).reshape(len(Y[:-k_steps]), n_nodes, 2)
    Y_uase = u[:, ::-1] * s_sqrt

    # insample performance
    proba_uase= []
    for t in range(len(Y)-k_steps):
        proba_uase.append(np.clip(Y_uase @ X_uase[t].T, 0, 1)[subdiag])
        R, _ = orthogonal_procrustes(X_uase[t], X[t])
        X_uase[t] = X_uase[t] @ R
    proba_uase = np.stack(proba_uase)

    # forecasting
    for k in range(k_steps):
        y_pred_k = proba_uase[-1]
        y_true = Y[-k_steps + k].toarray()[subdiag]
        probas_true_k = probas_true[-k_steps + k][subdiag]
        data[f'uase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 


    # MASE 
    mase = MASE(n_components=2).fit(Y[:-k_steps])
    X_mase_fit = mase.latent_left_

    proba_mase = []
    X_mase = []
    for t in range(len(Y)-k_steps):
        R, _ = orthogonal_procrustes(X_mase_fit, X[t])
        X_mase.append(X_mase_fit @ R)
        proba_mase.append(np.clip(mase.latent_left_ @ mase.scores_[t] @ mase.latent_left_.T, 0, 1)[subdiag])
    X_mase = np.stack(X_mase)
    proba_mase = np.stack(proba_mase)

    # forecasting
    for k in range(k_steps):
        y_pred_k = proba_mase[-1]
        y_true = Y[-k_steps + k].toarray()[subdiag]
        probas_true_k = probas_true[-k_steps + k][subdiag]
        data[f'mase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 


    # OMNI
    X_omni = OMNI(n_components=2).fit_transform(Y[:-k_steps])
    proba_omni = []
    for t in range(len(Y)-k_steps):
        R, _ = orthogonal_procrustes(X_omni[t], X[t])
        X_omni[t] = X_omni[t] @ R
        proba_omni.append(np.clip(X_omni[t] @ X_omni[t].T, 0, 1)[subdiag])
    proba_omni = np.stack(proba_omni)


    # forecasting
    for k in range(k_steps):
        y_pred_k = proba_omni[-1]
        y_true = Y[-k_steps + k].toarray()[subdiag]
        probas_true_k = probas_true[-k_steps + k][subdiag]
        data[f'omni_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 


    data = pd.DataFrame(data, index=[0])

    out_file = f'result_{seed}.csv'
    dir_base = f'output'
    dir_name = os.path.join(dir_base, f'd{density}', f"n{n_nodes}_T{n_time_steps}_d{density}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for n_time_steps in [100, 150]:
    for density in [0.1, 0.2, 0.3]:
        for i in range(n_reps):
            simulation(i, n_nodes=200, n_time_steps=n_time_steps, density=density)
