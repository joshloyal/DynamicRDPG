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
from dynrdpg.datasets import simulate_network_gp


k_steps = 5
n_time_steps = 100
n_nodes = 100
density = 0.1
rw_order = 2

Y, X, probas_true = simulate_network_gp(
        n_nodes=n_nodes, n_time_steps=n_time_steps + k_steps, density=density,
        random_state=2, length_scale=3) 
subdiag = np.tril_indices(Y[0].shape[0], k=-1)

data = {}

# Dynamic RDPG
rdpg = DynamicRDPG(n_features=2, rw_order=rw_order, random_state=42)
rdpg.sample(Y[:-k_steps], n_burnin=2500, n_samples=2500)

X_pred = rdpg.X_.copy()
for t in range(len(Y)-k_steps):
    R, _ = orthogonal_procrustes(X_pred[t], X[t])
    X_pred[t] = X_pred[t] @ R

# in-sample performance
data['rdpg_auc'] = rdpg.auc_
data['rpdg_mse'] = np.mean((X_pred - X[:-k_steps]) ** 2)

# forecasting
y_pred = rdpg.forecast(k_steps=k_steps).mean(axis=0)
for k in range(k_steps):
    y_pred_k = y_pred[k][subdiag]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'rdpg_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'rdpg_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'rdpg_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)
 
 # Naive Forecast
for k in range(k_steps):
    y_pred_k = Y[-k_steps - 1].toarray()[subdiag]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'naive_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'naive_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'naive_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)


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

# in-sample performance
data['ase_auc'] = calculate_auc(rdpg.y_vec_, proba_ase)
data['ase_mse'] = np.mean((X_ase - X[:-k_steps]) ** 2)

# forecasting
for k in range(k_steps):
    y_pred_k = proba_ase[-1]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'ase_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'ase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'ase_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)


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

data['uase_auc'] = calculate_auc(rdpg.y_vec_, proba_uase)
data['uase_mse'] = np.mean((X_uase - X[:-k_steps]) ** 2)

# forecasting
for k in range(k_steps):
    y_pred_k = proba_uase[-1]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'uase_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'uase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'uase_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)


# MASE (or COSIE)
mase = MASE(n_components=2).fit(Y[:-k_steps])
X_mase_fit = mase.latent_left_

proba_mase = []
X_mase = []
for t in range(len(Y)-k_steps):
    #if t == 0:
    #    R, _ = orthogonal_procrustes(X_mase_fit, X[t])
    #    X_mase_fit = X_mase_fit @ R
    #X_mase.append(X_mase_fit)
    R, _ = orthogonal_procrustes(X_mase_fit, X[t])
    X_mase.append(X_mase_fit @ R)

    #X_t = mase.latent_left_ @ mase.scores_[t]
    #R, _ = orthogonal_procrustes(X_t, X[t])
    #X_mase.append(X_t @ R)
    proba_mase.append(np.clip(mase.latent_left_ @ mase.scores_[t] @ mase.latent_left_.T, 0, 1)[subdiag])
X_mase = np.stack(X_mase)
proba_mase = np.stack(proba_mase)

data['mase_auc'] = calculate_auc(rdpg.y_vec_, proba_mase)
data['mase_mse'] = np.mean((X_mase - X[:-k_steps]) ** 2)

# forecasting
for k in range(k_steps):
    y_pred_k = proba_mase[-1]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'mase_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'mase_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'mase_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)


# OMNI
X_omni = OMNI(n_components=2).fit_transform(Y[:-k_steps])
proba_omni = []
for t in range(len(Y)-k_steps):
    R, _ = orthogonal_procrustes(X_omni[t], X[t])
    X_omni[t] = X_omni[t] @ R
    proba_omni.append(np.clip(X_omni[t] @ X_omni[t].T, 0, 1)[subdiag])
proba_omni = np.stack(proba_mase)

    
data['omni_auc'] = calculate_auc(rdpg.y_vec_, proba_omni)
data['omni_mse'] = np.mean((X_omni - X[:-k_steps]) ** 2)

# forecasting
for k in range(k_steps):
    y_pred_k = proba_omni[-1]
    y_true = Y[-k_steps + k].toarray()[subdiag]
    probas_true_k = probas_true[-k_steps + k][subdiag]
    data[f'omni_kstep_auc_{k+1}'] = roc_auc_score(y_true, y_pred_k)
    data[f'omni_kstep_mse_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) 
    data[f'omni_kstep_rel_{k+1}'] = np.mean((probas_true_k - y_pred_k) ** 2) / np.mean(probas_true_k ** 2)
