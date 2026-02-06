import numpy as np
import pandas as pd

from graspologic.embed import AdjacencySpectralEmbed as ASE
from joblib import Parallel, delayed
from scipy.special import binom, expit, xlogy, xlog1py
from scipy import stats

from .dynrdpg import DynamicRDPG, dynamic_adjacency_to_vec


def clamp_probs(probs):
    finfo = np.finfo(np.result_type(probs, float))
    #return np.clip(probs, finfo.tiny, 1.0 - finfo.eps)
    return np.clip(probs, finfo.eps, 1.0 - finfo.eps)


def bernoulli_logp(y, probs):
    ps_clamped = clamp_probs(probs)
    y = np.array(y, np.result_type(float))
    return xlogy(y, ps_clamped) + xlog1py(1 - y, -ps_clamped)


def ase(A, k=2):
    eigvals, eigvec = sp.linalg.eigsh(A, k=k)
    return eigvec[:, ::-1] * np.sqrt(np.abs(eigvals)[::-1])


def waic_selection_single(Y, rw_order=2, is_binary=True, n_features=2, n_burnin=2500, n_samples=2500):
    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, is_binary=is_binary, random_state=42)
    model.sample(Y, n_burnin=n_burnin, n_samples=n_samples)

    #if is_binary:
    #    return model, n_features, model.waic(is_binary=True), model.waic(), model.gcv()
    #else:
    #    return model, n_features, model.waic(), model.gcv()
    waic, se, p_waic = model.waic()
    return model, n_features, waic, se, p_waic


def waic_selection(Y, rw_order=2, is_binary=True, min_features=1, max_features=10,
                   n_burnin=500, n_samples=500, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs)(delayed(waic_selection_single)(
        Y=Y, rw_order=rw_order, is_binary=is_binary, n_features=d,
        n_burnin=n_burnin, n_samples=n_samples) for
            d in range(min_features, max_features + 1))
    
    models = [r[0] for r in res] 
    criteria = [r[1:] for r in res]

    #if is_binary:
    #    colnames = ['n_features', 'bernoulli waic', 'gaussian waic', 'gcv']
    #else:
    #    colnames = ['n_features', 'waic', 'gcv']
    colnames = ['n_features', 'waic', 'se', 'p_waic']

    return models, pd.DataFrame(np.asarray(criteria), columns=colnames)


def ase_one_step(Y, k):
    X = ASE(n_components=k).fit_transform(Y)
   
    P = X @ X.T
    P_clamped = clamp_probs(P)
    V = P_clamped * (1 - P_clamped)
    grad = ((Y - P) / V) @ X
    
    X_os = np.zeros(X.shape)
    for i in range(X.shape[0]):
        v = X / np.sqrt(V[i])[:, np.newaxis]
        hess = v.T @ v
        X_os[i] = X[i] + np.linalg.pinv(hess) @ grad[i]
    
    return X_os




def jic_single(Y, k=2, is_sparse=True, is_binary=False):
    
    n_time_steps = len(Y)
    n_nodes = Y[0].shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    y_vec = dynamic_adjacency_to_vec(Y, sparse=is_sparse)

    subdiag = np.tril_indices(n_nodes, k=-1)
    sse = 0.
    X_ase = []
    for t in range(len(Y)):
        X_ase.append(ASE(n_components=k).fit_transform(Y[t]))
        #X_ase.append(ase_one_step(Y[t], k=k))
        #sse += np.sum((Y[t][subdiag] - (X_ase[t] @ X_ase[t].T)[subdiag]) ** 2)
        #sse += np.sum((Y[t] - X_ase[t] @ X_ase[t].T) ** 2)
    X_ase = np.stack(X_ase)
    XXt = np.einsum('tid,tjd->tij', X_ase, X_ase)[..., subdiag[0], subdiag[1]]

    #sigma_sq = sse / (n_time_steps * (n_dyads - n_nodes * k ))
    #sigma_sq = sse / (n_time_steps * n_nodes * (n_nodes - k ))
    #sigma_sq = np.var(y_vec)

    #loglik = -0.5 * (sse / sigma_sq) - 0.5 * n_time_steps * (n_nodes ** 2) * np.log(sigma_sq)
    #loglik = -0.5 * (sse / sigma_sq) - 0.5 * n_time_steps * n_dyads * np.log(2 * np.pi * sigma_sq)

    if is_binary:
        loglik = bernoulli_logp(y_vec, XXt)
    else:
        sigma_sq = np.mean((y_vec - XXt) ** 2)
        loglik = stats.norm.logpdf(y_vec, loc=XXt, scale=np.sqrt(sigma_sq))
    
    #c = np.log(np.prod(loglik.shape))
    #c = np.log(n_time_steps * n_nodes)
    #c = np.log(n_nodes)
    #c = np.log(n_nodes)
    #return -2 * loglik + k * n_nodes * n_time_steps * c#k * n_nodes * n_time_steps * np.log(n_nodes)
    #return -2 * loglik.sum() + k * n_nodes * c#k * n_nodes * n_time_steps * np.log(n_nodes)
    #return -2 * loglik.sum() + k * n_nodes  * c
    return -2 * loglik.sum() + k * n_nodes * np.log(n_nodes)


def ebic_single(Y, k=2, k_max=10):
    n_time_steps = len(Y)
    n_nodes = Y[0].shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    y_vec = dynamic_adjacency_to_vec(Y, sparse=True)

    subdiag = np.tril_indices(n_nodes, k=-1)
    sse = 0.
    X_ase = []
    for t in range(len(Y)):
        X_ase.append(ASE(n_components=k).fit_transform(Y[t]))
        sse += np.sum((Y[t][subdiag] - (X_ase[t] @ X_ase[t].T)[subdiag]) ** 2)
    X_ase = np.stack(X_ase)

    #sigma_sq = sse / (n_time_steps * (n_dyads - n_nodes * k ))
    #sigma_sq = sse / (n_time_steps * n_nodes * (n_nodes - k ))
    #sigma_sq = np.var(y_vec)

    #loglik = -0.5 * (sse / sigma_sq) - 0.5 * n_time_steps * (n_nodes ** 2) * np.log(sigma_sq)
    #loglik = -0.5 * (sse / sigma_sq) - 0.5 * n_time_steps * n_dyads * np.log(2 * np.pi * sigma_sq)
    
    p_1 = np.log(n_time_steps * n_dyads)
    return -2 * loglik + k * n_nodes * n_time_steps * np.log(n_time_steps * n_dyads) + np.log(binom(k_max, k))
