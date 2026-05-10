import numpy as np
import pandas as pd

from graspologic.embed import AdjacencySpectralEmbed as ASE
from joblib import Parallel, delayed
from scipy.special import binom, expit, xlogy, xlog1py
from scipy import stats
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

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


def low_rank_completion(A, k):
    u, s, vt = sp.linalg.svds(sp.csr_array(A), k=k)
    A_tilde = (u * s) @ vt 
    A_tilde[np.diag_indices_from(A_tilde)] = 0
    return A_tilde

def waic_selection_single(Y, rw_order=2, is_binary=True, n_features=2, n_burnin=2500, n_samples=2500):
    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, is_binary=is_binary, random_state=42)
    model.sample(Y, n_burnin=n_burnin, n_samples=n_samples)
    waic, se, p_waic = model.waic(is_binary=is_binary)
    return model, n_features, waic, se, p_waic


def waic_selection(Y, rw_order=2, is_binary=True, min_features=1, max_features=10,
                   n_burnin=500, n_samples=500, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs)(delayed(waic_selection_single)(
        Y=Y, rw_order=rw_order, is_binary=is_binary, n_features=d,
        n_burnin=n_burnin, n_samples=n_samples) for
            d in range(min_features, max_features + 1))
    
    models = [r[0] for r in res] 
    criteria = [r[1:] for r in res]
    colnames = ['n_features', 'waic', 'se', 'p_waic']

    return models, pd.DataFrame(np.asarray(criteria), columns=colnames)


def edge_cv_single(Y_train, y_vec, test_indices, is_binary=True, rw_order=2, n_features=2, n_burnin=2500, n_samples=2500):
    n_time_steps, n_nodes = len(Y_train), Y_train[0].shape[0]

    # create low-rank completed training matrix
    Y_tilde = np.zeros((n_time_steps, n_nodes, n_nodes))
    for t in range(n_time_steps):
        Y_tilde[t] = low_rank_completion(Y_train[t], k=n_features)
    
    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, is_binary=False, random_state=42)
    model.sample(Y_tilde, n_burnin=n_burnin, n_samples=n_samples) 
    
    mse, auc, aupr = 0, 0, 0
    y_true, y_pred = [], []
    for t in range(n_time_steps):
        y_true.append(y_vec[t][test_indices[t]])
        if is_binary:
            y_pred.append(np.clip(model.means_[t][test_indices[t]], 0, 1))
        else:
            y_pred.append(model.means_[t][test_indices[t]])
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    auc = 1. - roc_auc_score(y_true, y_pred)
    aupr = 1. - average_precision_score(y_true, y_pred)

    # pseudo-loglikelihood 
    #scale_hat = model.samples_['scale'].mean()
    #loglik_hat = stats.norm.logpdf(y_true, loc=y_pred, scale=1. / np.sqrt(scale_hat)).sum() 
    loglik = log_loss(y_true, y_pred)

    return n_features, mse, loglik, auc, aupr


def edge_cv_selection(Y, rw_order=2, is_binary=True, min_features=1, max_features=10,
                      n_burnin=500, n_samples=500, p=0.9, n_reps=3, seed=42, n_jobs=-1):
    
    n_time_steps, n_nodes = len(Y),  Y[0].shape[0]
    subdiag = np.tril_indices_from(Y[0], k=-1)
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    rng = np.random.default_rng(seed)

    Y_vec = np.zeros((n_time_steps, n_dyads))
    for t in range(n_time_steps):
        Y_vec[t] = Y[t][subdiag]
    
    criteria = []
    for k in range(n_reps):
        # create training adjacency matrix
        test_indices = []
        Y_train = np.zeros((n_time_steps, n_nodes, n_nodes))
        #train_mask = rng.binomial(1, p=p, size=n_dyads)
        for t in range(n_time_steps):
            train_mask = rng.binomial(1, p=p, size=n_dyads)
            Y_train[t][subdiag] = (1. / p) * (train_mask * Y_vec[t])
            Y_train[t] += Y_train[t].T
            test_indices.append(train_mask == 0)
        
        criteria.append(Parallel(n_jobs=n_jobs)(delayed(edge_cv_single)(
            Y_train=Y_train, y_vec=Y_vec, test_indices=test_indices, is_binary=is_binary, 
            rw_order=rw_order, n_features=d,
            n_burnin=n_burnin, n_samples=n_samples) for
                d in range(min_features, max_features + 1)))

    colnames = ['n_features', 'mse', 'loglik', 'auc', 'aupr']    
    criteria = pd.DataFrame(np.concatenate(criteria), columns=colnames)
    group = criteria.groupby('n_features')
    return group.mean(), group.std()

def backtest_selection_single(Y, n_heldout=5, is_binary=True, rw_order=2, n_features=2, n_burnin=2500, n_samples=2500):

    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, is_binary=is_binary, random_state=42)
    model.sample(Y[:-n_heldout], n_burnin=n_burnin, n_samples=n_samples) 
    
    sparse = False if isinstance(Y, np.ndarray) else True
    y_true = dynamic_adjacency_to_vec(Y[-n_heldout:], sparse=sparse, is_binary=is_binary).ravel()
    y_pred = model.forecast(k_steps=n_heldout, n_samples=1000, return_subdiag=True).mean(axis=0).ravel()
    
    data = {'n_features': n_features}
    data['mse'] = np.mean((y_true - y_pred) ** 2)
    if is_binary:
        data['auc'] = 1. - roc_auc_score(y_true, y_pred)
        data['aupr'] = 1. - average_precision_score(y_true, y_pred)
        data['logloss'] = log_loss(y_true, y_pred)

    return model, data


def backtest_selection(Y, n_heldout=5, rw_order=2, is_binary=True, min_features=1, max_features=10,
                       n_burnin=500, n_samples=500, p=0.9, n_reps=3, seed=42, n_jobs=-1):
        
    res = Parallel(n_jobs=n_jobs)(delayed(backtest_selection_single)(
            Y=Y, n_heldout=n_heldout, is_binary=is_binary, rw_order=rw_order, n_features=d,
            n_burnin=n_burnin, n_samples=n_samples) for
                d in range(min_features, max_features + 1))

    #colnames = ['n_features', 'mse', 'loglik', 'auc', 'aupr']    
    models = [r[0] for r in res] 
    criteria = [r[1] for r in res]
    return models, pd.DataFrame(criteria)


#def loo_selection_single(Y, rw_order=2, is_binary=True, n_features=2, n_burnin=2500, n_samples=2500,
#                         subsample_frac=0.2):
#    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, is_binary=is_binary, random_state=42)
#    model.sample(Y, n_burnin=n_burnin, n_samples=n_samples)
#    
#    loo, se, subsampling_se = model.loo_subsample(subsample_frac=subsample_frac, seed=42)
#    return model, n_features, loo, se, subsampling_se
#
#
#def loo_selection(Y, rw_order=2, is_binary=True, min_features=1, max_features=10,
#                   n_burnin=500, n_samples=500, subsample_frac=0.2, n_jobs=-1):
#    res = Parallel(n_jobs=n_jobs)(delayed(loo_selection_single)(
#        Y=Y, rw_order=rw_order, is_binary=is_binary, n_features=d,
#        n_burnin=n_burnin, n_samples=n_samples, subsample_frac=subsample_frac) for
#            d in range(min_features, max_features + 1))
#    
#    models = [r[0] for r in res] 
#    criteria = [r[1:] for r in res]
#
#    colnames = ['n_features', 'loo', 'se', 'subsampling_se']
#
#    return models, pd.DataFrame(np.asarray(criteria), columns=colnames)


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
