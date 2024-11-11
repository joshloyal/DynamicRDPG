import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from scipy.special import expit


def uniform_simplex(n_nodes, n_features, random_state=None):
    rng = check_random_state(random_state)

    v = rng.gamma(0.5, 1, size=(n_nodes, n_features))
    vd = rng.gamma(1, 1, size=n_nodes)

    return np.sqrt(v / (v.sum(axis=1) + vd).reshape(-1, 1))


def simulate_network_rw(n_nodes=50, n_time_steps=20,  
                         sigma=0.025, density=0.2, random_state=2):
    rng = check_random_state(random_state)
    n_features = 2
    
    X = np.zeros((n_time_steps, n_nodes, n_features))
    
    # initial positions sampled uniformly from positive orthant
    X[0] =  uniform_simplex(n_nodes, n_features, random_state=rng)
        
    
    subdiag = np.tril_indices(n_nodes, k=-1)
    probas = []
    probas.append((X[0] @ X[0].T)[subdiag])
    
    for t in range(1, n_time_steps):
        X[t] = X[t-1] + sigma * rng.randn(n_nodes, n_features)
        
        # project latent positions onto the simplex
        X[t][X[t] < 0] = 0.
        norms = np.linalg.norm(X[t], axis=1)[:, None]
        X[t] = np.where(norms > 1, X[t] / norms, X[t])
        
        probas.append((X[t] @ X[t].T)[subdiag])
    
    scale = density / np.mean(probas)          
    Ys = []    
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))
        
        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * probas[t], 0, 1))

        Y[subdiag] = y
        Y += Y.T
        
        Y = sp.csr_array(Y, dtype=float)
        Ys.append(Y)

    return Ys, X


def simulate_network_gp(n_nodes=100, n_time_steps=100, n_features=2, density=0.2, random_state=42):
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps)
    dist_sq = euclidean_distances(ts[:, None], squared=True)
    ls = (3./n_time_steps) ** 2
    C = 3. * np.exp(-0.5 * dist_sq * ls )
    X = rng.multivariate_normal(np.zeros_like(ts), cov=C,
                                size=(n_nodes, n_features)).transpose((2, 0, 1))
    X = expit(X) / np.sqrt(n_features)

    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        means.append((X[t] @ X[t].T)[subdiag])

    scale = density / np.mean(means)

    Ys = []
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))

        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * means[t], 0, 1))

        Y[subdiag] = y
        Y += Y.T

        Y = sp.csr_array(Y, dtype=float) 
        Ys.append(Y)

    return Ys, X
