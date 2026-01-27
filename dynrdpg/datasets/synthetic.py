import numpy as np
import scipy.sparse as sp

from patsy import cr
from sklearn.gaussian_process.kernels import RBF, Matern 
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import SplineTransformer
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
    V = np.zeros((n_time_steps, n_nodes, n_features))
    
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
    probas_out = []
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))
        
        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * probas[t], 0, 1))
        probas_out.append(np.clip(X[t] @ X[t].T, 0, 1))

        Y[subdiag] = y
        Y += Y.T
        
        Y = sp.csr_array(Y, dtype=float)
        Ys.append(Y)

    return Ys, X, np.stack(probas_out)


def simulate_network_gp(n_nodes=100, n_time_steps=100, n_features=2, 
                        length_scale=3, gp_type='matern', nu=2.5,
                        density=0.2, random_state=42):
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps).reshape(-1, 1)
    dist_sq = euclidean_distances(ts, squared=True)
    
    if gp_type == 'rbf':
        cov = 5 * RBF(length_scale=n_time_steps/length_scale)(ts)
    else:
        cov = 5 * Matern(length_scale=n_time_steps/length_scale, nu=nu)(ts)
    
    #ls = (length_scale/n_time_steps) ** 2 
    #C = 5 * np.exp(-0.5 * dist_sq * ls )

    X = rng.multivariate_normal(np.zeros_like(ts.ravel()), cov=cov,
                                size=(n_nodes, n_features)).transpose((2, 0, 1))
    X = expit(X) / np.sqrt(n_features)

    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        means.append((X[t] @ X[t].T)[subdiag])

    scale = density / np.mean(means)

    Ys = []
    probas = []
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))

        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * means[t], 0, 1))
        probas.append(np.clip(X[t] @ X[t].T, 0, 1))

        Y[subdiag] = y
        Y += Y.T

        Y = sp.csr_array(Y, dtype=float) 
        Ys.append(Y)

    return Ys, X, np.stack(probas)


def simulate_network_gp(n_nodes=100, n_time_steps=100, n_features=2, 
                        length_scale=3, gp_type='matern', nu=2.5,
                        density=0.2, random_state=42):
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps).reshape(-1, 1)
    dist_sq = euclidean_distances(ts, squared=True)
    
    if gp_type == 'rbf':
        cov = 5 * RBF(length_scale=n_time_steps/length_scale)(ts)
    else:
        cov = 5 * Matern(length_scale=n_time_steps/length_scale, nu=nu)(ts)
    
    #ls = (length_scale/n_time_steps) ** 2 
    #C = 5 * np.exp(-0.5 * dist_sq * ls )

    X = rng.multivariate_normal(np.zeros_like(ts.ravel()), cov=cov,
                                size=(n_nodes, n_features)).transpose((2, 0, 1))
    X = expit(X) / np.sqrt(n_features)

    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        means.append((X[t] @ X[t].T)[subdiag])
    
    scale = density / np.mean(means)

    Ys = []
    probas = []
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))

        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * means[t], 0, 1))
        probas.append(np.clip(X[t] @ X[t].T, 0, 1))

        Y[subdiag] = y
        Y += Y.T

        Y = sp.csr_array(Y, dtype=float) 
        Ys.append(Y)

    return Ys, X, np.stack(probas)


def simulate_network_gp_continuous(n_nodes=100, n_time_steps=100, n_features=2, 
                        length_scale=3, gp_type='matern', nu=2.5, #nonzero_proba=0.8,
                        family='poisson', snr=1., random_state=42):
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps).reshape(-1, 1)
    dist_sq = euclidean_distances(ts, squared=True)
    
    if gp_type == 'rbf':
        cov = np.sqrt(snr) * RBF(length_scale=n_time_steps/length_scale)(ts)
    else:
        cov = np.sqrt(snr) * Matern(length_scale=n_time_steps/length_scale, nu=nu)(ts)
    
    X = rng.multivariate_normal(np.zeros_like(ts.ravel()), cov=cov,
                                size=(n_nodes, n_features)).transpose((2, 0, 1))

    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps):
        means.append((X[t] @ X[t].T)[subdiag])
    
    Ys = []
    sigma  = np.sqrt(n_features)
    for t in range(n_time_steps):
        Y = np.zeros((n_nodes, n_nodes))
        
        if family == 'poisson':
            errors = rng.poisson(sigma, size=means[t].shape[0]) - sigma 
        elif family == 'laplace':
            errors = rng.laplace(loc=0, scale=sigma / np.sqrt(2), size=means[t].shape[0])  
        else:
            errors = sigma * rng.randn(means[t].shape[0])
        
        #nonzero_mask = rng.binomial(1, p=nonzero_proba, size=means[t].shape[0])
        #y = nozero_mask * (means[t] + errors)
        y = means[t] + errors
        Y[subdiag] = y
        Y += Y.T

        Ys.append(Y)

    return np.asarray(Ys), X, np.stack(means)#, sigma



def simulate_network_ncr(n_nodes=100, n_time_steps=100, n_features=2, 
                         df=5, k_steps=5, density=0.2, random_state=42):
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps + k_steps)
    spline_basis = cr(ts, df, lower_bound=0, upper_bound=n_time_steps)
    w = rng.dirichlet(np.ones(df) / df, n_nodes * n_features).T
    X = spline_basis @ w #.reshape(-1, n_nodes, n_features) 

    X_min = np.min(X, axis=0)
    idx = np.where(X_min < 0)[0]
    X[:, idx] -= X_min[idx]
    X /= np.max(X, axis=0)
     
    X = X.reshape(-1, n_nodes, n_features)
    X /= np.sqrt(n_features)
    
    #w = rng.randn(df, n_nodes * n_features)
    #X = (spline_basis @ w).reshape(-1, n_nodes, n_features) 
    #X = expit(X) / np.sqrt(n_features)

    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps + k_steps):
        means.append((X[t] @ X[t].T)[subdiag])

    scale = density / np.mean(means)

    Ys = []
    probas = []
    for t in range(n_time_steps + k_steps):
        Y = np.zeros((n_nodes, n_nodes))
        
        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * means[t], 0, 1))
        probas.append(np.clip(X[t] @ X[t].T, 0, 1))

        Y[subdiag] = y
        Y += Y.T

        Y = sp.csr_array(Y, dtype=float) 
        Ys.append(Y)

    return Ys, X, np.stack(probas)


def simulate_network_bspline(n_nodes=100, n_time_steps=100, n_features=2, 
                         df=5, k_buffer=10, k_steps=5, density=0.2, 
                         return_ndarray=True, random_state=42):

    n_knots = df - 2
    
    rng = check_random_state(random_state)
    ts = np.arange(n_time_steps - k_buffer)
    bspline = SplineTransformer(extrapolation='linear', n_knots=n_knots).fit(ts[:, None])
    
    ts = np.arange(n_time_steps + k_steps)
    spline_basis = bspline.transform(ts[:, None])
    w = rng.dirichlet(np.ones(df)/df, n_nodes * n_features).T
    X = (spline_basis @ w ).reshape(-1, n_nodes, n_features) 
    X /= np.sqrt(n_features)
    
    means = []
    subdiag = np.tril_indices(n_nodes, k=-1)
    for t in range(n_time_steps + k_steps):
        means.append((X[t] @ X[t].T)[subdiag])

    scale = density / np.mean(means)

    Ys = []
    probas = []
    for t in range(n_time_steps + k_steps):
        Y = np.zeros((n_nodes, n_nodes))
        
        X[t] *= np.sqrt(scale)
        y = rng.binomial(1, np.clip(scale * means[t], 0, 1))
        probas.append(np.clip(X[t] @ X[t].T, 0, 1))

        Y[subdiag] = y
        Y += Y.T

        Y = sp.csr_array(Y, dtype=float) 
        Ys.append(Y)
    
    if return_ndarray:
        Ys = np.stack([Y.toarray() for Y in Ys])

    return Ys, X, np.stack(probas)
