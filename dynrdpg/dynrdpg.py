import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
import statsmodels.api as sm

from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from scipy.linalg import block_diag, orthogonal_procrustes
from scipy.special import expit, logsumexp
from scipy import stats 

import numpy as np


def ase(A, k=2):
    eigvals, eigvec = sp.linalg.eigsh(A, k=k)
    return eigvec[:, ::-1] * np.sqrt(np.abs(eigvals)[::-1])

def smooth_positions_procrustes(U):
    n_time_steps, _, _ = U.shape
    for t in range(1, n_time_steps):
        R, _ = orthogonal_procrustes(U[t], U[t-1])
        U[t] = U[t] @ R

    return U


def dynamic_adjacency_to_vec(Y, sparse=False):
    if not sparse:
        n_time_points, n_nodes, _ = Y.shape
    else:
        n_time_points = len(Y)
        n_nodes = Y[0].shape[0]

    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = np.zeros((n_time_points, n_dyads), dtype=int)
    for t in range(n_time_points):
        y[t] = Y[t].todense()[subdiag] if sparse else Y[t][subdiag]

    return y


def calculate_auc(Y, probas):
    n_time_steps, _ = Y.shape
    y_true, y_pred = [], []
    for t in range(n_time_steps):
        y_true.append(Y[t])
    
    return roc_auc_score(np.concatenate(y_true), probas.ravel())


class DynamicRDPG(object):
    def __init__(self,
                 n_features=2,
                 init_algorithm='ase',
                 scale='auto',
                 rw_order=2,
                 is_binary=True,
                 sample_scale=True,
                 random_state=42):
        self.n_features = n_features
        self.init_algorithm = init_algorithm
        self.scale = scale
        self.rw_order = rw_order
        self.is_binary = is_binary
        self.sample_scale = sample_scale
        self.random_state = random_state

    def sample(self, Y, n_burnin=500, n_samples=2000):
        
        if isinstance(Y, np.ndarray): 
            n_time_points, n_nodes, _ = Y.shape
            Y = [sp.csr_array(Y[t]) for t in range(Y.shape[0])]
        else:
            n_time_points = len(Y)
            n_nodes = Y[0].shape[0]

        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        self.y_vec_ = dynamic_adjacency_to_vec(Y, sparse=True)
        rng = check_random_state(self.random_state)
        subdiag = np.tril_indices(n_nodes, k=-1)
        
        # initialize samples
        self.samples_ = {
            'X': np.zeros(
                (n_samples, n_time_points, n_nodes, self.n_features)),
            'sigma': np.zeros((n_samples, n_nodes))
        }
        if self.sample_scale:
            self.samples_['scale'] = np.zeros(n_samples)
        
        if self.scale == 'auto':
            self.scale = 1. / np.var(self.y_vec_.ravel())
        
        # initialize parameters
        if self.init_algorithm == 'random':
            X = rng.randn(n_nodes, self.n_features, n_time_points)
            sigma = np.ones(n_nodes)
            nu = np.ones(n_nodes)
            scale = self.scale
        else:
            X = np.zeros((n_time_points, n_nodes, self.n_features))
            for t in range(n_time_points):
                X[t] = ase(Y[t], k=self.n_features)
            X = smooth_positions_procrustes(X).transpose((1, 2, 0))

            #if self.scale == 'auto':
            #    XXt = np.einsum('tid,tjd->tij', X, X)[..., subdiag[0], subdiag[1]]
            #    self.scale = 1. / np.mean((self.y_vec_ - XXt) ** 2)
            
            sigma = np.mean(np.diff(X, axis=2) ** 2, axis=(1, 2))
            nu = np.ones(n_nodes)
            scale = self.scale
        
        # initialize running statistics for sampler
        XtX = []
        for t in range(n_time_points):
            XtX.append(np.zeros((self.n_features, self.n_features)))
            for i in range(n_nodes):
                XtX[t] += X[i, :, t][:, None] @ X[i, :, t][:, None].T 
            
        # the K matrix under a diffuse prior in Chan and Jeliazkov (2009).
        # e.g. vec(X_i) ~ N_{Td}(0, (1/sigma_i^2) [D_r^T D \times I_d]) 
        D = np.diff(np.eye(n_time_points), self.rw_order, axis=0)
        K = sp.dia_array(np.kron(D.T @ D, np.eye(self.n_features))) 

        for idx in tqdm(range(n_burnin + n_samples)):

            # sample latent positions
            for i in range(n_nodes): 
                # calculate needed statistics
                XtY = []
                for t in range(n_time_points):
                    XtX[t] -= X[i, :, t][:, None] @ X[i, :, t][:, None].T
                    
                    # get indices of node i's neighbors at time t
                    indices = Y[t].indices[Y[t].indptr[i]:Y[t].indptr[i+1]]

                    if self.is_binary:
                        XtY.append(np.sum(X[indices, :, t], axis=0))
                    else:
                        yt = Y[t].data[Y[t].indptr[i]:Y[t].indptr[i+1]]
                        XtY.append(np.sum(X[indices, :, t] * yt[:, None], axis=0))
                
                # calculate P and its (upper) cholesky decomposition
                precision = (1. / sigma[i]) * K
                P = sp.dia_array(precision + scale * sp.block_diag(XtX, format='dia'))
                
                # put P matrix into upper-diagonal form to perform banded cholesky
                ab = np.zeros((self.rw_order * self.n_features + 1, P.shape[1]))
                diag_id = np.where(P.offsets == 0)[0][0]
                for k, offset_id in enumerate(P.offsets[diag_id:]):
                    ab[offset_id] = P.data[diag_id + k]
                L = linalg.cholesky_banded(ab[::-1])
                #L = linalg.cholesky_banded(P.T.data[:(self.n_features+1)])

                # solve for mean
                X_hat = linalg.cho_solve_banded((L, False), scale * np.ravel(XtY))
                
                # sample z ~ N(0, [L.T @ L]^{-1})
                # convert L to a sparse array
                #offsets = np.arange(self.n_features+1)[::-1]
                offsets = np.arange(L.shape[0])[::-1]
                k = n_time_points * self.n_features
                L = sp.dia_array((L, offsets) , shape=(k, k))
                z = sp.linalg.spsolve_triangular(sp.csr_array(L), 
                        rng.randn(n_time_points * self.n_features), 
                        lower=False)

                # sample (X_i1, ...., X_iT)
                X[i] = (X_hat + z).reshape(n_time_points, self.n_features).T
                
                # reset running statistics
                for t in range(n_time_points):
                    XtX[t] += X[i, :, t][:, None] @ X[i, :, t][:, None].T
                    
                 
            # sample transition variances from a half-cauchy prior
            shape_sigma = 0.5 * ((n_time_points - self.rw_order) * self.n_features + 1)
            scale_sigma = 0.5 * np.sum(np.diff(X, self.rw_order, axis=2) ** 2, axis=(1, 2)) + (1. / nu)
            sigma = stats.invgamma.rvs(shape_sigma, scale=scale_sigma, random_state=rng)
            nu = stats.invgamma.rvs(1, 1 + (1. / sigma), random_state=rng)

            # sample scale
            # XXX: Computationally expensive
            if self.sample_scale:
                x = X.transpose((2, 0, 1))  # (n_time_steps, n_nodes, n_features)
                XXt = np.einsum('tid,tjd->tij', x, x)[..., subdiag[0], subdiag[1]]
                a = 0.25 * n_nodes * (n_nodes - 1) * n_time_points + 1e-3
                b = 0.5 * np.sum((self.y_vec_ - XXt) ** 2) + 1e-3
                scale = stats.gamma.rvs(a, scale = 1. / b)
            
            if idx >= n_burnin:
                self.samples_['X'][idx-n_burnin] = X.transpose((2, 0, 1))
                self.samples_['sigma'][idx-n_burnin] = sigma
                if self.sample_scale:
                    self.samples_['scale'][idx-n_burnin] = scale

        # post processing
        # smooth reference latent position and procrustes match
        self.samples_['X'][-1] = smooth_positions_procrustes(self.samples_['X'][-1])
        for idx in range(n_samples):
            for t in range(n_time_points):
                R, _ = orthogonal_procrustes(
                    self.samples_['X'][idx][t], self.samples_['X'][-1][t])
                self.samples_['X'][idx][t] = self.samples_['X'][idx][t] @ R
        
        self.X_ = self.samples_['X'].mean(axis=0)
        self.sigma_ = self.samples_['sigma'].mean(axis=0)
        
        if self.is_binary:
            self.probas_ = np.zeros((n_time_points, n_dyads))
            for t in range(n_time_points):
                self.probas_[t] = (self.X_[t] @ self.X_[t].T)[subdiag]

            self.auc_ = calculate_auc(self.y_vec_, np.clip(self.probas_, 0, 1))
        else:
            self.means_ = np.zeros((n_time_points, n_dyads))
            for t in range(n_time_points):
                self.means_[t] = (self.X_[t] @ self.X_[t].T)[subdiag]

            self.rmse_ = np.sqrt(np.mean((self.y_vec_ - self.means_) ** 2))

        return self

    def dic(self):
        X = self.samples_['X']
        subdiag = np.tril_indices(X.shape[2], k=-1)
        XXt = np.einsum('stid,stjd->stij', X, X)[..., subdiag[0], subdiag[1]]

        loglik_mean = np.mean(-0.5 * np.sum((self.y_vec_ - XXt) ** 2, axis=(1,2)))
        loglik_hat = -0.5 * np.sum((self.y_vec_ - self.probas_) ** 2)

        p_dic = 2 * (loglik_hat - loglik_mean)

        return -2 * loglik_hat + 2 * p_dic
    
    def waic(self):
        X = self.samples_['X']
        subdiag = np.tril_indices(X.shape[2], k=-1)
        XXt = np.einsum('stid,stjd->stij', X, X)[..., subdiag[0], subdiag[1]]
        loglik = -0.5 * (self.y_vec_ - XXt) ** 2
        
        #n_samples, n_time_steps, n_nodes, _ = self.samples_['X'].shape
        #n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        #loglik = np.zeros((n_samples, n_time_steps, n_dyads))
        #subdiag = np.tril_indices(n_nodes, k=-1)
        #for i in range(n_samples):
        #    X = self.samples_['X'][i]
        #    for t in range(n_time_steps):
        #        loglik[i, t] = -0.5 * (self.y_vec_[t] - (X[t] @ X[t].T)[subdiag]) ** 2

        lppd = (logsumexp(loglik, axis=0) - np.log(X.shape[0])).sum()
        p_waic = loglik.var(axis=0).sum()

        return -2 * (lppd - p_waic)
    
    def forecast_positions(self, k_steps=1, n_samples=100):
        rng = check_random_state(self.random_state)
        _, n_time_steps, n_nodes, n_features = self.samples_['X'].shape
        n_samples = min(n_samples, self.samples_['X'].shape[0])
        samples = np.zeros((n_samples, k_steps, n_nodes, n_features))
        for i in range(n_samples):
            sigma = np.sqrt(self.samples_['sigma'][i])
            if self.rw_order == 1:
                samples[i] = self.samples_['X'][i, -1] + sigma[:, None] * np.cumsum(
                        rng.randn(k_steps, n_nodes, n_features), axis=0)
            else:
                samples[i, 0] = (
                        2 * self.samples_['X'][i, -1] - 
                        self.samples_['X'][i, -2] + 
                        sigma[:, None] * rng.randn(n_nodes, n_features) )

                if k_steps > 1:
                    samples[i, 1] = (
                            2 * samples[i, 0] - 
                            self.samples_['X'][i, -1] + 
                            sigma[:, None] * rng.randn(n_nodes, n_features) )
                    for h in range(2, k_steps):
                        samples[i, h] = (
                            2 * samples[i, h-1] - samples[i, h-2] + 
                            sigma[:, None] * rng.randn(n_nodes, n_features))

        return samples

    def forecast(self, k_steps=1, n_samples=100):
        samples = self.forecast_positions(k_steps=k_steps, n_samples=n_samples)
        
        n_samples, k_steps, n_nodes, _ = samples.shape
        probas = np.zeros((n_samples, k_steps, n_nodes, n_nodes))
        for i in range(samples.shape[0]):
            for t in range(k_steps):
                probas[i, t] = samples[i, t] @ samples[i, t].T
        
        return np.clip(probas, 0, 1) if self.is_binary else probas
    
    def predict(self, n_samples=100):
        rng = check_random_state(self.random_state)
        n_samples = min(n_samples, self.samples_['X'].shape[0])
        _, n_time_steps, n_nodes, _ = self.samples_['X'].shape
        samples = np.zeros((n_samples, n_time_steps, n_nodes, n_nodes))
        for i in range(n_samples):
            for t in range(n_time_steps):
                Xt = self.samples_['X'][i, t]
                samples[i, t] = Xt @ Xt.T

        return np.clip(samples, 0, 1) if self.is_binary else samples
