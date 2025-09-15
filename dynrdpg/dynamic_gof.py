import jax.numpy as jnp
import jax
import pandas as pd

from numpyro.distributions.util import vec_to_tril_matrix

import dynrdpg.static_gof as sgof


def tril_vec_to_matrix(x):
    A = vec_to_tril_matrix(x.astype(float), diagonal=-1)
    return A + A.T


def degree(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        return None, sgof.degree(y_vec[t])
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))

    return res.astype(int)


def avg_degree(y_vec):
    return degree(y_vec).mean(axis=0)


def node_degree(y_vec, node_id=0):
    n_time_steps = y_vec.shape[0]
    
    def stat_fun(carry, t):
        Y = sgof.vec_to_adjacency(y_vec[t])
        return None, Y[node_id].sum()
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))
    
    return res.astype(int)


def density(y_vec):
    return y_vec.mean(axis=1)


def transitivity(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        return None, sgof.transitivity(y_vec[t])
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))

    return res


def dyadic_stability(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = (y_vec[t] * y_vec[t+1]).mean()
        out += ((1 - y_vec[t]) * (1 - y_vec[t+1])).mean()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def edge_persistence(y_vec):
    # y_vec is n_time_steps x n_dyads
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = (y_vec[t] * y_vec[t+1]).sum() / y_vec[t].sum()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def nonedge_persistence(y_vec):
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        out = ((1 - y_vec[t]) * (1 - y_vec[t+1])).sum() / (1 - y_vec[t]).sum()
        return None, out
    _, res = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps-1))

    return res


def lag1_degree_correlation(y_vec):
    n_time_steps = y_vec.shape[0]

    def stat_fun(carry, t):
        return None, sgof.degree(y_vec[t])
    _, deg = jax.lax.scan(stat_fun, None, jnp.arange(n_time_steps))

    v1 = deg[:-1].ravel()
    v2 = deg[1:].ravel()

    return jnp.corrcoef(v1, v2)[0, 1]


def stat_distribution(stats):
    stat = pd.melt(pd.DataFrame(stats), var_name='t')
    stat['t'] = stat['t'] + 1
    return stat
