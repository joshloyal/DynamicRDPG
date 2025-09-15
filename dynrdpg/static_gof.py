import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd

from functools import partial


def shape_from_tril_vec(x, k=0):
    return round((np.sqrt(1 + 8 * x.shape[-1]) - 1) / 2) - k


def vec_to_adjacency(y_vec, include_nan=False):
    n = shape_from_tril_vec(y_vec, k=-1)
    indices = jnp.tril_indices(n, k=-1)
    Y = jnp.zeros((n, n))
    Y = Y.at[indices].set(y_vec)
    Y += Y.T
    if include_nan:
        Y.at[jnp.diag_indices(n)].set(jnp.nan)
    return Y


def density(y_vec):
    return y_vec.mean()


def std_degree(y_vec, is_vec=False):
    Y = vec_to_adjacency(y_vec, include_nan=True)
    # ignore diagonal entries
    return jnp.nanstd(jnp.nansum(Y, axis=1), ddof=1)


def degree(y_vec):
    return vec_to_adjacency(y_vec).sum(axis=0).astype(int)


def degree_distribution(degrees):
    max_bin = np.max(degrees) + 1
    counts = np.apply_along_axis(
        partial(np.bincount, minlength=max_bin), 1, degrees)
    return pd.melt(pd.DataFrame(counts), var_name='degree', value_name='count')


def transitivity(y_vec):
    Y = vec_to_adjacency(y_vec, include_nan=False)
    n_triangles = jnp.trace(jnp.linalg.matrix_power(Y, 3))
    Y_sq = Y @ Y
    n_triplets = jnp.sum(Y_sq) - jnp.trace(Y_sq)
    return n_triangles / n_triplets
