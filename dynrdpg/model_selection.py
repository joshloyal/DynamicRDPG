import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from .dynrdpg import DynamicRDPG


def waic_selection_single(Y, rw_order=2, is_binary=True, n_features=2, n_burnin=2500, n_samples=2500):
    model = DynamicRDPG(n_features=n_features, rw_order=rw_order, random_state=42)
    model.sample(Y, n_burnin=n_burnin, n_samples=n_samples)
    return model, n_features, model.waic()


def waic_selection(Y, rw_order=2, is_binary=True, min_features=1, max_features=10,
                   n_burnin=500, n_samples=500, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs)(delayed(waic_selection_single)(
        Y=Y, rw_order=rw_order, is_binary=is_binary, n_features=d,
        n_burnin=n_burnin, n_samples=n_samples) for
            d in range(min_features, max_features + 1))
    
    models = [r[0] for r in res] 
    criteria = [r[1:] for r in res]
    return models, pd.DataFrame(np.asarray(criteria), columns=['n_features', 'waic'])
