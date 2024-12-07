import pandas as pd
import numpy as np
import networkx as nx
import joblib

from os.path import dirname, join


__all__ = ['load_news']


def load_news(sparse=True):
    module_path = dirname(__file__)
    
    if sparse:
        Y = joblib.load(open(join(module_path, 'data', 'audience_italy_sparse.npy'), 'rb'))
    else:
        Y = joblib.load(open(join(module_path, 'data', 'audience_italy.npy'), 'rb'))

    node_names = pd.read_csv(join(module_path, 'data', 'news_names_italy.csv'), header=None)[0].values

    return Y, node_names
