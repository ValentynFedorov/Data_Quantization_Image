# src/metrics.py
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import ot

def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= (p.sum() + 1e-12)
    q /= (q.sum() + 1e-12)
    return jensenshannon(p, q)**2

def ks_test(a, b):
    stat, p = ks_2samp(a, b)
    return stat, p

def wasserstein_1d(u, v):
    u = np.asarray(u).reshape(-1,1)
    v = np.asarray(v).reshape(-1,1)
    a = np.ones((u.shape[0],))/u.shape[0]
    b = np.ones((v.shape[0],))/v.shape[0]
    M = ot.dist(u, v, metric='euclidean')
    return ot.emd2(a, b, M)
