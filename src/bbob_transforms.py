"""The BBOB transformations that the lattice and optimum-centred helpers need to write f from its
formula: the rotations of an instance, T_osz, T_asy, the box penalty, and the 1D wells of f20.
Everything is rebuilt from the BBOB generator (bbob_gallagher_port.py).
"""
import functools

import numpy as np
from scipy.optimize import minimize_scalar

from bbob_gallagher_port import _compute_rotation


def _rotation(seed, d):
    return _compute_rotation(seed, d)


def _tosz(X):
    # T_osz: a smooth, coordinate-wise wobble that keeps 0 and the sign
    Y = np.zeros_like(X)
    nz = X != 0
    xh = np.log(np.abs(X[nz]))
    pos = X[nz] > 0
    c1, c2 = np.where(pos, 10.0, 5.5), np.where(pos, 7.9, 3.1)
    Y[nz] = np.sign(X[nz]) * np.exp(xh + 0.049 * (np.sin(c1 * xh) + np.sin(c2 * xh)))
    return Y


def _tasy(X, beta):
    # T_asy: positive coordinates are raised to a power that grows with the coordinate's index
    d = X.shape[1]
    P = np.maximum(X, 0.0)
    return np.where(X > 0, P ** (1 + beta * np.arange(d) / (d - 1) * np.sqrt(P)), X)


def _penalty(X):
    return (np.maximum(0.0, np.abs(X) - 5.0) ** 2).sum(axis=1)


@functools.lru_cache(maxsize=None)
def schwefel_wells(d):
    """The 1D function of f20, phi(t) = 0.01 (max(0, |t| - 500)^2 - t sin(sqrt|t|) / d), and its wells:
    the crests c_0 < ... < c_{K-1} (local maxima), the minima, and t*, the deepest minimum."""
    def phi(t):
        t = np.asarray(t, dtype=float)
        return 0.01 * (np.maximum(0.0, np.abs(t) - 500.0) ** 2 - t * np.sin(np.sqrt(np.abs(t))) / d)

    t = np.linspace(-700.0, 700.0, 1_400_001)      # beyond |t| = 520 the penalty only rises
    y = phi(t)
    inner = np.arange(1, len(t) - 1)
    is_max = inner[(y[inner] > y[inner - 1]) & (y[inner] >= y[inner + 1])]
    is_min = inner[(y[inner] < y[inner - 1]) & (y[inner] <= y[inner + 1])]

    def polish(k, sign):
        res = minimize_scalar(lambda u: sign * float(phi(u)), bounds=(t[k - 1], t[k + 1]), method="bounded",
                              options=dict(xatol=1e-10))
        return res.x

    crests = np.array([polish(k, -1) for k in is_max])
    minima = np.array([polish(k, 1) for k in is_min])
    t_star = minima[np.argmin(phi(minima))]
    return phi, crests, minima, t_star
