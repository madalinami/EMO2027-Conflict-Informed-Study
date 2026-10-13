"""The Gallagher peaks of BBOB f21 and f22, rebuilt from the BBOB generator.

A line-by-line port of the IOHexperimenter generator (gallagher101.hpp, bbob_problem.hpp and
common/random.hpp, which follow COCO's f_gallagher.c). ioh does not expose the peak centres,
weights and scales, and the conflict-informed helpers on f21/f22 are written in terms of single
peaks. The BBOB rotation matrices of the other functions come from the same generator.
"""
import functools
import math

import ioh
import numpy as np


def x_star(f_id, instance, dim):
    """The global minimiser x* of f, as ioh reports it."""
    return np.array(ioh.get_problem(int(f_id), int(instance), int(dim)).optimum.x, dtype=float)


_MULT, _MOD, _MOD_DIV, _MOD_MULT = 16807, 2147483647, 127773, 2836


def _lcg_rand(seed):
    # One step of the Park-Miller generator (x16807 mod 2^31-1), with Schrage's trick like the
    # C++ code, so it never overflows.
    seed_mod = math.floor(seed / _MOD_DIV)
    seed = _MULT * (seed - seed_mod * _MOD_DIV) - _MOD_MULT * seed_mod
    if seed < 0:
        seed += _MOD
    return seed


def _bbob_uniform(n, initial_seed):
    # BBOB's shuffled generator: 40 warm-up steps, a table of 32 values, and each output picks
    # a table slot from the previous value.
    gens = [0] * 32
    seed = max(initial_seed, 1)
    for i in range(39, -1, -1):
        seed = _lcg_rand(seed)
        if i < 32:
            gens[i] = seed
    x = np.empty(n)
    rn = float(gens[0])
    for i in range(n):
        idx = int(math.floor(rn / 67108865.0))
        seed = _lcg_rand(seed)
        rn = float(gens[idx])
        gens[idx] = seed
        v = rn / 2.147483647e9
        x[i] = 1e-99 if v == 0.0 else v
    return x


def _bbob_normal(n, seed):
    u = _bbob_uniform(2 * n, seed)
    x = np.sqrt(-2 * np.log(u[:n])) * np.cos(2 * math.pi * u[n:])
    x[x == 0.0] = 1e-99
    return x


def _compute_rotation(seed, d):
    r = _bbob_normal(d * d, seed)
    m = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            m[i, j] = r[j * d + i]
    for i in range(d):
        for j in range(i):
            prod = np.dot(m[:, i], m[:, j])
            m[:, i] -= prod * m[:, j]
        m[:, i] /= math.sqrt(np.dot(m[:, i], m[:, i]))
    return m


def _sorted_perm_indices(n, seed):
    u = _bbob_uniform(n, seed)
    return [i for _, i in sorted(zip(u, range(n)), key=lambda t: t[0])]


@functools.lru_cache(maxsize=None)
def _gallagher_params_cached(f_id, instance, dim):
    """The random parameters of f21/f22 we need (peak centres y, weights w, scales, rotation R),
    drawn in the generator's order. Row 0 of y is the global peak."""
    d = dim
    n_peaks, b, c, cond0 = (101, 10.0, 5.0, math.sqrt(1000.0)) if f_id == 21 else (21, 9.8, 4.9, 1000.0)
    seed = f_id + 10000 * instance
    R = _compute_rotation(seed, d)
    divisor = n_peaks - 2.0
    perm = _sorted_perm_indices(n_peaks - 1, seed)
    # The global peak has weight 10; the local peaks get weights spread evenly between 1.1 and
    # 9.1 and a random condition number between 1 and 1000.
    w = [10.0]
    conds = [cond0]
    for i in range(1, n_peaks):
        w.append((i - 1.0) / divisor * (9.1 - 1.1) + 1.1)
        conds.append(1000.0 ** (perm[i - 1] / divisor))
    scales = np.empty((n_peaks, d))
    for i in range(n_peaks):
        p = _sorted_perm_indices(d, seed + 1000 * i)
        for j in range(d):
            scales[i, j] = conds[i] ** (p[j] / (d - 1.0) - 0.5)
    u = _bbob_uniform(d * n_peaks, seed)
    y = np.empty((n_peaks, d))
    for j in range(n_peaks):
        y[j] = b * u[j * d:(j + 1) * d] - c
    y[0] *= 0.8  # the global peak is kept a bit away from the border, as in BBOB
    # A cheap proof that the port follows ioh: our global peak is ioh's x*.
    assert np.abs(y[0] - x_star(f_id, instance, d)).max() < 1e-12, f"Gallagher port is off on f{f_id}"
    return dict(R=R, y=y, w=np.array(w), scales=scales)


def _gallagher_q(P, X):
    """Quadratic form q_i(x) = (x - mu_i)^T R^T C_i R (x - mu_i) for every peak i, shape (n, N).
    """
    n, d = X.shape
    N = len(P["w"])
    RY = P["y"] @ P["R"].T
    out = np.empty((n, N))
    chunk = max(1, int(4e6 // (N * d)))
    for s in range(0, n, chunk):
        RX = X[s:s + chunk] @ P["R"].T
        diff = RX[:, None, :] - RY[None, :, :]
        out[s:s + chunk] = np.einsum("npd,pd->np", diff ** 2, P["scales"])
    return out


def _gallagher_parts(f_id, instance, X):
    """Weighted peaks W[:, i] = w_i g_i(x) at the rows of X, and the parameters."""
    P = _gallagher_params_cached(int(f_id), int(instance), X.shape[1])
    logg = -0.5 / X.shape[1] * _gallagher_q(P, X)
    W = P["w"][None, :] * np.exp(logg)
    return W, P


def _visible_mask(P):
    # A peak is visible when it wins at its own centre: w_i > w_j g_j(mu_i) for every other j.
    # Only then is mu_i a local minimum of f; a hidden peak is swallowed by a stronger one.
    C = P["w"][None, :] * np.exp(-0.5 / P["y"].shape[1] * _gallagher_q(P, P["y"]))
    N = len(P["w"])
    own = np.diag(C).copy()
    others = np.where(np.eye(N, dtype=bool), -np.inf, C).max(axis=1)
    return own > others
