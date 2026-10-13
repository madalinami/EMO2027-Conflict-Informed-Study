"""The optimum-centred helpers: Gallagher functions whose global component sits on x* of a BBOB function.

    h = w0 - max(w0 g0(x), max_{i>=1} w_i g_i(x)),   w0 = 10

    g0   the pit of h at x*: g0 = exp(-1/2 (x - x*)^T S^-1 (x - x*))
    g_i  only in the _rnd helpers: 100 random spheres, centres uniform in [-5, 5]^D, weights equally
         spaced over [1.1, 9.1] (as in BBOB f21), g_i = exp(-|x - y_i|^2 / (2D)) (BBOB's 1/(2D) with C = I).
         They are pits of h that have nothing to do with f.

Since w0 > w_i, h(x*) = 0 and h > 0 elsewhere: x* is the only minimiser of every helper here. Like
h_orc, they all know x*.

  name in the code     name in the paper    the pit
  g_sph                h_sph                S = D I (the Gallagher convention)
  g_sph_k<k>           h_sph,k              S = (k r_B)^2 I
  g_ell_k<k>           h_ell,k              S = the shape of the basin of x*, scaled to a geometric-mean
                                            radius of k r_B
  g_ell_k2_rot         h_rot,2              as g_ell_k2, with the ellipse turned by a random rotation
  g_sph_rnd            h^rnd_sph            g_sph + the random spheres
  g_sph_k2_rnd         h^rnd_sph,2          g_sph_k2 + the random spheres
  g_ell_k2_rnd         h^rnd_ell,2          g_ell_k2 + the random spheres
  g_ell_k2_rot_rnd     h^rnd_rot,2          g_ell_k2_rot + the random spheres

The width of the pit is set by k = (width of the pit) / (size of the basin of x* in f):
  - the basin B of x*: q <= 1 (the cell of x* in z for f3, f4, f15, f24, the first crest for f17/f18,
    the valley s < pi for f19, the well of x* for f20, the step of x* for f7), and for f21/f22 the
    region where the global peak is the highest;
  - r(u): the distance from x* to the edge of B along the direction u (2000 directions; the first exit,
    found on a log-spaced grid and refined by bisection; clipped at the box);
  - r_B = (mean_u r(u)^D)^(1/D), the radius of the ball with the volume of the part of B seen from x* along
    straight rays (the first exit; where B is not star-shaped about x*, e.g. the trench of f19 in 2D, this is
    less than B: 0.42 against 0.54 there);
  - the shape: the second-moment matrix of that part, D/(D+2) sum_u r(u)^(D+2) u u^T / sum_u r(u)^D. In 10D,
    2000 rays fix its aspect ratio well but its axes only loosely.
k < 1: the pit of h lies inside the basin of x*; k = 1: as large as it; k > 1: it spreads over the traps
around it. Far from x* (many sigma) a narrow pit is flat, so h tells nothing there.

The geometry is computed once per (function, instance, dimension) and stored in data/static/pointer.

    python src/bbob_pointer.py --build [--fids ...] [--dims ...] [--instances ...]   builds and checks it
"""
import argparse
import functools
import math
from pathlib import Path

import ioh
import numpy as np
from scipy.optimize import minimize_scalar

import bbob_gallagher_port as GP
import bbob_transforms as BA

LO, HI, W0 = -5.0, 5.0, 10.0
KAPPAS = (0.5, 2.0, 8.0)
POINTER_FUNCTIONS = (3, 4, 7, 15, 17, 18, 19, 20, 21, 22, 24)
POINTER_HELPERS = ("g_sph", *(f"g_sph_k{k:g}" for k in KAPPAS), *(f"g_ell_k{k:g}" for k in KAPPAS), "g_ell_k2_rot",
                   "g_sph_rnd", "g_sph_k2_rnd", "g_ell_k2_rnd", "g_ell_k2_rot_rnd")
N_RANDOM, N_DIRECTIONS = 100, 2000
STATIC = Path(__file__).resolve().parents[1] / "data" / "static" / "pointer"


def has_pointer(f_id, name=None):
    return int(f_id) in POINTER_FUNCTIONS and (name is None or name in POINTER_HELPERS)


# --- The basin of x*: q(X) <= 1 ------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def schaffer_first_crest():
    """The first local maximum of g(s) = sqrt(s)(1 + sin^2(50 s^0.2)) for s > 0 (1.2367e-5)."""
    def g(s):
        return np.sqrt(s) * (1 + np.sin(50 * s ** 0.2) ** 2)
    s = np.logspace(-12, 0, 2_000_001)
    y = g(s)
    k = int(np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0]) + 1
    res = minimize_scalar(lambda u: -float(g(u)), bounds=(s[k - 1], s[k + 1]), method="bounded",
                          options=dict(xatol=1e-14))
    return float(res.x)


def basin_coordinate(f_id, instance, d, xs):
    """q(X): 0 at x*, 1 on the edge of the basin of x*, above 1 outside. For f21/f22: 0 where the global
    peak is the highest, 2 elsewhere (an indicator, no inner coordinate)."""
    i = np.arange(d)
    if f_id in (3, 4, 15):
        seed = 3 + 10000 * instance if f_id == 4 else f_id + 10000 * instance
        if f_id == 15:
            A, B = BA._rotation(seed + 1000000, d), BA._rotation(seed, d)
            Mz = A @ np.diag(math.sqrt(10.0) ** (i / (d - 1))) @ B

        def z(X):
            if f_id == 3:
                return 10 ** (0.5 * i / (d - 1)) * BA._tasy(BA._tosz(X - xs), 0.2)
            if f_id == 4:
                Y = BA._tosz(X - xs)
                return np.where((Y > 0) & (i % 2 == 0), 10.0, 1.0) * 10 ** (0.5 * i / (d - 1)) * Y
            return BA._tasy(BA._tosz((X - xs) @ A.T), 0.2) @ Mz.T
        return lambda X: (4.0 * z(X) ** 2).max(axis=1)
    if f_id == 24:
        seed = 24 + 10000 * instance
        A, B = BA._rotation(seed + 1000000, d), BA._rotation(seed, d)
        c = 10.0 ** (i / (d - 1))
        return lambda X: (4.0 * ((((2.0 * X * np.sign(xs) - 2.5) @ B.T) * c) @ A.T) ** 2).max(axis=1)
    if f_id in (17, 18):
        seed = 17 + 10000 * instance
        A, B = BA._rotation(seed + 1000000, d), BA._rotation(seed, d)
        lam = math.sqrt(10.0 if f_id == 17 else 1000.0) ** (i / (d - 1))
        s_c = schaffer_first_crest()

        def q(X):
            Z = (BA._tasy((X - xs) @ A.T, 0.5) @ B.T) * lam
            return np.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2).max(axis=1) / s_c
        return q
    if f_id == 19:
        R = max(1.0, math.sqrt(d) / 8.0) * BA._rotation(19 + 10000 * instance, d)

        def q(X):
            Z = X @ R.T + 0.5
            return (100.0 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (1.0 - Z[:, :-1]) ** 2).max(axis=1) / math.pi
        return q
    if f_id == 20:
        phi, crests, _, t_star = BA.schwefel_wells(d)
        k = int(np.searchsorted(crests, t_star))
        c_lo = crests[k - 1]
        if k < len(crests):
            c_hi = crests[k]
        else:   # the outer well climbs into the penalty wall: it ends where phi is back at the crest height
            t = np.linspace(t_star, 1000.0, 1_000_001)
            c_hi = float(t[np.argmax(phi(t) >= phi(c_lo))])
        a, sign = 2.0 * np.abs(xs), np.sign(xs)
        lam = math.sqrt(10.0) ** (i / (d - 1))

        def q(X):
            Xh = 2.0 * sign * X
            Zh = Xh.copy()
            Zh[:, 1:] += 0.25 * (Xh[:, :-1] - a[:-1])
            Z = 100.0 * (lam * (Zh - a) + a)
            r = np.where(Z >= t_star, (Z - t_star) / (c_hi - t_star), (t_star - Z) / (t_star - c_lo))
            return r.max(axis=1) ** 2
        return q
    if f_id == 7:
        B = BA._rotation(7 + 10000 * instance, d)
        lam = np.sqrt(10.0 ** (i / (d - 1)))
        return lambda X: (np.abs(((X - xs) @ B.T) * lam).max(axis=1) / 0.05) ** 2
    if f_id in (21, 22):
        def q(X):
            W = GP._gallagher_parts(f_id, instance, np.atleast_2d(X))[0]
            return np.where(W.argmax(axis=1) == 0, 0.0, 2.0)
        return q
    raise ValueError(f"no basin model for f{f_id}")


def _first_exit(q, xs, U, n_grid=120, n_bisect=40):
    """r(u) for every row u of U: the first distance at which q > 1 along x* + t u, clipped at the box."""
    with np.errstate(divide="ignore"):
        t_box = np.min(np.where(U > 0, (HI - xs) / U, np.where(U < 0, (LO - xs) / U, np.inf)), axis=1)
    T = np.logspace(-10, 0, n_grid)[None, :] * t_box[:, None]                  # (n_dir, n_grid)
    n, d = U.shape
    out = q((xs[None, None, :] + T[:, :, None] * U[:, None, :]).reshape(-1, d)).reshape(n, n_grid) > 1.0
    first = np.where(out.any(axis=1), out.argmax(axis=1), n_grid)
    r = t_box.copy()
    inside = first < n_grid
    lo = np.where(first > 0, T[np.arange(n), np.maximum(first - 1, 0)], 0.0)[inside]
    hi = T[np.arange(n), np.minimum(first, n_grid - 1)][inside]
    Ui = U[inside]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        o = q(xs[None, :] + mid[:, None] * Ui) > 1.0
        hi, lo = np.where(o, mid, hi), np.where(o, lo, mid)
    r[inside] = 0.5 * (lo + hi)
    return r


def basin_geometry(f_id, instance, d, seed=0):
    """r_B (volume-equivalent radius of the basin of x*), its second-moment matrix, and r(u)."""
    xs = np.array(ioh.get_problem(f_id, instance, d).optimum.x, dtype=float)
    q = basin_coordinate(f_id, instance, d, xs)
    U = np.random.default_rng(seed).standard_normal((N_DIRECTIONS, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    r = _first_exit(q, xs, U)
    lr = d * np.log(r)                                    # r^D can underflow (f17/f18: r ~ 1e-6)
    wD = np.exp(lr - lr.max())
    r_B = float(np.exp((np.log(wD.mean()) + lr.max()) / d))
    w2 = wD * (r / r.max()) ** 2
    second = d / (d + 2) * (U.T * w2) @ U / wD.sum() * r.max() ** 2
    return dict(x_star=xs, r_B=r_B, second=second, r=r, U=U)


def _path(f_id, instance, d):
    return STATIC / f"f{f_id}_i{instance}_d{d}.npz"


@functools.lru_cache(maxsize=None)
def geometry(f_id, instance, d):
    """The stored basin geometry (built with --build; built and stored here if missing)."""
    p = _path(f_id, instance, d)
    if not p.exists():
        g = basin_geometry(f_id, instance, d)
        STATIC.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{np.random.default_rng().integers(1 << 30)}.part")
        with open(tmp, "wb") as fh:
            np.savez(fh, **g)
        tmp.replace(p)
    z = np.load(p)
    return {k: z[k] for k in z.files}


@functools.lru_cache(maxsize=None)
def random_spheres(f_id, instance, d):
    """The 100 random spheres of the _rnd helpers: centres and weights (the same for all of them)."""
    rng = np.random.default_rng(7_000_000 + 10000 * f_id + 100 * instance + d)
    centres = rng.uniform(LO, HI, (N_RANDOM, d))
    weights = np.linspace(1.1, 9.1, N_RANDOM)
    rng.shuffle(weights)
    return centres, weights


def _shape(name, f_id, instance, d):
    """S^-1 of the pit g0 for helper `name`."""
    G = geometry(f_id, instance, d)
    if name.startswith("g_sph_k"):
        k = float(name.split("_")[2][1:])
        return np.eye(d) / (k * G["r_B"]) ** 2
    if name.startswith("g_sph"):
        return np.eye(d) / d
    k = float(name.split("_")[2][1:])
    vals, vecs = np.linalg.eigh(G["second"])
    vals = np.maximum(vals, vals.max() * 1e-12)
    scale = (k * G["r_B"]) ** 2 / np.exp(np.log(vals).mean())   # geometric-mean radius k r_B
    if "_rot" in name:
        vecs = BA._rotation(9_000_000 + 10000 * f_id + 100 * instance + d, d)
    return vecs @ np.diag(1.0 / (scale * vals)) @ vecs.T


@functools.lru_cache(maxsize=None)
def pointer_helper(f_id, instance, dim, name):
    """Helper `name` as a function of a population X, its minimiser (x*) and the basin of x* (for the monitor)."""
    f_id, instance, d = int(f_id), int(instance), int(dim)
    if not has_pointer(f_id, name):
        raise ValueError(f"{name} does not exist for f{f_id}")
    xs = geometry(f_id, instance, d)["x_star"]
    Sinv = _shape(name, f_id, instance, d)
    rnd = name.endswith("_rnd")
    if rnd:
        C, w = random_spheres(f_id, instance, d)

    def h(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = X - xs
        top = W0 * np.exp(-0.5 * np.einsum("ni,ij,nj->n", Y, Sinv, Y))
        if rnd:
            sq = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            top = np.maximum(top, (w[None, :] * np.exp(-sq / (2.0 * d))).max(axis=1))
        return W0 - top

    q = basin_coordinate(f_id, instance, d, xs)
    return h, xs, lambda X: q(np.atleast_2d(X)) <= 1.0


def check(f_ids, dims, instances, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    for f_id in f_ids:
        for d in dims:
            for inst in instances:
                G = geometry(f_id, inst, d)
                xs = G["x_star"]
                q = basin_coordinate(f_id, inst, d, xs)
                r = G["r"]
                # r(u) is an exit: inside just before it, outside just after it (unless clipped at the box)
                U = G["U"]
                ok_in = (q(xs + 0.999 * r[:, None] * U) <= 1).mean()
                X = np.vstack([rng.uniform(LO, HI, (n, d)), xs + rng.normal(0, 3 * G["r_B"], (n, d))])
                worst = min(float((pointer_helper(f_id, inst, d, h)[0](X) - 0).min()) for h in POINTER_HELPERS)
                at = max(abs(float(pointer_helper(f_id, inst, d, h)[0](xs)[0])) for h in POINTER_HELPERS)
                axes = np.sqrt(np.linalg.eigvalsh(G["second"]) * (d + 2))
                print(f"f{f_id:<2} {d:>2}D i{inst}: r_B = {G['r_B']:.3g}, basin axes {axes.min():.3g}..{axes.max():.3g}, "
                      f"r(u) {r.min():.3g}..{r.max():.3g}, inside before exit {ok_in:.3f}; min h = {worst:.3g}, "
                      f"max |h(x*)| = {at:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fids", nargs="+", type=int, default=list(POINTER_FUNCTIONS))
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--instances", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args()
    if a.build:
        for f_id in a.fids:
            for d in a.dims:
                for inst in a.instances:
                    p = _path(f_id, inst, d)
                    if p.exists():
                        p.unlink()
                    geometry.cache_clear()
                    geometry(f_id, inst, d)
    check(a.fids, a.dims, a.instances)
