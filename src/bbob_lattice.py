"""The conflict-informed helpers on BBOB functions that are not Gallagher functions: one Gallagher
component on every trap of f, the traps read off the BBOB formula (never listed, never searched).

f3, f4, f15 and f24 share the Rastrigin relief  sum_i r(z_i),  r(t) = 10 (1 - cos 2 pi t) <= 20,  in an
internal coordinate z(x) that the formula gives (x* <-> z = 0). The relief repeats with period 1 along every
z axis, so f has one trap next to every point k of the integer lattice Z^D in z:
  - f3, f4, f15: f - f* = sum_i z_i^2 + r(z_i) is separable in z; its traps are the products of the 1D
    minima m(k) of t^2 + r(t) (Newton from t = k; they exist for |k| <= 31);
  - f24: f - f* = min(F0, G1) + sum_i r(z_i) (two funnels); the trap of k is Newton from k on the funnel
    active there plus the relief.

One component per trap, in z:  g_k(x) = exp(-|z(x) - c_k|^2 / (2 s2)),  height B = 20 (the barrier of the
relief), s2 = B / (40 pi^2) (then the curvature of the bump equals the curvature r''(0) of f's well).
Component 0 is the trap of x* (c_0 = 0). All heights are equal and the bumps isotropic in z, so the
highest component at x is the nearest centre in z: for f3/f4/f15 the nearest 1D minimum per coordinate,
for f24 the best of round(z) and its 2D axis neighbours. About 63^D components, none of them stored.

    G = B g_0,  L = max_{k != 0} B g_k
    h_orc_L   = B - G                  the pit of x*'s trap only
    h_cli_L   = B - G + L              h_cli: the pit kept, every other trap turned into a hill
    h_exact_L = f + 2 max(0, L - G)    h_exact: f where the pit is the highest component
    h_conf_L  = L                      the hills only

A control with the same lattice, height and width but NOT on f's traps (hills only, no pit, as h_conf_L):
    h_mis_L   the hills moved by half a period along every z axis: they sit on the crests of f's relief,
              so h is high where f is high (no conflict designed at the traps)

Conflict design: in the cell of x* h and f both rise away from x* (harmony); in every other cell f rises
from the trap bottom while h falls from the hill top (conflict). Far from x*, h has no slope towards x*:
unlike the optimum-centred helpers (bbob_pointer.py) it does not point at x* outside x*'s own cell. Where
f has no trap along some z axis (no optimum beyond |z_i| = m(31) = 30.78 for f3/f4/f15; the hills fade out
by |z_i| ~ 32) h is flat.

Everything comes from the formula and x* (ioh): a white-box oracle. Building it costs 0 evaluations of f.

    python src/bbob_lattice.py [--fids ...] [--dims ...] [--instances ...]   checks the formula and the traps
"""
import argparse
import math

import ioh
import numpy as np
from scipy.optimize import minimize

import bbob_transforms as BA

LATTICE_FUNCTIONS = (3, 4, 15, 24)
LATTICE_HELPERS = ("h_orc_L", "h_cli_L", "h_exact_L", "h_conf_L", "h_mis_L")
B = 20.0
S2 = B / (40.0 * math.pi ** 2)
TWO_PI = 2.0 * math.pi


def has_lattice(f_id, name=None):
    return int(f_id) in LATTICE_FUNCTIONS and (name is None or name in LATTICE_HELPERS)


def r(T):
    return 10.0 * (1.0 - np.cos(TWO_PI * T))


# --- The 1D traps of f3/f4/f15 -------------------------------------------------------------------
def minima_1d():
    """The local minima of t^2 + r(t), one per integer k while they exist (|k| <= 31)."""
    ks, ms = [], []
    for k in range(-40, 41):
        t = float(k)
        for _ in range(60):
            t -= (2 * t + 20 * math.pi * math.sin(TWO_PI * t)) / (2 + 40 * math.pi ** 2 * math.cos(TWO_PI * t))
        curv = 2 + 40 * math.pi ** 2 * math.cos(TWO_PI * t)
        if abs(t - k) < 0.5 and curv > 0 and abs(2 * t + 20 * math.pi * math.sin(TWO_PI * t)) < 1e-9:
            ks.append(k)
            ms.append(t)
    return np.array(ks), np.array(ms)


KS, MS = minima_1d()
MIDS = 0.5 * (MS[1:] + MS[:-1])   # the nearest 1D minimum of t is MS[searchsorted(MIDS, t)]
I0 = int(np.flatnonzero(KS == 0)[0])


# --- The inverse transformations (only for the traps in x, in check()) ----------------------------
def _tosz_inv(Y):
    X = np.zeros_like(Y)
    nz = Y != 0
    ly = np.log(np.abs(Y[nz]))
    pos = Y[nz] > 0
    c1, c2 = np.where(pos, 10.0, 5.5), np.where(pos, 7.9, 3.1)
    lo, hi = ly - 0.2, ly + 0.2
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        big = mid + 0.049 * (np.sin(c1 * mid) + np.sin(c2 * mid)) > ly
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    X[nz] = np.sign(Y[nz]) * np.exp(0.5 * (lo + hi))
    return X


def _tasy_inv(W, beta):
    d = W.shape[1]
    a = np.broadcast_to(beta * np.arange(d) / (d - 1), W.shape)
    out = W.copy()
    pos = W > 0
    A, lw = a[pos], np.log(W[pos])
    lo, hi = np.full_like(lw, -90.0), np.maximum(lw, 0.0) + 1.0
    for _ in range(110):
        mid = 0.5 * (lo + hi)
        big = (1.0 + A * np.exp(mid / 2)) * mid > lw
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    out[pos] = np.exp(0.5 * (lo + hi))
    return out


class Lattice:
    """The internal coordinate z of f, f - f* written from the formula, and the trap centres in z."""

    def __init__(self, f_id, instance, d):
        self.f_id, self.inst, self.d = int(f_id), int(instance), int(d)
        if self.f_id not in LATTICE_FUNCTIONS:
            raise ValueError(f"the trap lattice exists only on f{LATTICE_FUNCTIONS}")
        p = ioh.get_problem(self.f_id, self.inst, self.d)
        self.xs = np.array(p.optimum.x, dtype=float)
        self.fopt = float(p.optimum.y)
        d, i = self.d, np.arange(self.d)
        self.i = i
        if self.f_id in (3, 4, 15):
            seed = self.f_id + 10000 * self.inst
            self.lam = 10 ** (0.5 * i / (d - 1))
            if self.f_id == 15:
                self.A = BA._rotation(seed + 1000000, d)
                self.M = self.A @ np.diag(math.sqrt(10.0) ** (i / (d - 1))) @ BA._rotation(seed, d)
                self.Minv = np.linalg.inv(self.M)
        else:
            seed = 24 + 10000 * self.inst
            self.A, self.Bm = BA._rotation(seed + 1000000, d), BA._rotation(seed, d)
            self.c = 10.0 ** (i / (d - 1))
            self.mu0 = 2.5
            self.s = 1.0 - 0.5 / (math.sqrt(d + 20.0) - 4.1)
            self.mu1 = -math.sqrt((self.mu0 ** 2 - 1.0) / self.s)
            self.sign = np.sign(self.xs)
            self.P = (self.A @ np.diag(1.0 / self.c) @ self.Bm).T   # x_hat - mu0 = P z
            self.PtP = self.P.T @ self.P
            self.delta = (self.mu0 - self.mu1) * np.ones(d)
            self._cache = {}

    def z(self, X):
        X = np.atleast_2d(X)
        if self.f_id == 3:
            return self.lam * BA._tasy(BA._tosz(X - self.xs), 0.2)
        if self.f_id == 4:
            Y = BA._tosz(X - self.xs)
            return np.where((Y > 0) & (self.i % 2 == 0), 10.0, 1.0) * self.lam * Y
        if self.f_id == 15:
            return BA._tasy(BA._tosz((X - self.xs) @ self.A.T), 0.2) @ self.M.T
        return (((2.0 * X * self.sign - self.mu0) @ self.Bm.T) * self.c) @ self.A.T

    def x_of_z(self, Z):
        Z = np.atleast_2d(Z)
        if self.f_id == 3:
            return self.xs + _tosz_inv(_tasy_inv(Z / self.lam, 0.2))
        if self.f_id == 4:
            return self.xs + _tosz_inv(Z / (np.where((Z > 0) & (self.i % 2 == 0), 10.0, 1.0) * self.lam))
        if self.f_id == 15:
            return self.xs + _tosz_inv(_tasy_inv(Z @ self.Minv.T, 0.2)) @ self.A
        return ((((Z @ self.A) / self.c) @ self.Bm) + self.mu0) / (2.0 * self.sign)

    def f(self, X):
        """f - f*, written from the formula."""
        X = np.atleast_2d(X)
        Z = self.z(X)
        if self.f_id in (3, 4, 15):
            pen = 100.0 * BA._penalty(X) if self.f_id == 4 else 0.0
            return (Z ** 2 + r(Z)).sum(axis=1) + pen
        Xh = 2.0 * X * self.sign
        F0 = ((Xh - self.mu0) ** 2).sum(axis=1)
        G1 = self.d + self.s * ((Xh - self.mu1) ** 2).sum(axis=1)
        return np.minimum(F0, G1) + r(Z).sum(axis=1) + 1e4 * BA._penalty(X)

    # --- trap centres in z -------------------------------------------------------------------------
    def _newton_f24(self, K):
        """Newton from lattice point k on the active funnel + relief in z; ok where it is a strict minimum
        within half a period of k."""
        Z = np.atleast_2d(K).astype(float).copy()

        def grad_hess(Z):
            V = Z @ self.P.T
            use0 = (V ** 2).sum(axis=1) <= self.d + self.s * ((V + self.delta) ** 2).sum(axis=1)
            g = np.where(use0[:, None], 2.0 * (V @ self.P), 2.0 * self.s * ((V + self.delta) @ self.P)) \
                + 20 * math.pi * np.sin(TWO_PI * Z)
            H = np.where(use0[:, None, None], 2.0 * self.PtP, 2.0 * self.s * self.PtP) \
                + np.einsum("ni,ij->nij", 40 * math.pi ** 2 * np.cos(TWO_PI * Z), np.eye(self.d))
            return g, H

        for _ in range(8):
            g, H = grad_hess(Z)
            Z = Z - np.linalg.solve(H, g[:, :, None])[:, :, 0]
        g, H = grad_hess(Z)
        ok = (np.abs(g).max(axis=1) < 1e-8) & (np.linalg.eigvalsh(H)[:, 0] > 0) & (np.abs(Z - K).max(axis=1) < 0.5)
        return Z, ok

    def _centres_f24(self, Kc):
        keys = [tuple(k) for k in Kc.astype(int)]
        miss = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if miss:
            C, ok = self._newton_f24(np.array(miss, float))
            self._cache.update({k: (c, o) for k, c, o in zip(miss, C, ok)})
        return np.array([self._cache[k][0] for k in keys]), np.array([self._cache[k][1] for k in keys])

    def centre(self, K):
        """The centres c_k in z of the lattice indices K (n x d), and whether f has a trap there."""
        K = np.atleast_2d(K)
        if self.f_id in (3, 4, 15):
            idx = K - KS[0]
            return MS[np.clip(idx, 0, len(KS) - 1)], ((idx >= 0) & (idx < len(KS))).all(axis=1)
        return self._newton_f24(K)

    def distances(self, X):
        """Squared distance in z to the centre of x*'s trap (0) and to the nearest other trap centre."""
        Z = self.z(X)
        n, d = Z.shape
        if self.f_id in (3, 4, 15):
            j = np.searchsorted(MIDS, Z)                       # nearest 1D minimum per coordinate
            d1 = (Z - MS[j]) ** 2
            jo = np.where(Z > MS[j], j + 1, j - 1)             # the next one, on the other side
            jo_ok = (jo >= 0) & (jo < len(MS))
            d2 = np.where(jo_ok, (Z - MS[np.clip(jo, 0, len(MS) - 1)]) ** 2, np.inf)
            all0 = (j == I0).all(axis=1)                       # nearest is x*'s trap: take the second nearest
            return (Z ** 2).sum(axis=1), d1.sum(axis=1) + np.where(all0, (d2 - d1).min(axis=1), 0.0)
        K0 = np.rint(Z)
        offs = np.vstack([np.zeros(d), np.eye(d), -np.eye(d)])
        Kc = (K0[:, None, :] + offs[None]).reshape(-1, d)
        C, ok = self._centres_f24(Kc)
        dist = ((np.repeat(Z, len(offs), axis=0) - C) ** 2).sum(axis=1)
        dist[~ok | (np.abs(Kc).sum(axis=1) == 0)] = np.inf
        return (Z ** 2).sum(axis=1), dist.reshape(n, len(offs)).min(axis=1)

    def GL(self, X):
        d0, dL = self.distances(X)
        return B * np.exp(-d0 / (2 * S2)), B * np.exp(-dL / (2 * S2))

    def shifted_hills(self, delta):
        """Hills of the same lattice moved by delta (in periods) along the z axes: the control."""
        def hills(X):
            Z = self.z(np.atleast_2d(X)) - delta
            if self.f_id in (3, 4, 15):
                d2 = ((Z - MS[np.searchsorted(MIDS, Z)]) ** 2).sum(axis=1)
            else:
                d2 = ((Z - np.rint(Z)) ** 2).sum(axis=1)
            return B * np.exp(-d2 / (2 * S2))
        return hills


def lattice_helper(f_id, instance, dim, name):
    """Helper `name` as a function of a population X, its minimiser (x*, for h_orc_L and h_exact_L) or
    None, and the cell of x*'s trap (where the pit is the highest component), for the monitor."""
    if not has_lattice(f_id, name):
        raise ValueError(f"{name} does not exist for f{f_id}")
    T = Lattice(f_id, instance, dim)

    def h_orc_L(X):
        return B - T.GL(X)[0]

    def h_cli_L(X):
        G, L = T.GL(X)
        return B - G + L

    def h_exact_L(X):
        G, L = T.GL(X)
        return T.fopt + T.f(X) + 2 * np.maximum(0.0, L - G)

    def h_conf_L(X):
        return T.GL(X)[1]

    helpers = dict(h_orc_L=h_orc_L, h_cli_L=h_cli_L, h_exact_L=h_exact_L, h_conf_L=h_conf_L,
                   h_mis_L=T.shifted_hills(0.5 * np.ones(T.d)))

    def region(X):
        G, L = T.GL(X)
        return (G >= L) & (G > 0)

    x_h = T.xs if name in ("h_orc_L", "h_exact_L") else None
    return helpers[name], x_h, region


# --- Checks ------------------------------------------------------------------------------------------
def check(f_ids, dims, instances, n=2000, n_traps=40, seed=1):
    """f from the formula against ioh, and the traps: a sample of the first-shell centres and of the
    nearest centres of uniform points must be local minima of ioh f (L-BFGS-B started there does not move,
    and no point around is lower)."""
    rng = np.random.default_rng(seed)
    for f_id in f_ids:
        for d in dims:
            for inst in instances:
                T = Lattice(f_id, inst, d)
                p = ioh.get_problem(f_id, inst, d)
                X = rng.uniform(-5, 5, (n, d))
                fio = np.array([p(x) for x in X]) - T.fopt
                port = float(np.max(np.abs(T.f(X) - fio) / np.maximum(1.0, np.abs(fio))))
                # traps: the 2D first shell around x* and the nearest traps of uniform points
                K = np.vstack([np.eye(d, dtype=int), -np.eye(d, dtype=int)])
                Zu = T.z(X[:n_traps])
                K = np.vstack([K, np.rint(Zu).astype(int)])
                C, ok = T.centre(K)
                Xc = T.x_of_z(C[ok])
                inside = (np.abs(Xc) < 5 - 1e-6).all(axis=1)
                Xc = Xc[inside]
                fc = np.array([p(x) for x in Xc])
                moved, lower = [], 0
                for x, fx in zip(Xc, fc):
                    res = minimize(lambda y: p(y), x, method="L-BFGS-B", bounds=[(-5, 5)] * d,
                                   options=dict(ftol=1e-15, gtol=1e-12, maxiter=200))
                    moved.append(float(np.linalg.norm(res.x - x)))
                    dirs = rng.normal(size=(40, d))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    lower += int(min(p(x + 1e-4 * u) for u in dirs) < fx)
                hx = lattice_helper(f_id, inst, d, "h_cli_L")[0](T.xs)[0]
                hmin = float(lattice_helper(f_id, inst, d, "h_cli_L")[0](X).min())
                print(f"f{f_id:<2} {d:>2}D i{inst}: formula vs ioh {port:.1e}; {len(Xc)} traps in the box, "
                      f"L-BFGS-B moves them at most {max(moved):.1e}, {lower} with a lower point around; "
                      f"h_cli_L(x*) = {hx:.2e}, min over {n} uniform points {hmin:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fids", nargs="+", type=int, default=[15, 24])
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--instances", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args()
    check(a.fids, a.dims, a.instances)
