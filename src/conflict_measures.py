"""Conflict measures between an objective f and a helper h, computed on a set of points.
Both f and h are minimised. A pair of points is concordant (harmony) when f and h
order it the same way, discordant (conflict) when they order it in opposite ways, and tied when
f or h gives both points the same value.
"""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

NAN = float("nan")


def _as_pair(f, h):
    return np.asarray(f, dtype=np.float64).ravel(), np.asarray(h, dtype=np.float64).ravel()


def _tied_pairs(*cols):
    """Number of pairs that are equal in every given column.
    We sort the rows and cut them into runs of identical values; a run of size s holds s(s-1)/2
    tied pairs.
    """
    n = len(cols[0])
    order = np.lexsort(cols[::-1])
    starts_new_run = np.zeros(n, dtype=bool)
    starts_new_run[0] = True
    for c in cols:
        s = c[order]
        starts_new_run[1:] |= s[1:] != s[:-1]
    sizes = np.diff(np.append(np.flatnonzero(starts_new_run), n)).astype(np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def _scaled_centered(W):
    """Centre each row and divide it by its largest absolute value.
    Pearson ignores scale, and this keeps huge helpers (1/f reaches 1e8) and tiny differences
    away from overflow and underflow. A constant row becomes NaN.
    """
    C = W - W.mean(axis=-1, keepdims=True)
    s = np.abs(C).max(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return C / s


def _window_pearson(Fw, Hw):
    """Pearson of f and h along the last axis of Fw and Hw; NaN where f or h is constant."""
    defined = (Fw.max(-1) > Fw.min(-1)) & (Hw.max(-1) > Hw.min(-1))
    a, b = _scaled_centered(Fw), _scaled_centered(Hw)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(defined, np.clip(r, -1.0, 1.0), NAN)

# Level of conflict on the whole set
def pair_counts(f, h):
    """Concordant, discordant and tied pairs of (f, h), as exact integers.
    """
    f, h = _as_pair(f, h)
    n = len(f)
    n0 = n * (n - 1) // 2
    Tf, Th, Tboth = _tied_pairs(f), _tied_pairs(h), _tied_pairs(f, h)
    tied = Tf + Th - Tboth
    if n0 - Tf > 0 and n0 - Th > 0:
        tau = float(kendalltau(f, h, variant="b").statistic)
        diff = int(round(tau * np.sqrt(float(n0 - Tf) * float(n0 - Th))))
    else:
        diff = 0  # f or h is constant (h0 is), so no pair is concordant or discordant
    untied = n0 - tied
    assert (untied - diff) % 2 == 0, "pair_counts: tau_b does not give an integer C - Dis"
    C, Dis = (untied + diff) // 2, (untied - diff) // 2
    return dict(n=n, n0=n0, C=C, Dis=Dis, Tf=Tf, Th=Th, Tboth=Tboth, tied=tied)


def kendall(f, h):
    """Kendall tau_b of f and h, and p_inc = Dis / n0, the share of pairs in conflict.

    tau_b is NaN when f or h is constant on the set.
    """
    c = pair_counts(f, h)
    den = float(c["n0"] - c["Tf"]) * float(c["n0"] - c["Th"])
    return dict(tau_b=(c["C"] - c["Dis"]) / np.sqrt(den) if den > 0 else NAN, p_inc=c["Dis"] / c["n0"])


def pearson(f, h):
    """Pearson correlation of f and h. NaN if f or h is constant."""
    return float(_window_pearson(*_as_pair(f, h)))


def delta_violations(f, h):
    """Per point y: how much worse on h is the worst point that is not worse than y on f.

    v(y) = max(0, max over x with f(x) <= f(y) of h(x) - h(y)), from Brockhoff & Zitzler. All v
    are 0 exactly when h never disagrees with the order of f.
    """
    f, h = _as_pair(f, h)
    order = np.argsort(f, kind="stable")
    fs, hs = f[order], h[order]
    running_max_h = np.maximum.accumulate(hs)
    # Last sorted position whose f is still <= f(y); ties in f count as violations too.
    last = np.searchsorted(fs, f, side="right") - 1
    return np.maximum(0.0, running_max_h[last] - h)


def delta_stats(f, h):
    """The 95% quantile of the delta violations, and the standard deviation of h to scale it.

    We use the quantile rather than the maximum of Brockhoff & Zitzler, because the maximum is
    set by a single extreme point.
    """
    f, h = _as_pair(f, h)
    v = delta_violations(f, h)
    return dict(delta_q95=float(np.quantile(v, 0.95)), sd_ref_h=float(np.std(h)))


def nondominated_mask(F):
    """True for the rows of F = (f, h) that no other row dominates.

    We sort by f, then by h. A point is dominated if a group with a smaller f already reached an
    h at least as good, or if its own group (same f) has a smaller h. Exact copies of a point do
    not dominate each other, so all of them stay.
    """
    order = np.lexsort((F[:, 1], F[:, 0]))
    f, h = F[order, 0], F[order, 1]
    new_group = np.r_[True, f[1:] != f[:-1]]
    group = np.cumsum(new_group) - 1
    group_min_h = h[new_group]  # the first row of each group has its smallest h
    best_h_before = np.r_[np.inf, np.minimum.accumulate(group_min_h)[:-1]]
    dominated = (best_h_before[group] <= h) | (h > group_min_h[group])
    mask = np.empty(len(F), dtype=bool)
    mask[order] = ~dominated
    return mask


def nd_count(f, h):
    """Number of non-dominated points of the bi-objective problem (f, h)."""
    f, h = _as_pair(f, h)
    return int(nondominated_mask(np.column_stack([f, h])).sum())


def _diameter(P):
    """Largest distance between two rows of P.

    The two farthest points are corners of the convex hull, so for many points we only compare
    the hull corners (we only call this on the 2D grid, where the hull is cheap).
    """
    if len(P) > 3000:
        P = P[ConvexHull(P).vertices]
    best = 0.0
    for a in range(0, len(P), 1000):
        best = max(best, float(cdist(P[a:a + 1000], P).max()))
    return best


def nd_spread(X, f, h):
    """Size of the non-dominated set of (f, h) and the largest distance between two of its points.

    A helper that fights f everywhere makes almost every point non-dominated, spread over the
    whole box; a helper that agrees with f leaves a small set around x*.
    """
    f, h = _as_pair(f, h)
    nd = nondominated_mask(np.column_stack([f, h]))
    return dict(nd_count=int(nd.sum()), nd_max_dist=_diameter(X[nd]))


def level_bundle(f, h):
    """All level measures of one set in a flat dict: tau_b, p_inc, pearson, delta, nd_count."""
    return {**kendall(f, h), "pearson": pearson(f, h), **delta_stats(f, h), "nd_count": nd_count(f, h)}


# Where the conflict is: local correlation (Tusar & Cork 2025) and gradient angles
def knn(X, k):
    """Distances and indices of the k nearest points of every point, the point itself first.
    """
    dist, idx = cKDTree(X).query(X, k=k)
    if not (idx[:, 0] == np.arange(len(X))).all():
        raise ValueError("knn: some point is not its own nearest neighbour (duplicate points?)")
    return dist, idx


def local_corr(f, h, idx, k):
    """Local Pearson correlation r_i of f and h over each point and its k-1 nearest neighbours.

    idx comes from knn. r_i < 0 means that around point i, going down in f tends to go up in h.
    r_i is NaN where f or h is constant in the window (plateaus, or h0).
    """
    f, h = _as_pair(f, h)
    return _window_pearson(f[idx[:, :k]], h[idx[:, :k]])


def window_corr_grid(Fg, Hg):
    """Local correlation on a 2D grid: Pearson of f and h over the 3 x 3 window of each node.

    This is the k = 9 window of an interior node, so grid and scattered maps agree. 
    """
    Fg, Hg = np.asarray(Fg, dtype=np.float64), np.asarray(Hg, dtype=np.float64)
    n1, n2 = Fg.shape
    shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    Fs = np.stack([Fg[1 + i:n1 - 1 + i, 1 + j:n2 - 1 + j] for i, j in shifts], axis=-1)
    Hs = np.stack([Hg[1 + i:n1 - 1 + i, 1 + j:n2 - 1 + j] for i, j in shifts], axis=-1)
    out = np.full(Fg.shape, NAN)
    out[1:-1, 1:-1] = _window_pearson(Fs, Hs)
    return out


def cos_angle(Gf, Gh, tol=1e-8):
    """Cosine of the angle between grad f and grad h at every point.

    cos < 0 means the steepest way down in f is a way up in h. Where either gradient is shorter
    than tol the angle has no meaning and we return NaN (h0 is NaN everywhere).
    """
    nf, nh = np.linalg.norm(Gf, axis=1), np.linalg.norm(Gh, axis=1)
    ok = (nf > tol) & (nh > tol)
    cos = np.full(len(Gf), NAN)
    cos[ok] = np.clip((Gf[ok] * Gh[ok]).sum(1) / (nf[ok] * nh[ok]), -1.0, 1.0)
    return cos


def summarise_map(r):
    """Share of negative values and mean of a map of local values, ignoring NaN.
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    ok = np.isfinite(r)
    n_def = int(ok.sum())
    if n_def == 0:
        return dict(frac_lt0=NAN, mean_r=NAN)
    return dict(frac_lt0=int((r[ok] < 0).sum()) / n_def, mean_r=float(r[ok].mean()))
