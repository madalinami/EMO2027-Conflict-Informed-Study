"""Conflict between f and the helper h, measured while an optimiser runs.

The monitor is a pymoo callback. It sees every point the optimiser evaluates (after mutation), the
two parents of every child, and the population at the end of every generation g. Each generation
it looks at three sets:
  S(g)  the population after survival,
  O(g)  the children evaluated in generation g,
  U(g)  S(g-1) together with O(g), the set that survival chooses from.
In generation 1 there are no parents yet: O(1) is the initial sample and U(1) = O(1).
U(g) and O(g) keep each distinct x once (MOEA/D can evaluate the same x twice).
MOEA/D replaces points one child at a time, so for MOEA/D U(g) is only "everything generation g
saw", not the input of one selection step.

The monitor draws no random numbers, so a run with the monitor is the same run as without it.
The values of one generation are added up over blocks of generations (see block_edges), so even a
run of 15000 generations gives a small file.
"""
import sys
from pathlib import Path

import numpy as np
from pymoo.core.callback import Callback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conflict_measures as CM  # noqa: E402

NAN = float("nan")
K_LOCAL = 10                                  # local correlation window: a point and its 9 nearest points
TARGETS = ("1e-1", "1e-3", "1e-5", "1e-8")    # precision targets f - f* for the hitting times


# --- Blocks of generations ---------------------------------------------------------------------
def block_edges(B):
    """Edges of the blocks: block k holds the generations g with edges[k] < g <= edges[k+1].

    Each of the first 100 generations is its own block. After that, about 100 blocks evenly spaced
    in log g, plus edges at 10, 20, 30, 90 and 95% of the run. The last generation is a block alone.
    """
    B = int(B)
    edges = list(range(0, min(B, 100) + 1))
    if B > 100:
        edges += list(np.round(np.geomspace(100, B, 101)).astype(int))
    edges += list(np.round(B * np.array([0.1, 0.2, 0.3, 0.9, 0.95])).astype(int))
    edges += [B - 1, B]
    return np.unique(np.clip(edges, 0, B))


class BlockStore:
    """Per-generation values added up over the blocks of block_edges(B).

    Integers (counts) are summed. For floats (tau_b, p_inc, distances, ...) we keep the number of
    generations where the value was defined, and the sum and sum of squares of (value minus the
    first value of the block), which give the block mean and sd. NaN means "not defined in this
    generation" and is left out.
    """

    def __init__(self, B):
        self.edges = block_edges(B)
        n = len(self.edges) - 1
        self.n_gen = np.zeros(n, dtype=np.int64)
        self.counts, self.first, self.sums, self.squares, self.n_def = {}, {}, {}, {}, {}

    def add(self, g, values):
        b = int(np.searchsorted(self.edges, g, side="left")) - 1
        n = len(self.n_gen)
        if not 0 <= b < n:
            raise ValueError(f"generation {g} is outside the budget")
        self.n_gen[b] += 1
        for key, v in values.items():
            if isinstance(v, (int, np.integer)):
                if key not in self.counts:
                    self.counts[key] = np.zeros(n, dtype=np.int64)
                self.counts[key][b] += int(v)
                continue
            if key not in self.sums:
                self.first[key], self.sums[key], self.squares[key] = np.zeros(n), np.zeros(n), np.zeros(n)
                self.n_def[key] = np.zeros(n, dtype=np.int64)
            if np.isfinite(v):
                # We add v minus the first value of the block, so a constant value gets sd exactly 0.
                if self.n_def[key][b] == 0:
                    self.first[key][b] = v
                d = v - self.first[key][b]
                self.sums[key][b] += d
                self.squares[key][b] += d * d
                self.n_def[key][b] += 1

    def result(self):
        """Arrays with one entry per block: lo and hi (the block holds generations lo+1 .. hi),
        n_gen, the integers under their own name, and each float as `name_mean`, `name_sd` and
        `name_ndef` (the number of generations where it was defined)."""
        out = dict(lo=self.edges[:-1], hi=self.edges[1:], n_gen=self.n_gen)
        out.update(self.counts)
        with np.errstate(invalid="ignore", divide="ignore"):
            for key, s in self.sums.items():
                n = self.n_def[key]
                m = s / n
                out[f"{key}_mean"] = self.first[key] + m
                out[f"{key}_sd"] = np.sqrt(np.maximum(self.squares[key] / n - m * m, 0.0))
                out[f"{key}_ndef"] = n
        return out


# --- Measures of one generation ----------------------------------------------------------------
def pair_counts(a, b):
    """Concordant, discordant and tied pairs of two value lists, comparing every pair directly.
    Our sets have at most 100 points, where this is faster than scipy's kendalltau."""
    i, j = np.triu_indices(len(a), 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    ta, tb = da == 0, db == 0
    return dict(n=len(a), n0=len(i), C=int((da * db > 0).sum()), Dis=int((da * db < 0).sum()),
                Tf=int(ta.sum()), Th=int(tb.sum()), Tboth=int((ta & tb).sum()))


def tau_b(c):
    """Kendall tau_b from pair counts; NaN when one of the two lists is constant."""
    den = float(c["n0"] - c["Tf"]) * float(c["n0"] - c["Th"])
    return (c["C"] - c["Dis"]) / np.sqrt(den) if den > 0 else NAN


def level(F, prefix):
    """Pair counts of (f, h) on a set of points, Kendall tau_b and p_inc = Dis / n0.

    tau_b is NaN when f or h is constant on the set (h0 always), p_inc when there is no pair.
    """
    c = pair_counts(F[:, 0], F[:, 1])
    out = {f"{prefix}_{key}": v for key, v in c.items()}
    out[f"{prefix}_tau_b"] = tau_b(c)
    out[f"{prefix}_p_inc"] = c["Dis"] / c["n0"] if c["n0"] > 0 else NAN
    return out


def local_counts(X, F, prefix, k=K_LOCAL):
    """Local correlation r_i of f and h over each point and its k-1 nearest points. We keep how many r_i are defined, how many are negative, and the window radius:
    the mean distance from a point to the farthest of its k-1 nearest points. Nothing is defined
    when there are fewer than k points."""
    if len(X) < k:
        return {f"{prefix}_loc_def": 0, f"{prefix}_loc_lt0": 0, f"{prefix}_loc_radius": NAN}
    dist, idx = CM.knn(X, k)
    r = CM.local_corr(F[:, 0], F[:, 1], idx, k)
    ok = np.isfinite(r)
    return {f"{prefix}_loc_def": int(ok.sum()), f"{prefix}_loc_lt0": int((r[ok] < 0).sum()),
            f"{prefix}_loc_radius": float(dist[:, k - 1].mean())}


def steps(off_F, par_F):
    """Child minus parent on (f, h): one row per (child, parent) pair, each child with both parents."""
    return (np.asarray(off_F, float)[:, None, :] - np.asarray(par_F, float)).reshape(-1, 2)


def parent_offspring_counts(off_F, par_F):
    """Compare every child with each of its two parents, on (f, h).

    off_F has shape (m, 2), par_F (m, 2, 2): the (f, h) of the two parents of each child. A pair is
    'off_dom' when the child dominates the parent, 'par_dom' when the parent dominates the child,
    'equal' when f and h are both unchanged, and 'incomp' when the child is better on one
    objective and worse on the other.
    """
    d = steps(off_F, par_F)
    equal = (d == 0).all(axis=1)
    off_dom = (d <= 0).all(axis=1) & ~equal
    par_dom = (d >= 0).all(axis=1) & ~equal
    return dict(n_pairs=len(d), off_dom=int(off_dom.sum()), par_dom=int(par_dom.sum()),
                incomp=int(len(d) - off_dom.sum() - par_dom.sum() - equal.sum()),
                equal=int(equal.sum()))


def parent_offspring(off_F, par_F, accepted):
    """The four classes of parent_offspring_counts, the same counts over the children that made
    it into S(g), and the correlation of the steps (df, dh) from parent to child.

    A negative correlation means that the steps that improve f tend to make h worse.
    """
    out = {f"po_{key}": v for key, v in parent_offspring_counts(off_F, par_F).items()}
    acc = parent_offspring_counts(off_F[accepted], par_F[accepted])
    out.update({f"po_acc_{key}": acc[key] for key in ("off_dom", "par_dom", "incomp", "equal")})
    d = steps(off_F, par_F)
    out["po_pearson"] = CM.pearson(d[:, 0], d[:, 1])
    out["po_tau_b"] = tau_b(pair_counts(d[:, 0], d[:, 1]))
    return out


def best_slot(FS, ids):
    """The slot with the smallest f. On ties (plateaus, as in f7) the oldest point, so the choice
    does not depend on how pymoo orders the population."""
    return int(np.lexsort((ids, FS[:, 0]))[0])


def population(XS, FS, ids, x_star, x_h):
    """Where the population S(g) is. XS, FS and ids (the point ids) hold every slot of the
    population (MOEA/D can keep the same point in several slots).

    S_n and S_nd count distinct points. best = best_slot, the centroid is the mean over the slots,
    and the spread is the mean distance of the slots to the centroid. The distances to x* and x_h
    are stored as log10 (floored at 1e-16), because they shrink by orders of magnitude; the spread
    is stored as it is. x_h is the minimiser of the helper, None when it is not known (only h1, h2
    and h3 have one).
    """
    best = XS[best_slot(FS, ids)]
    centroid = XS.mean(axis=0)
    _, first = np.unique(XS, axis=0, return_index=True)
    nd = CM.nondominated_mask(FS[np.sort(first)])

    def log_dist(a, b):
        return NAN if b is None else float(np.log10(max(np.linalg.norm(a - b), 1e-16)))

    return {"S_n": len(first), "S_nd": int(nd.sum()), "S_all_nd": int(nd.all()),
            "S_log_dist_best_xstar": log_dist(best, x_star),
            "S_log_dist_cent_xstar": log_dist(centroid, x_star),
            "S_log_dist_best_xh": log_dist(best, x_h),
            "S_log_dist_cent_xh": log_dist(centroid, x_h),
            "S_spread": float(np.linalg.norm(XS - centroid, axis=1).mean())}


def inside_outside(FU, in_U, off_F=None, par_F=None, in_children=None):
    """Conflict inside and outside a region of the space (for the Gallagher functions: the basin of
    x*). The level on the points of U(g) inside it (Uin_) and outside it (Uout_), and the
    parent-offspring classes of the children born inside (poin_) and outside (poout_)."""
    out = {**level(FU[in_U], "Uin"), **level(FU[~in_U], "Uout")}
    if off_F is not None:
        for name, mask in (("poin", in_children), ("poout", ~in_children)):
            counts = parent_offspring_counts(off_F[mask], par_F[mask])
            out.update({f"{name}_{key}": v for key, v in counts.items()})
    return out


# --- The callback ------------------------------------------------------------------------------
def _unique_points(ids, X):
    """The ids whose point x appears for the first time in the list (exact float64 equality)."""
    seen, keep = set(), []
    for i in ids:
        key = X[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


class ConflictMonitor(Callback):
    """Collects the conflict measures of one run.

    Every evaluated point gets an id, its position in the order of evaluation (from 0). We keep its
    x, its (f, h), the generation it was born in, the last generation it was in the population, and
    the ids of its two parents (-1 for the initial sample).

    B            number of generations of the run (saved as gen_budget)
    f_star       f at the global optimum
    x_star       the global minimiser of f
    x_h          the minimiser of the helper, or None
    save_points  if True, result() also returns every evaluated point, from which the sets of every
                 generation can be rebuilt later (for MOEA/D without the number of slots that hold
                 each point; large: for small subsets only)
    region       None, or a function of a population X that is True for the points inside a region
                 we also watch on its own (for the Gallagher functions: the basin of x*). Then we
                 also keep S_in, the number of distinct points of S(g) inside it, and the measures
                 of inside_outside.
    """

    def __init__(self, B, f_star, x_star, x_h=None, save_points=False, region=None):
        super().__init__()
        self.B = int(B)
        self.f_star = float(f_star)
        self.x_star = np.asarray(x_star, float)
        self.x_h = None if x_h is None else np.asarray(x_h, float)
        self.save_points = save_points
        self.region = region
        self.inside = []        # one entry per evaluated point: is it inside the region?
        self.blocks = BlockStore(B)
        # Best point and centroid of S(g) at the last generation of each block
        self.block_end = {g: b for b, g in enumerate(self.blocks.edges[1:])}
        self.best_x_end = np.full((len(self.block_end), len(self.x_star)), NAN)
        self.centroid_end = np.full((len(self.block_end), len(self.x_star)), NAN)
        self.gen = 0
        # One entry per evaluated point
        self.X, self.F, self.born, self.last, self.p1, self.p2 = [], [], [], [], [], []
        # One entry each time a point beats the best f seen so far
        self.events = dict(eval=[], gen=[], f=[], h=[], x=[])
        self.best_f = np.inf
        # One entry per generation: (f, h) of the best point of S(g) (first slot on ties, as the old
        # code saved it), and the best f seen so far
        self.gen_best_F, self.gen_bsf_f = [], []
        # First generation where all of S(g) is non-dominated, -1 if never. Under MOEA/D the
        # population falls to a few distinct points almost at once, so there it is small (1 to 7)
        # and says little; with h0 it is always -1.
        self.first_all_nd = -1
        self.S_prev = []        # ids of S(g-1)
        self.first_new = 0      # id of the first point evaluated in the current generation

    # pymoo calls this once, at the end of generation 1, on its own copy of the algorithm. The
    # initial sample is already evaluated then, and notify registers it. From then on:
    # - evaluator._eval registers each point as it is evaluated (it gets only the new points),
    # - crossover.do writes the ids of the two parents on each child.
    def initialize(self, algorithm):
        evaluator = algorithm.evaluator
        evaluate = evaluator._eval

        def evaluate_and_register(problem, pop, *args, **kwargs):
            out = evaluate(problem, pop, *args, **kwargs)
            for ind in pop:
                self._register(ind, self.gen + 1)
            return out

        evaluator._eval = evaluate_and_register

        crossover = algorithm.mating.crossover
        cross = crossover.do

        def cross_and_stamp(problem, pop, parents=None, *args, **kwargs):
            off = cross(problem, pop, parents, *args, **kwargs)
            # pymoo returns the first child of every mating, then the second child of every
            # mating, so child r comes from mating r % n_matings
            matings = pop if parents is None else [pop[m] for m in parents]
            n_matings = len(matings)
            for r, child in enumerate(off):
                child.set("p1", matings[r % n_matings][0].get("pid"))
                child.set("p2", matings[r % n_matings][1].get("pid"))
            return off

        crossover.do = cross_and_stamp

    def _register(self, ind, g):
        if ind.get("pid") is not None:
            raise RuntimeError("a point was evaluated twice")
        p1, p2 = ind.get("p1"), ind.get("p2")
        if g > 1 and (p1 is None or p2 is None):
            raise RuntimeError("a child has no parents: the crossover hook was bypassed")
        pid = len(self.F)
        ind.set("pid", pid)
        x, F = np.array(ind.X, dtype=float), np.array(ind.F, dtype=float)
        self.X.append(x)
        self.F.append(F)
        self.born.append(g)
        self.last.append(-1)
        self.p1.append(-1 if p1 is None else int(p1))
        self.p2.append(-1 if p2 is None else int(p2))
        if F[0] < self.best_f:
            self.best_f = F[0]
            for key, v in zip(("eval", "gen", "f", "h", "x"), (pid + 1, g, F[0], F[1], x)):
                self.events[key].append(v)

    def notify(self, algorithm):
        g = algorithm.n_gen
        if g != self.gen + 1:
            raise RuntimeError(f"expected generation {self.gen + 1}, got {g}")
        if g == 1:
            for ind in algorithm.off:
                self._register(ind, 1)
        self.gen = g
        X, F = self.X, self.F

        # The sets as lists of point ids. S_prev comes first in U, so the first n_prev points of U
        # are the distinct points of S(g-1).
        children = list(range(self.first_new, len(F)))
        if self.region is not None and children:
            self.inside.extend(bool(v) for v in self.region(np.array([X[i] for i in children])))
        S_ids = [ind.get("pid") for ind in algorithm.pop]
        if None in S_ids:
            raise RuntimeError("a point of the population was never registered")
        for i in S_ids:
            self.last[i] = g
        n_prev = len(_unique_points(self.S_prev, X))
        U_ids = _unique_points(self.S_prev + children, X)
        O_ids = _unique_points(children, X)

        XU, FU = np.array([X[i] for i in U_ids]), np.array([F[i] for i in U_ids])
        FO = np.array([F[i] for i in O_ids]).reshape(-1, 2)
        XS, FS = np.array([X[i] for i in S_ids]), np.array([F[i] for i in S_ids])

        # P = the S(g-1) part of U. U minus P gives the pairs that involve a child.
        values = {**level(FU, "U"), **level(FU[:n_prev], "P"), **local_counts(XU, FU, "U"),
                  "U_nd": CM.nd_count(FU[:, 0], FU[:, 1]), **level(FO, "O"),
                  **population(XS, FS, S_ids, self.x_star, self.x_h)}
        # Every evaluated child against its two parents, copies included
        off_F = par_F = None
        if g > 1:
            in_S = set(S_ids)
            off_F = np.array([F[i] for i in children])
            par_F = np.array([[F[self.p1[i]], F[self.p2[i]]] for i in children])
            accepted = np.array([i in in_S for i in children])
            values.update(parent_offspring(off_F, par_F, accepted))
        if self.region is not None:
            ins = self.inside
            values["S_in"] = int(sum(ins[i] for i in _unique_points(S_ids, X)))
            values.update(inside_outside(FU, np.array([ins[i] for i in U_ids]), off_F, par_F,
                                         np.array([ins[i] for i in children])))
        self.blocks.add(g, values)

        if g in self.block_end:
            self.best_x_end[self.block_end[g]] = XS[best_slot(FS, S_ids)]
            self.centroid_end[self.block_end[g]] = XS.mean(axis=0)
        if self.first_all_nd < 0 and values["S_all_nd"]:
            self.first_all_nd = g
        self.gen_best_F.append(FS[int(np.argmin(FS[:, 0]))])
        self.gen_bsf_f.append(self.best_f)
        self.S_prev = S_ids
        self.first_new = len(F)

    def result(self):
        """Everything the run collected, as a flat dict (these become the keys of the .npz file).

        blk_*         one value per block of generations (see BlockStore.result), plus
                      blk_best_x_end and blk_centroid_end: best point and centroid of S at the
                      last generation of each block
        gen_best_F    per generation (row 0 = generation 1): (f, h) of the best point of S(g)
        gen_bsf_f     per generation: the best f found so far
        bsf_*         one row each time the best f found so far improved: evaluation number (from 1),
                      generation, f, h, x
        evals_to_<t>  the first evaluation number with f - f* <= t, -1 if never
        pt_*          (only with save_points) every evaluated point; row i = point id i. pt_in says
                      whether it is inside the region (only when a region is given)
        """
        out = dict(gen_budget=self.B, n_gen=self.gen, n_evals=len(self.F), f_star=self.f_star,
                   x_star=self.x_star, first_all_nd_gen=self.first_all_nd,
                   gen_best_F=np.array(self.gen_best_F), gen_bsf_f=np.array(self.gen_bsf_f),
                   bsf_eval=np.array(self.events["eval"], dtype=np.int64),
                   bsf_gen=np.array(self.events["gen"], dtype=np.int64),
                   bsf_f=np.array(self.events["f"]), bsf_h=np.array(self.events["h"]),
                   bsf_x=np.array(self.events["x"]).reshape(-1, len(self.x_star)),
                   blk_best_x_end=self.best_x_end, blk_centroid_end=self.centroid_end,
                   **{f"blk_{key}": v for key, v in self.blocks.result().items()})
        if self.x_h is not None:
            out["x_h"] = self.x_h
        prec = out["bsf_f"] - self.f_star
        for t in TARGETS:
            hit = np.flatnonzero(prec <= float(t))
            out[f"evals_to_{t}"] = int(out["bsf_eval"][hit[0]]) if len(hit) else -1
        if self.save_points:
            out.update(pt_x=np.array(self.X), pt_F=np.array(self.F),
                       pt_born=np.array(self.born, dtype=np.int32),
                       pt_last=np.array(self.last, dtype=np.int32),
                       pt_p1=np.array(self.p1, dtype=np.int64),
                       pt_p2=np.array(self.p2, dtype=np.int64))
            if self.region is not None:
                out["pt_in"] = np.array(self.inside, dtype=bool)
        return out


def load_run(path):
    """One run file written by the benchmark, as a dict of arrays."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}
