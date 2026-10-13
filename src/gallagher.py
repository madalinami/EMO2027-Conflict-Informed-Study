"""Gallagher-like functions and the conflict-informed helper.

f(x) = w0 - max_i w_i g_i(x), where each g_i is a Gaussian bump centred at mu_i and component 0
(weight w0, the largest) holds the global minimum f(mu0) = 0. The conflict-informed helper flips
the local components that are local optima of f: h(x) = w0 - w0 g0(x) + max_{i in L} w_i g_i(x).

The noise of the helper:
- is applied to the local components only. Component 0 is kept exact, so that x* stays the
  minimum of the helpers built on it (h_orc, h_exact); the valley error x0 moves component 0 instead;
- the weight error never makes a local component deeper than the deepest local component of f, so
  no noisy trap can beat component 0 at x*;
- the elongation error (alpha_error) only stretches the component and keeps its long axis.
"""
import copy

import numpy as np

import bbob_gallagher_port as GP


class GaussianComponent:
    def __init__(self, center, weight, alpha, rotation=None, rng=None, is_global=False):
        self.center = np.asarray(center, dtype=float)
        self.n_dim = self.center.shape[0]
        self.weight = float(weight)
        self.alpha = float(alpha)
        self.is_global = is_global
        self.rotation = np.eye(self.n_dim) if rotation is None else np.asarray(rotation, dtype=float)
        self.C = self._conditioning_matrix(rng or np.random.default_rng())

    def _conditioning_matrix(self, rng):
        n = self.n_dim
        if n == 1:
            exponents = np.array([0.0])
        else:
            exponents = np.arange(n) / (n - 1) - 0.5
        rng.shuffle(exponents)            # which axis is long
        self.exponents = exponents
        return np.diag(self.alpha ** exponents)

    def value(self, x):
        x = np.asarray(x, dtype=float)

        # Single point: shape (n_dim,)
        if x.ndim == 1:
            z = self.rotation @ (x - self.center)
            exponent = -(z @ self.C @ z) / (2.0 * self.n_dim)
            return self.weight * np.exp(exponent)

        # Population: shape (n_points, n_dim)
        elif x.ndim == 2:
            z = (x - self.center) @ self.rotation.T
            exponent = -np.einsum("ni,ij,nj->n", z, self.C, z) / (2.0 * self.n_dim)
            return self.weight * np.exp(exponent)

        raise ValueError(f"Expected x with 1 or 2 dimensions, got shape {x.shape}")

    def __call__(self, x):
        return self.value(x)


class GallagherFunction:
    def __init__(
        self,
        n_components: int,
        n_dim: int,
        bounds=(-5.0, 5.0),
        global_weight: float = 10.0,
        other_weight_range=(1.1, 9.1),
        global_alpha: float = 100.0,
        other_alpha_range=(1.1, 100.0),
        use_rotation: bool = True,
        seed: int | None = None,
    ):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        self.n_components = n_components
        self.n_dim = n_dim
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

        lo, hi = bounds
        self.components: list[GaussianComponent] = []

        # component 0: global component
        global_center = self.rng.uniform(lo, hi, size=n_dim)
        global_rotation = self._random_rotation() if use_rotation else None
        self.components.append(GaussianComponent(global_center, global_weight, global_alpha, rotation=global_rotation, rng=self.rng, is_global=True))

        # components 1..n-1: local components (distractors)
        weights = np.linspace(other_weight_range[0], other_weight_range[1], n_components - 1)
        self.rng.shuffle(weights)
        for w in weights:
            center = self.rng.uniform(lo, hi, size=n_dim)
            alpha = self.rng.uniform(*other_alpha_range)
            rotation = self._random_rotation() if use_rotation else None
            self.components.append(GaussianComponent(center, w, alpha, rotation=rotation, rng=self.rng, is_global=False))

        self.x_opt = self.components[0].center
        self._offset = global_weight

        self.dominating_components = [] # local optimas are located at the center of these components
        self.dominated_components = [] # components that were dominated by neighbour component
        for i, component in enumerate(self.components):
            if component.is_global:
                continue
            if self.active_component(component.center) == i:
                self.dominating_components.append(component)
            else:
                self.dominated_components.append(component)

    def _random_rotation(self):
        n = self.n_dim
        a = self.rng.standard_normal((n, n))
        q, r = np.linalg.qr(a)
        return q @ np.diag(np.sign(np.diag(r)))

    def component_values(self, x):
        x = np.asarray(x, dtype=float)
        values = np.stack([c.value(x) for c in self.components], axis=-1)
        return values

    def active_component(self, x):
        return int(np.argmax(self.component_values(x)))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        values = self.component_values(x)
        if x.ndim == 1:
            return self._offset - np.max(values)
        elif x.ndim == 2:
            return self._offset - np.max(values, axis=1)
        raise ValueError(f"Expected x with 1 or 2 dimensions, got shape {x.shape}")

    @property
    def global_component(self):
        return self.components[0]


def _random_small_rotation(n_dim, std_angle, rng):
    if std_angle <= 0.0:
        return np.eye(n_dim)
    if n_dim == 2:
        angle = rng.normal(0, std_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    else:
        A = rng.normal(0, std_angle, size=(n_dim, n_dim))
        Skew = (A - A.T) / 2.0
        Q, R = np.linalg.qr(np.eye(n_dim) + Skew)
        return Q @ np.diag(np.sign(np.diag(R)))


class ComponentLevelInversionHelper:
    """The conflict-informed helper h_cli = w0 - w0 g0 + max over the local optima of w_i g_i, built
    from the components of f, with optional errors in the local components:
      miss_rate    share of local optima left out
      mu_error     the centres move by a normal step whose sd is mu_error times a log-normal factor
      w_error      the weights are multiplied by a log-normal factor with sigma w_error, and kept
                   at most as large as the largest local weight of f
      alpha_error  the elongations are multiplied by a log-normal factor with sigma alpha_error
      rot_error    the components turn by a random rotation with angle sd rot_error (radians)
    and an optional error in component 0:
      x0_shift     the centre of component 0 moves away from x* by x0_shift of its own widths
    """
    def __init__(self, gallagher: GallagherFunction, seed=0, miss_rate=0.0, mu_error=0.0, w_error=0.0, alpha_error=0.0, rot_error=0.0,
                 x0_shift=0.0):
        self.rng = np.random.default_rng(seed)
        self.gallagher = gallagher
        self.bounds = gallagher.bounds

        dominating_components = [copy.deepcopy(c) for c in gallagher.dominating_components]
        global_component = copy.deepcopy(gallagher.global_component)

        # Missing local optima
        if miss_rate > 0.0 and len(dominating_components) > 0:
            n_keep = max(0, int(np.round(len(dominating_components) * (1.0 - miss_rate))))
            if n_keep < len(dominating_components):
                keep_indices = self.rng.choice(len(dominating_components), size=n_keep, replace=False)
                dominating_components = [dominating_components[i] for i in keep_indices]

        # The other errors touch the local components only
        noisy_components = dominating_components

        # Component centre error
        if mu_error > 0.0:
            lo, hi = self.bounds
            for comp in noisy_components:
                sigma_i = mu_error * self.rng.lognormal(mean=0.0, sigma=0.25)
                offset = self.rng.normal(0, sigma_i, size=comp.center.shape)
                comp.center = np.clip(comp.center + offset, lo, hi)

        # Component depth error
        if w_error > 0.0:
            deepest = max(c.weight for c in gallagher.components[1:])
            for comp in noisy_components:
                scale = self.rng.lognormal(mean=0.0, sigma=w_error)
                comp.weight = float(min(comp.weight * scale, deepest))

        # Covariance matrix errors
        if alpha_error > 0.0 or rot_error > 0.0:
            for comp in noisy_components:
                if alpha_error > 0.0:
                    scale = self.rng.lognormal(mean=0.0, sigma=alpha_error)
                    comp.alpha = float(np.clip(comp.alpha * scale, 1.0, 1000.0))
                    comp.C = np.diag(comp.alpha ** comp.exponents)   # same long axis, new elongation
                if rot_error > 0.0:
                    R_err = _random_small_rotation(comp.n_dim, rot_error, self.rng)
                    comp.rotation = R_err @ comp.rotation

        # Valley error: the centre of component 0 moves away from x* by x0_shift in units of its own
        # width (Mahalanobis distance), in one random direction per function that all the levels share.
        # It has its own generator, so the draws of the errors above do not depend on it.
        if x0_shift > 0.0:
            n = global_component.n_dim
            u = np.random.default_rng(seed + 50_000).normal(size=n)
            u /= np.linalg.norm(u)
            step = x0_shift * np.sqrt(n) * global_component.rotation.T @ (np.diag(global_component.C) ** -0.5 * u)
            lo, hi = self.bounds
            global_component.center = np.clip(global_component.center + step, lo, hi)

        self.global_component = global_component
        self.dominating_components = dominating_components
        self.components = [global_component] + dominating_components

        self._offset = self.global_component.weight
        self.x_opt = self.global_component.center

    def local_value(self, x):
        x = np.asarray(x)
        if not self.dominating_components:
            if x.ndim == 1:
                return 0.0
            elif x.ndim == 2:
                return np.zeros(x.shape[0])
            raise ValueError(f"Expected x with 1 or 2 dimensions, got shape {x.shape}")

        vals = np.stack([c.value(x) for c in self.dominating_components], axis=-1)
        return np.max(vals, axis=-1)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return self._offset - self.global_component.value(x) + self.local_value(x)


# --- The helpers we compare on a Gallagher function -----------------------------------------
# h_cli    the conflict-informed helper
# h_orc    only its first part, w0 - w0 g0: it points to the global optimum, with no conflict
# h_conf   only its second part, the flipped local optima: conflict, with no pointer to x*
# h_exact  w0 - 2 w0 g0 + max_{i>=0} w_i g_i, equal to f wherever component 0 is the largest, so in
#          exact harmony with f around x*. h_cli is not: the tails of the local components reach into
#          the basin of x* and move its minimum away from x*, a lot in 2D and 5D, little in 10D.
# h5       -f, conflict everywhere
# h0       constant
GALLAGHER_HELPERS = ("h0", "h_cli", "h_orc", "h_conf", "h_exact", "h5")
NOISY_HELPERS = ("h_cli", "h_conf", "h_exact")   # the ones built from the local components


def gallagher_helper(f, name, seed=0, **noise):
    """Helper `name` of the Gallagher function f, as a function of a population X (n_points, D).
    Also returns its minimiser when we know it exactly (h_orc and h_exact: x* of f), else None."""
    cli = ComponentLevelInversionHelper(f, seed=seed, **noise)
    g0, w0 = cli.global_component, cli.global_component.weight
    helpers = {
        "h0": lambda X: np.zeros(len(X)),
        "h_cli": cli,
        "h_orc": lambda X: w0 - g0.value(X),
        "h_conf": cli.local_value,
        "h_exact": lambda X: w0 - 2 * g0.value(X) + np.maximum(g0.value(X), cli.local_value(X)),
        "h5": lambda X: -f(X),
    }
    x_h = f.x_opt if name in ("h_orc", "h_exact") else None
    return helpers[name], x_h


# --- The functions of our runs ------------------------------------------------------------------
# 100 components, global alpha 100, local alphas between 50 and 100. The runs build f and h
# through make_pair.
GALLAGHER_DEFAULTS = dict(components=100, global_alpha=100.0, local_alpha=(50.0, 100.0))

# Noise of the helper, written as on the command line: "none", "rho=0.2" or "mu=0.4,w=0.5"
NOISE_KEYS = dict(rho="miss_rate", mu="mu_error", w="w_error", alpha="alpha_error", rot="rot_error", x0="x0_shift")
# The 25 main noise settings (rot in radians: 10, 30, 60, 90 and 180 degrees)
NOISE_SETTINGS = ("rho=0.2", "rho=0.4", "rho=0.6", "rho=0.8", "rho=1",
                  "mu=0.2", "mu=0.4", "mu=0.6", "mu=0.8", "mu=1",
                  "w=0.2", "w=0.4", "w=0.6", "w=0.8", "w=1",
                  "alpha=0.25", "alpha=0.5", "alpha=1", "alpha=2", "alpha=3",
                  "rot=0.17", "rot=0.52", "rot=1.05", "rot=1.57", "rot=3.14")


def parse_noise(text):
    if text == "none":
        return {}
    return {NOISE_KEYS[key]: float(value) for key, value in (part.split("=") for part in text.split(","))}


def make_pair(dim, f_seed, helper, noise="none", params=GALLAGHER_DEFAULTS):
    """Gallagher function number f_seed in dimension dim, and its helper `helper` with the noise
    `noise`, exactly as in the runs. Returns f, h (both take a population X) and x_h (or None)."""
    f = GallagherFunction(params["components"], dim, seed=f_seed, global_alpha=params["global_alpha"],
                          other_alpha_range=tuple(params["local_alpha"]))
    h, x_h = gallagher_helper(f, helper, seed=1000 + f_seed, **parse_noise(noise))
    return f, h, x_h


def in_basin(f, X):
    """True where component 0 is the largest component of f: the basin of x* (only there can f go
    below w0 minus the largest local weight)."""
    return f.component_values(X).argmax(axis=-1) == 0


# --- The same helpers on BBOB f21 and f22, which are Gallagher functions -------------------------
BBOB_GALLAGHER_HELPERS = ("h_cli", "h_exact", "h_orc", "h_conf")


def bbob_gallagher_helper(f_id, instance, dim, name):
    """Helper `name` of BBOB f21 or f22, built from the exact peaks of the BBOB generator (rebuilt in
    bbob_gallagher_port.py), with the visible local peaks only, as on our Gallagher functions.

    f21 and f22 are an increasing function of w0 - max_i w_i g_i, so the helpers mean the same as on
    our Gallagher functions. Returns h (a function of a population X), its minimiser (the global
    peak, for h_orc and h_exact) or None, and the basin of x* (True where the global peak is the
    largest), for the monitor.
    """
    P = GP._gallagher_params_cached(int(f_id), int(instance), int(dim))
    local = GP._visible_mask(P)
    local[0] = False                  # the visible peaks other than the global one
    w0 = P["w"][0]

    def peaks(X):
        return GP._gallagher_parts(f_id, instance, np.atleast_2d(X))[0]

    def h_cli(X):
        W = peaks(X)
        return w0 - W[:, 0] + W[:, local].max(axis=1)

    def h_exact(X):
        W = peaks(X)
        return w0 - 2 * W[:, 0] + np.maximum(W[:, 0], W[:, local].max(axis=1))

    def h_orc(X):
        return w0 - peaks(X)[:, 0]

    def h_conf(X):
        return peaks(X)[:, local].max(axis=1)

    def basin(X):
        return peaks(X).argmax(axis=1) == 0

    helpers = dict(h_cli=h_cli, h_exact=h_exact, h_orc=h_orc, h_conf=h_conf)
    x_h = P["y"][0] if name in ("h_orc", "h_exact") else None
    return helpers[name], x_h, basin
