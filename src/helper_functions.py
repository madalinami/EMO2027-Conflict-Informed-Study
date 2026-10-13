import numpy as np

class Optimum:
    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x)
        self.y = y


class MetaData:
    def __init__(self, name, problem_id, n_variables=None):
        self.name = name
        self.problem_id = problem_id
        self.n_variables = n_variables


class HelperObjective:
    def __init__(self, function, bounds, name, problem_id, optimum_x=None, optimum_y=None):
        self.function = function
        self.bounds = bounds
        self.optimum = Optimum(x=optimum_x, y=optimum_y)
        dimension = len(bounds.lb)
        self.meta_data = MetaData(name=name, problem_id=problem_id, n_variables=dimension)
        self.is_helper = True

    def __call__(self, x):
        return self.function(x)

    def __repr__(self):
        return self.meta_data.name

# h0
def make_constant_helper(f):
    def helper(x):
        return 0.0
    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Constant",
        problem_id=0,
        optimum_x=None,
        optimum_y=None
    )

# h1
def make_sphere_helper(f):
    # Sphere helper
    # h(x) = ||x - center||^2
    lb = np.asarray(f.bounds.lb)
    ub = np.asarray(f.bounds.ub)

    center = (lb + ub) / 2

    def helper(x):
        x = np.asarray(x)
        return np.sum((x - center)**2)

    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Centered Sphere",
        problem_id=1,
        optimum_x=center,
        optimum_y=0.0
    )

# h2
def make_random_reference_sphere_helper(f, seed=None):
    # Random reference sphere
    # h(x) = ||x - c||^2
    rng = np.random.default_rng(seed)

    lb = np.asarray(f.bounds.lb)
    ub = np.asarray(f.bounds.ub)

    reference = rng.uniform(lb, ub)

    def helper(x):
        x = np.asarray(x)
        return np.sum((x - reference)**2)

    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Random Reference Sphere",
        problem_id=2,
        optimum_x=reference,
        optimum_y=0.0
    )

# h3
def make_random_reference_linear_helper(f, seed=None):
    # Random reference linear helper
    # h(x) = a^T x
    rng = np.random.default_rng(seed)
    d = f.meta_data.n_variables
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    def helper(x):
        x = np.asarray(x)
        return np.dot(direction, x)

    lb = np.asarray(f.bounds.lb)
    ub = np.asarray(f.bounds.ub)
    optimum_x = np.where(direction >= 0, lb, ub)
    optimum_y = np.dot(direction, optimum_x)

    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Random Reference Linear",
        problem_id=3,
        optimum_x=optimum_x,
        optimum_y=optimum_y
    )

# h4
def make_random_sinusoidal_helper(f, seed=None):
    # Random sinusoidal helper
    # h(x) = sum_i a_i sin(w_i x_i + phi_i)
    rng = np.random.default_rng(seed)
    d = f.meta_data.n_variables
    amplitude = rng.normal(size=d)
    frequency = rng.uniform(0.2, 2.0, size=d)
    phase = rng.uniform(0, 2 * np.pi, size=d)

    def helper(x):
        x = np.asarray(x)
        return np.sum(amplitude * np.sin(frequency * x + phase))
    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Random Sinusoid",
        problem_id=4,
        optimum_x=None,
        optimum_y=None
    )

# h5
def make_inverted_helper(f):
    # Inverted objective
    # h(x) = -f(x)
    def helper(x):
        return -f(x)
    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Inverted",
        problem_id=5,
        optimum_x=None,
        optimum_y=None
    )

# h6
def make_reciprocal_helper(f, epsilon=1e-8):
    # Reciprocal helper: h(x) = 1 / (f(x) + eps)
    def helper(x):
        val = f(x)
        return 1.0 / (val + epsilon) if val >= 0 else 1.0 / (val - epsilon)
    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name="Reciprocal",
        problem_id=6,
        optimum_x=None,
        optimum_y=None
    )

# h7
def make_smoothed_helper(f, sigma=1.0, n_samples=20, seed=None):
    # Smoothed helper
    # h(x) = 1/K sum_k f(x + sigma*z_k)
    rng = np.random.default_rng(seed)
    d = f.meta_data.n_variables
    perturbations = rng.normal(size=(n_samples, d))
    lb = np.asarray(f.bounds.lb)
    ub = np.asarray(f.bounds.ub)

    def helper(x):
        x = np.asarray(x)
        values = []
        for z in perturbations:
            y = x + sigma * z
            y = np.clip(y, lb, ub)
            values.append(f(y))
        return np.mean(values)

    return HelperObjective(
        function=helper,
        bounds=f.bounds,
        name=f"Smoothed",
        problem_id=7,
        optimum_x=None,
        optimum_y=None
    )
