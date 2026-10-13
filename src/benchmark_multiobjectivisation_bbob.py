import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import ioh
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from bbob_lattice import LATTICE_FUNCTIONS, LATTICE_HELPERS, has_lattice, lattice_helper
from bbob_pointer import POINTER_FUNCTIONS, POINTER_HELPERS, has_pointer, pointer_helper
from conflict_monitor import ConflictMonitor
from gallagher import (BBOB_GALLAGHER_HELPERS, GALLAGHER_DEFAULTS, GALLAGHER_HELPERS,
                       NOISY_HELPERS, bbob_gallagher_helper, in_basin, make_pair, parse_noise)
from helper_functions import (
    make_constant_helper,
    make_sphere_helper,
    make_random_reference_sphere_helper,
    make_random_reference_linear_helper,
    make_random_sinusoidal_helper,
    make_inverted_helper,
    make_reciprocal_helper,
    make_smoothed_helper
)

# The versions that produced a run. Runs are bit-identical only with the same versions (uv.lock).
def _version(package):
    try:
        return version(package)
    except Exception:
        return "missing"

ENV = f"python {platform.python_version()} {platform.machine()}, " + ", ".join(
    f"{p} {_version(p)}" for p in ("numpy", "scipy", "pymoo", "ioh", "moocore"))

class ProblemWrapper(ElementwiseProblem):
    """The bi-objective problem (f, h) that the optimisers solve, one point at a time (BBOB)."""
    def __init__(self, f, h, name, d=None):
        self.name = name
        self.f = f
        self.h = h
        if d is None:
            d = len(f.bounds.lb)
        xl = np.array(f.bounds.lb, dtype=float)
        xu = np.array(f.bounds.ub, dtype=float)
        super().__init__(n_var=d, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out):
        out["F"] = np.array([self.f(x), self.h(x)], dtype=float)

class PopulationProblem(Problem):
    """(f, h) for functions that take the whole population at once (the Gallagher functions,
    about five times faster than one point at a time)."""
    def __init__(self, f, h, dim, bounds):
        self.f = f
        self.h = h
        super().__init__(n_var=dim, n_obj=2, xl=float(bounds[0]), xu=float(bounds[1]))

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack([self.f(X), self.h(X)])

def get_helpers(f, seed):
    return [
        make_constant_helper(f),
        make_sphere_helper(f),
        make_random_reference_sphere_helper(f, seed=seed),
        make_random_reference_linear_helper(f, seed=seed),
        make_random_sinusoidal_helper(f, seed=seed),
        make_inverted_helper(f),
        make_reciprocal_helper(f),
        make_smoothed_helper(f, seed=seed)
    ]

def helper_seed(f_id, instance, dim):
    """The seed of the random helpers (h2, h3, h4, h7). Repetition r of a run uses the seed helper_seed + r."""
    return 100000 * dim + 10000 * f_id + 100 * instance

def calculate_aocc(val, optimum, initial, B):
    if len(val) < B:
        val = np.pad(val, (0, B - len(val)), mode='edge')
    denom = np.abs(initial - optimum)
    if np.isclose(denom, 0):
        normalized_values = np.zeros_like(val)
    else:
        normalized_values = np.abs(val - optimum) / denom
    normalized_values = np.clip(normalized_values, 0.0, 1.0)
    return 1.0 - (np.trapezoid(normalized_values, dx=1) / max(B - 1, 1))

# pymoo's default operators, written out. The monitor records the children (after mutation).
# NSGA-II and SMS-EMOA evaluate 50 points per generation, MOEA/D (13 reference directions) 13.
# MOEA/D keeps pymoo's n_neighbors=20, more than its 13 slots, so every neighbourhood is the whole
# population. pymoo's SMS-EMOA makes 50 children per generation (not the steady-state version).
OPTIMISERS = {
    "nsgaii": NSGA2(pop_size=50, crossover=SBX(eta=15, prob=0.9)),
    'moead': MOEAD(get_reference_directions("uniform", 2, n_partitions=12), crossover=SBX(prob=1, eta=20)),
    'smsemoa': SMSEMOA(pop_size=50, crossover=SBX())
}

def generations(opt_name, budget):
    """The number of generations that spends the budget of evaluations. NSGA-II and SMS-EMOA use it
    exactly (1500 * D is a multiple of 50); MOEA/D may use up to 12 evaluations more."""
    return math.ceil(budget / OPTIMISERS[opt_name].n_offsprings)

# --- The two suites of problems ---------------------------------------------------------------
def run_filepath(job):
    opt, dim, rep = job["optimiser"], job["dim"], job["rep"]
    if job["suite"] == "bbob":
        helper = job["helper"]
        name = f"f{job['f_id']}_inst{job['instance']}_" + (f"h{helper}" if isinstance(helper, int) else helper)
    else:
        noise = job["noise"].replace("=", "").replace(",", "-")
        name = f"gal{job['f_seed']}_{job['helper']}_{noise}"
    return f"{opt}/{dim}D/{opt}_{dim}D_{name}_rep{rep}.npz"

def bbob_problem(dim, f_id, instance, helper, **_):
    """f = BBOB function f_id, instance, and the helper `helper`: a number 0..7 (the generic helpers),
    for f21 and f22 the name of a conflict-informed helper (see gallagher.bbob_gallagher_helper), on
    bbob_lattice.LATTICE_FUNCTIONS one of the helpers with a Gallagher component on every trap of f,
    or on bbob_pointer.POINTER_FUNCTIONS one of the optimum-centred helpers (a pit at x*).
    The last value returned is the region the monitor watches separately: the basin of x* for the
    f21/f22 and optimum-centred helpers, the region where the bump of x* is the highest for the
    lattice helpers."""
    h_seed = helper_seed(f_id, instance, dim)
    f = ioh.get_problem(f_id, instance, dim)
    if isinstance(helper, int):
        h = get_helpers(f, h_seed)[helper]
        h_name, h_id, x_h, region = h.meta_data.name, h.meta_data.problem_id, h.optimum.x, None
    else:
        make = (pointer_helper if helper in POINTER_HELPERS else
                lattice_helper if helper in LATTICE_HELPERS else bbob_gallagher_helper)
        h_pop, x_h, region = make(f_id, instance, dim, helper)
        h = lambda x: float(h_pop(x)[0])   # the problem gives one point at a time
        h_name = h_id = helper
    problem = ProblemWrapper(f, h, f'(f{f_id}, {h_id})', d=dim)
    columns = {"f_name": f.meta_data.name, "f_id": f.meta_data.problem_id, "f_instance": instance,
               "h_name": h_name, "h_id": h_id, "helper_seed": h_seed}
    return problem, float(f.optimum.y), f.optimum.x, x_h, h_seed, columns, region

def gallagher_problem(dim, f_seed, helper, noise, gallagher, **_):
    """f = Gallagher function number f_seed, and the helper `helper` (see gallagher.py). The monitor
    also watches the basin of x* separately, to see conflict at the traps and harmony at x*."""
    f, h, x_h = make_pair(dim, f_seed, helper, noise, gallagher)
    problem = PopulationProblem(f, h, dim, f.bounds)
    # The run seeds must differ from f_seed: f draws its global optimum first, so a run with the
    # same seed would start with x* in its initial population.
    base_seed = 100000 * dim + 100 * f_seed
    # f < w0 - (largest local weight) only inside the basin of x*: the run has left the traps
    basin_f = f.global_component.weight - max(c.weight for c in f.components[1:])
    columns = {"f_seed": f_seed, "n_components": gallagher["components"], "helper": helper, "noise": noise,
               "basin_f": basin_f}
    return problem, 0.0, f.x_opt, x_h, base_seed, columns, lambda X: in_basin(f, X)

def run_rep(job):
    """One run of one optimiser on (f, h), with the conflict monitor attached.

    Everything the monitor collected goes to one .npz file (see conflict_monitor.py); the function
    returns the row of the results table.
    """
    make_problem = bbob_problem if job["suite"] == "bbob" else gallagher_problem
    problem, f_star, x_star, x_h, base_seed, columns, region = make_problem(**job)
    run_seed = base_seed + job["rep"]
    B = generations(job["optimiser"], job["budget"])
    monitor = ConflictMonitor(B, f_star, x_star, x_h, save_points=job["save_points"], region=region)

    start = time.time()
    res = minimize(problem, OPTIMISERS[job["optimiser"]], ('n_gen', B), seed=run_seed, verbose=False, callback=monitor)
    data = monitor.result()
    if data["n_evals"] != res.algorithm.evaluator.n_eval:
        raise RuntimeError(f"the monitor saw {data['n_evals']} evaluations, pymoo made {res.algorithm.evaluator.n_eval}")

    # AOCC on the best f found so far at the end of each generation. (The old code used the best
    # point of the current population; SMS-EMOA often drops that point.) It runs over generations,
    # so compare AOCC only within one optimiser.
    bsf = data["gen_bsf_f"]
    aocc = calculate_aocc(bsf, f_star, bsf[0], B)

    raw_filepath = run_filepath(job)
    data.update(optimiser=job["optimiser"], suite=job["suite"], dim=job["dim"], rep=job["rep"], seed=run_seed,
                budget=job["budget"], aocc=aocc, env=ENV, final_X=res.pop.get("X"), final_F=res.pop.get("F"),
                **columns)
    # Write to a temporary name first, so a .npz file is always complete
    path = Path(job["out_dir"]) / raw_filepath
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    row = {
            "optimiser": job["optimiser"],
            **columns,
            "seed": run_seed,
            "dim": job["dim"],
            "budget": job["budget"],
            "rep": job["rep"],
            "aocc": aocc,
            "raw_filepath": raw_filepath,
            "n_gen": B,
            "n_evals": data["n_evals"],
            "best_f": data["bsf_f"][-1],
            "precision": data["bsf_f"][-1] - f_star,
            **{key: data[key] for key in data if key.startswith("evals_to_")},
            "first_all_nd_gen": data["first_all_nd_gen"],
            "time_s": round(time.time() - start, 1),
        }
    if "basin_f" in columns:   # Gallagher: the first evaluation inside the basin of x*, -1 if never
        hit = np.flatnonzero(data["bsf_f"] < columns["basin_f"])
        row["evals_to_basin"] = int(data["bsf_eval"][hit[0]]) if len(hit) else -1
    return row

def main(argv=None):
    ap = argparse.ArgumentParser(description="Runs the optimisers on (f, h) and measures the conflict along each "
                                             "run. Suite bbob: the BBOB functions with the helpers h0..h7. Suite "
                                             "gallagher: Gallagher functions with the conflict-informed helper.")
    ap.add_argument("--suite", default="bbob", choices=["bbob", "gallagher"])
    ap.add_argument("--optimisers", nargs="+", default=list(OPTIMISERS), choices=list(OPTIMISERS))
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--helpers", nargs="+", default=None,
                    help="bbob: 0..7 (default all), for f21/f22 also h_cli h_exact h_orc h_conf, on "
                         "bbob_lattice.LATTICE_FUNCTIONS h_orc_L h_cli_L h_exact_L h_conf_L h_mis_L, on "
                         "bbob_pointer.POINTER_FUNCTIONS the optimum-centred helpers of "
                         "bbob_pointer.POINTER_HELPERS; gallagher: names from gallagher.py (default all of "
                         "GALLAGHER_HELPERS)")
    # Repetition r uses the seed base_seed + r. We start at r = 1: for BBOB, r = 0 gives the run the seed
    # of the helpers, and the first point of the initial population is then the minimum of h2.
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(1, 16)))
    ap.add_argument("--budget-scale", type=int, default=1500, help="budget = scale * D evaluations")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=None,
                    help="output folder; pass an existing one, with the same settings, to finish it")
    ap.add_argument("--save-points", action="store_true",
                    help="also save every evaluated point with its parents (large: for small subsets only)")
    bbob = ap.add_argument_group("bbob")
    bbob.add_argument("--fids", nargs="+", type=int, default=list(range(1, 25)))
    bbob.add_argument("--instances", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    gal = ap.add_argument_group("gallagher")
    gal.add_argument("--fseeds", nargs="+", type=int, default=list(range(1, 11)), help="which Gallagher functions")
    gal.add_argument("--noise", nargs="+", default=["none"],
                     help='noise settings of the helper, e.g. none rho=0.2 mu=0.4 w=0.5 alpha=0.8 rot=0.5 "alpha=0.8,rot=0.5"')
    gal.add_argument("--components", type=int, default=GALLAGHER_DEFAULTS["components"])
    gal.add_argument("--global-alpha", type=float, default=GALLAGHER_DEFAULTS["global_alpha"])
    gal.add_argument("--local-alpha", nargs=2, type=float, default=list(GALLAGHER_DEFAULTS["local_alpha"]))
    args = ap.parse_args(argv)
    if args.suite == "bbob":
        helpers = [int(h) if h.lstrip("-").isdigit() else h for h in args.helpers] if args.helpers else list(range(0, 8))
        named = BBOB_GALLAGHER_HELPERS + POINTER_HELPERS + LATTICE_HELPERS
        if any(h not in range(8) and h not in named for h in helpers):
            sys.exit(f"unknown BBOB helpers {helpers}; choose from 0..7 and {named}")
        if any(h in BBOB_GALLAGHER_HELPERS for h in helpers) and set(args.fids) - {21, 22}:
            sys.exit("the conflict-informed helpers exist only for f21 and f22 (use --fids 21 22)")
        if any(h in POINTER_HELPERS for h in helpers) and not all(has_pointer(f) for f in args.fids):
            sys.exit(f"the optimum-centred helpers exist only on f{POINTER_FUNCTIONS}")
        if any(h in LATTICE_HELPERS for h in helpers) and not all(has_lattice(f) for f in args.fids):
            sys.exit(f"the trap-lattice helpers exist only on f{LATTICE_FUNCTIONS}")
    else:
        helpers = args.helpers or list(GALLAGHER_HELPERS)
        unknown = set(helpers) - set(GALLAGHER_HELPERS)
        if unknown:
            sys.exit(f"unknown Gallagher helpers {sorted(unknown)}; choose from {GALLAGHER_HELPERS}")
        for text in args.noise:
            parse_noise(text)   # stop now on a typo, not in the workers

    DATA_DIR = Path(args.out or f"data/data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # One folder holds runs made with one suite, budget, save_points choice and environment.
    gallagher = dict(components=args.components, global_alpha=args.global_alpha, local_alpha=args.local_alpha)
    settings = dict(suite=args.suite, budget_scale=args.budget_scale, save_points=args.save_points, env=ENV)
    if args.suite == "gallagher":
        settings["gallagher"] = gallagher
    settings_path = DATA_DIR / "settings.json"
    if settings_path.exists():
        old = json.loads(settings_path.read_text())
        if old != settings:
            sys.exit(f"{DATA_DIR} was started with other settings:\n  {old}\nnow:\n  {settings}\n"
                     f"Use the same settings, or another --out.")
    elif any(DATA_DIR.glob("*_results.csv")):
        sys.exit(f"{DATA_DIR} holds results but no settings.json; use another --out.")
    else:
        settings_path.write_text(json.dumps(settings, indent=1) + "\n")

    logging.basicConfig(
        filename=DATA_DIR / "benchmark.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # A run counts as done when it has a row in a results file. A run with a .npz but no row (the
    # grid was stopped right after it) is run again; it gives the same file.
    done = set()
    for csv in DATA_DIR.glob("*_results.csv"):
        done |= set(pd.read_csv(csv, usecols=["raw_filepath"])["raw_filepath"])

    # Order: optimiser, dimension, function, helper, (noise,) repetition.
    if args.suite == "bbob":
        problems = [dict(f_id=f_id, instance=instance, helper=h)
                    for f_id in args.fids for instance in args.instances for h in helpers]
    else:
        problems = [dict(f_seed=s, helper=h, noise=noise, gallagher=gallagher)
                    for s in args.fseeds for h in helpers
                    for noise in (args.noise if h in NOISY_HELPERS else ["none"])]
    jobs = []
    for opt_name in args.optimisers:
        for dim in args.dims:
            for p in problems:
                for rep in args.reps:
                    job = dict(suite=args.suite, optimiser=opt_name, dim=dim, rep=rep, budget=args.budget_scale * dim,
                               out_dir=DATA_DIR, save_points=args.save_points, **p)
                    if run_filepath(job) not in done:
                        jobs.append(job)

    logger.info(f'suite: {args.suite}')
    logger.info(f'optimisers: {args.optimisers}')
    logger.info(f'dimensions: {args.dims}')
    logger.info(f'helpers: {helpers}')
    logger.info(f'repetitions: {args.reps}')
    if args.suite == "bbob":
        logger.info(f'bbob ids: {args.fids}, instances: {args.instances}')
    else:
        logger.info(f'gallagher functions: {args.fseeds}, {gallagher}, noise: {args.noise}')
    logger.info(f'budget: {args.budget_scale} * D evaluations')
    logger.info(f'max workers: {args.workers}')
    logger.info(f'save every point: {args.save_points}')
    logger.info(f'environment: {ENV}')
    logger.info(f'runs to do: {len(jobs)} (already done: {len(done)})')
    print(f"{len(jobs)} runs to do, {len(done)} already done, output in {DATA_DIR}", flush=True)

    # Stop the grid with Ctrl-C (not kill): the runs already done are kept, and the same command
    # with --out finishes the rest.
    n_failed = 0
    executor = ProcessPoolExecutor(max_workers=args.workers)
    futures = {executor.submit(run_rep, job): job for job in jobs}
    try:
        for k, future in enumerate(as_completed(futures), 1):
            job = futures.pop(future)
            name = run_filepath(job).split("/")[-1][:-4]
            try:
                row = future.result()
            except BrokenProcessPool:  # a worker died (out of memory?): stop, the rest is not lost
                n_failed += len(futures) + 1
                logger.error(f"a worker process died; stopping. Run again with --out {DATA_DIR} to finish")
                break
            except Exception as e:  # one failed run must not stop the others
                n_failed += 1
                logger.exception(f"FAILED {name}: {e}")
                continue
            CSV_PATH = DATA_DIR / f"{job['optimiser']}_results.csv"
            pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
            logger.info(f"[{k}/{len(jobs)}] {name}: aocc {row['aocc']:.4f}, {row['time_s']} s")
    except KeyboardInterrupt:
        # Stop at once: end the runs in progress and drop the ones waiting
        logger.info("stopped with Ctrl-C")
        for process in executor._processes.values():
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    logger.info(f"finished, {n_failed} failed runs")
    print(f"finished, {n_failed} failed runs (see {DATA_DIR / 'benchmark.log'})", flush=True)
    return n_failed

if __name__ == "__main__":
    sys.exit(1 if main() else 0)
