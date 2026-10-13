# From conflict characterisation to helper function design: code

The code of all the runs in the paper. A run solves the bi-objective problem (f, h) with NSGA-II,
SMS-EMOA or MOEA/D (pymoo) and records, generation by generation, the conflict between f and h.

## Install

Python 3.12 and the pinned packages (the runs are bit-identical only with these versions):

```bash
pip install -r requirements.txt
```

## Run

All the campaigns of the paper (about 154,000 runs, several days on 8 cores):

```bash
WORKERS=8 bash run_experiments.sh
```

A single small campaign, for example h0 and h5 on BBOB f15 in 2D:

```bash
python src/benchmark_multiobjectivisation_bbob.py --fids 15 --instances 1 --dims 2 --helpers 0 5 --reps 1 2 3 --out results/test
```

Each campaign folder holds one `.npz` file per run (everything the conflict monitor recorded) and one
`<optimiser>_results.csv` with a row per run: the final error `precision` (best f minus f*), the AOCC,
and the evaluations needed to reach 1e-1, 1e-3, 1e-5 and 1e-8.

## Files

| file | content |
|---|---|
| `src/benchmark_multiobjectivisation_bbob.py` | the runs: problems, optimisers, seeds, output |
| `src/helper_functions.py` | the generic helpers h0–h7 |
| `src/gallagher.py` | the Gallagher functions, the conflict-informed helpers and their noise; the same helpers on BBOB f21/f22 |
| `src/bbob_lattice.py` | the conflict-informed helpers on the trap lattice of BBOB f15 and f24, and the control h_mis |
| `src/bbob_pointer.py` | the optimum-centred helpers |
| `src/bbob_gallagher_port.py`, `src/bbob_transforms.py` | the BBOB generator rebuilt (peaks of f21/f22, rotations, transformations) |
| `src/conflict_monitor.py`, `src/conflict_measures.py` | the conflict measured during a run (Kendall's tau_b, trade-off children, ...) |
| `noise.html`, `optimum-centred.html`, `ert.html` | interactive pages: the noisy helper; the optimum-centred helpers with their conflict maps; the runtime ECDFs and the ERT of every helper |

## Names in the code and in the paper

| code | paper |
|---|---|
| `--helpers 0 ... 7` | h0–h7 |
| `h_cli`, `h_orc`, `h_conf`, `h_exact` (Gallagher, f21, f22) | the conflict-informed helpers |
| `h_cli_L`, `h_orc_L`, `h_conf_L`, `h_exact_L`, `h_mis_L` (f15, f24) | the same helpers on the trap lattice, and h_mis |
| `g_sph`, `g_sph_k<k>`, `g_ell_k<k>`, `g_ell_k2_rot`, `..._rnd` | h_sph, h_sph,k, h_ell,k, h_rot,2, h^rnd (table in `src/bbob_pointer.py`) |
| `--noise rho mu w alpha rot x0` | missing optima ρ, position ε_μ, depth ε_w, elongation ε_α, orientation ε_θ (radians), valley shift x0 |
