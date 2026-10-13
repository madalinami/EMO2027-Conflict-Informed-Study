#!/bin/bash
# All the runs of the paper. NSGA-II, SMS-EMOA and MOEA/D, D = 2, 5, 10, a budget of 1500 D
# evaluations and 15 runs (repetitions 1..15) everywhere. Each campaign writes one folder under
# results/: one .npz file per run and one <optimiser>_results.csv with a row per run (final error
# "precision", AOCC and the evaluations to reach 1e-1, 1e-3, 1e-5 and 1e-8). A stopped campaign
# finishes when the same command runs again.
#
# Run from the folder of this script. Set WORKERS to the number of cores to use.
set -u
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
WORKERS=${WORKERS:-8}
RUN="$PY src/benchmark_multiobjectivisation_bbob.py --workers $WORKERS"
REPS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

for INST in 1 2; do
    # Generic helpers h0..h7 on the 24 BBOB functions
    $RUN --instances $INST --reps $REPS --out results/bbob_generic_inst$INST

    # Conflict-informed helpers on BBOB f21 and f22 (Gallagher functions)
    $RUN --fids 21 22 --instances $INST --helpers h_cli h_exact h_orc h_conf --reps $REPS \
        --out results/bbob_f21_f22_inst$INST

    # Conflict-informed helpers on the trap lattice of f15 and f24, and the control h_mis
    $RUN --fids 15 24 --instances $INST --helpers h_orc_L h_cli_L h_exact_L h_conf_L h_mis_L --reps $REPS \
        --out results/bbob_lattice_inst$INST

    # Optimum-centred helpers on the 11 multimodal BBOB functions
    $RUN --fids 3 4 7 15 17 18 19 20 21 22 24 --instances $INST --reps $REPS \
        --helpers g_sph g_sph_k0.5 g_sph_k2 g_sph_k8 g_ell_k0.5 g_ell_k2 g_ell_k8 g_ell_k2_rot \
                  g_sph_rnd g_sph_k2_rnd g_ell_k2_rnd g_ell_k2_rot_rnd \
        --out results/bbob_optimum_centred_inst$INST
done

# Gallagher functions 1..10: h0, h5 and the conflict-informed helpers
$RUN --suite gallagher --reps $REPS --out results/gallagher

# Noisy h_cli: the 25 main error settings
$RUN --suite gallagher --helpers h_cli --reps $REPS --out results/gallagher_noise \
    --noise rho=0.2 rho=0.4 rho=0.6 rho=0.8 rho=1 mu=0.2 mu=0.4 mu=0.6 mu=0.8 mu=1 \
            w=0.2 w=0.4 w=0.6 w=0.8 w=1 alpha=0.25 alpha=0.5 alpha=1 alpha=2 alpha=3 \
            rot=0.17 rot=0.52 rot=1.05 rot=1.57 rot=3.14

# Noisy h_cli: the larger errors and the shifted valley x0
$RUN --suite gallagher --helpers h_cli --reps $REPS --out results/gallagher_noise_larger \
    --noise x0=0.25 x0=0.5 x0=1 x0=2 x0=4 x0=8 mu=1.5 mu=2 mu=3 mu=5 w=1.5 w=2 w=3 alpha=4 alpha=6
