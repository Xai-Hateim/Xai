#!/bin/bash

cat script.sh

output_path="results/regression/multi-experiment"

mkdir -p "${output_path}/csv"

# Sweep over datasets x dimensions x rhos. Each (dataset, dim, rho) cell is
# written to its own results file thanks to the dim-aware naming in
# `src/parse_utils.py`, so concurrent runs do not overwrite each other.
#
# Note: at large D (e.g. 50, 100) the exact methods/metrics that enumerate
# 2**D feature subsets (`brutekernelshap`, `shapr`, `shapley`,
# `shapley_corr`) become extremely expensive. They are kept enabled here as
# requested; trim the dimension list or the explainer/metric list in
# `configs/experiment_config.jsonc` if a particular cell is too slow.
for dataset in "gaussianLinear" "gaussianNonLinearAdditive" "gaussianPiecewiseConstant";
do
    echo "Running experiment for ${dataset}"
    for dim in 5 10 20 50 100;
    do
        for rho in 0.0 0.25 0.5 0.75 0.99;
        do
            python main_driver.py --mode regression --seed 7 --experiment \
                --experiment-json configs/experiment_config.jsonc \
                --rho $rho --dim $dim --dataset $dataset \
                --results-dir $output_path &
        done
        wait
    done
done
wait
