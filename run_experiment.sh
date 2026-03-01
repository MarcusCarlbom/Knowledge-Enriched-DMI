#!/bin/bash

# Run all defense experiments and save results to individual files
# Skips runs where result file already exists
# Each run tests one defense at a time with others disabled (0 = disabled)
# Results saved to results/ directory

mkdir -p results

BASE="python recovery.py --model VGG16 --improved_flag --dist_flag"

run_experiment() {
    local filename=$1
    shift
    if [ -f "results/${filename}" ]; then
        echo "Skipping ${filename} (already done)"
    else
        echo "Running ${filename}..."
        $BASE "$@" | tee results/${filename}
    fi
}

# --- Baseline (no defense) ---
run_experiment "baseline.txt"

# --- Noise only ---
for noise in 0.01 0.02 0.03 0.05; do
    run_experiment "noise_${noise}.txt" --noise_std $noise
done

# --- Top-k only ---
for k in 1 5 10 50; do
    run_experiment "topk_${k}.txt" --top_k $k
done

# --- Truncation only ---
for dec in 2 3; do
    run_experiment "truncate_${dec}.txt" --truncate_decimals $dec
done

# --- Combined: noise + top-k ---
for noise in 0.01 0.03; do
    for k in 5 10; do
        run_experiment "noise_${noise}_topk_${k}.txt" --noise_std $noise --top_k $k
    done
done

# --- Combined: noise + truncation ---
for noise in 0.01 0.03; do
    for dec in 2 3; do
        run_experiment "noise_${noise}_truncate_${dec}.txt" --noise_std $noise --truncate_decimals $dec
    done
done

# --- Combined: top-k + truncation ---
for k in 5 10; do
    for dec in 2 3; do
        run_experiment "topk_${k}_truncate_${dec}.txt" --top_k $k --truncate_decimals $dec
    done
done

# --- All three combined ---
run_experiment "noise_0.02_topk_10_truncate_2.txt" --noise_std 0.02 --top_k 10 --truncate_decimals 2
run_experiment "noise_0.03_topk_5_truncate_3.txt" --noise_std 0.03 --top_k 5 --truncate_decimals 3

echo "All experiments done. Results saved to results/"