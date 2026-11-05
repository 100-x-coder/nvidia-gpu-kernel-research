#!/usr/bin/env python3
"""Synthetic benchmark study of a paired, noise-aware kernel promotion gate.

This is a CPU-only simulation, not a GPU benchmark. It uses a fixed-sample
one-sided confidence bound on paired log-latency differences. All assumptions
are explicit in the output JSON.
"""
import json
import math
import random
from pathlib import Path

SEED = 20261105
TRIALS = 30_000
REPEATS = (8, 16, 32)
TRUE_SPEEDUPS = (0.0, 0.005, 0.01, 0.02, 0.05)
MIN_GAIN = 0.01
Z_ONE_SIDED_95 = 1.6448536269514722
COMMON_DRIFT_SD = 0.05
INDEPENDENT_JITTER_SD = 0.02


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def simulate():
    rng = random.Random(SEED)
    rows = []
    threshold = -math.log1p(MIN_GAIN)
    for n in REPEATS:
        for speedup in TRUE_SPEEDUPS:
            delta = -math.log1p(speedup)
            accepted = 0
            unpaired_accepted = 0
            paired_spread, base_spread, cand_spread = [], [], []
            for _ in range(TRIALS):
                paired = []
                base, cand = [], []
                for _ in range(n):
                    drift = rng.gauss(0.0, COMMON_DRIFT_SD)
                    e0 = rng.gauss(0.0, INDEPENDENT_JITTER_SD)
                    e1 = rng.gauss(0.0, INDEPENDENT_JITTER_SD)
                    base.append(drift + e0)
                    cand.append(drift + delta + e1)
                    paired.append((drift + delta + e1) - (drift + e0))
                m, sd = mean_sd(paired)
                if m + Z_ONE_SIDED_95 * sd / math.sqrt(n) < threshold:
                    accepted += 1
                bm, bsd = mean_sd(base)
                cm, csd = mean_sd(cand)
                unpaired_se = math.sqrt((bsd * bsd + csd * csd) / n)
                if cm - bm + Z_ONE_SIDED_95 * unpaired_se < threshold:
                    unpaired_accepted += 1
                if speedup == 0.0:
                    paired_spread.append(sd)
                    base_spread.append(mean_sd(base)[1])
                    cand_spread.append(mean_sd(cand)[1])
            row = {
                "repeats": n,
                "true_speedup": speedup,
                "promotion_probability": accepted / TRIALS,
                "unpaired_promotion_probability": unpaired_accepted / TRIALS,
                "minimum_practical_gain": MIN_GAIN,
                "one_sided_confidence": 0.95,
            }
            if speedup == 0.0:
                row["null_mean_sample_sd_log_latency"] = {
                    "paired_difference": sum(paired_spread) / len(paired_spread),
                    "unpaired_baseline": sum(base_spread) / len(base_spread),
                    "unpaired_candidate": sum(cand_spread) / len(cand_spread),
                }
            rows.append(row)
    return {
        "study": "synthetic_paired_promotion_gate",
        "seed": SEED,
        "trials_per_cell": TRIALS,
        "model": {
            "log_latency_common_drift_sd": COMMON_DRIFT_SD,
            "log_latency_independent_jitter_sd": INDEPENDENT_JITTER_SD,
            "independent_jitter_distribution": "Gaussian",
            "paired_order": "same drift draw for baseline and candidate in each pair",
            "gate": "promote iff mean paired log-latency difference + z_0.95 * estimated_SE < -log(1 + minimum_practical_gain)",
            "minimum_practical_gain": MIN_GAIN,
            "fixed_sample_sizes": list(REPEATS),
            "z_value": Z_ONE_SIDED_95,
        },
        "results": rows,
        "scope": "Synthetic CPU simulation only; no GPU timings or claims about real benchmark noise.",
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(simulate(), indent=2) + "\n")
    print(out)
