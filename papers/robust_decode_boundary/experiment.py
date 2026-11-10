#!/usr/bin/env python3
"""Reproducible synthetic study of fixed vs. scenario-adaptive serving choices.

This is a dimensionless sensitivity experiment, not a calibrated hardware model.
Only the Python standard library is required.
"""
import itertools
import json
import math
import random
from pathlib import Path

SEED = 20261110
N = 20_000
RNG = random.Random(SEED)


def scenario():
    # Log-uniform workload and hardware ratios deliberately span broad regimes.
    kv_pressure = math.exp(RNG.uniform(math.log(0.35), math.log(3.0)))
    ffn_pressure = math.exp(RNG.uniform(math.log(0.35), math.log(3.0)))
    prefill_pressure = math.exp(RNG.uniform(math.log(0.35), math.log(3.0)))
    mem_compute_ratio = math.exp(RNG.uniform(math.log(0.4), math.log(2.5)))
    fabric = math.exp(RNG.uniform(math.log(0.08), math.log(2.0)))
    pooling_loss = RNG.uniform(0.0, 0.25)
    # Role-specific placement may give either AFD pool a better resource match.
    attn_match = math.exp(RNG.uniform(math.log(0.65), math.log(1.8)))
    ffn_match = math.exp(RNG.uniform(math.log(0.65), math.log(1.8)))
    # Scores are normalized service ceilings per unit power. Values are
    # intentionally transparent surrogates rather than device measurements.
    prefill = 1.0 / prefill_pressure
    colocated_decode = min(mem_compute_ratio / kv_pressure, 1.0 / ffn_pressure)
    pd = min(prefill, colocated_decode)
    # Splitting lets attention and FFN use independent resource budgets. It
    # also pays recurring exchange and independent-pool fragmentation costs.
    attn = mem_compute_ratio / kv_pressure * attn_match
    ffn = 1.0 / ffn_pressure * ffn_match
    exchange_eff = fabric / (fabric + 0.35)
    afd_decode = min(attn, ffn) * exchange_eff * (1.0 - pooling_loss)
    afd = min(prefill, afd_decode)
    return {"PD": pd, "AFD": afd}


def summarize():
    samples = [scenario() for _ in range(N)]
    names = ("PD", "AFD")
    total = {n: 0.0 for n in names}
    wins = {n: 0 for n in names}
    regrets = {n: [] for n in names}
    ties = 0
    for utilities in samples:
        best = max(utilities.values())
        ties += utilities["PD"] == utilities["AFD"]
        for n in names:
            total[n] += utilities[n]
            wins[n] += utilities[n] == best
            regrets[n].append(best - utilities[n])
    means = {n: total[n] / N for n in names}
    mean_regret = {n: sum(regrets[n]) / N for n in names}
    p95_regret = {n: sorted(regrets[n])[math.ceil(0.95 * N) - 1] for n in names}
    minimax = min(names, key=lambda n: max(regrets[n]))
    # Exhaustively check structural properties over all generated scenarios.
    assert all(r >= 0.0 for rs in regrets.values() for r in rs)
    assert max(regrets[minimax]) <= max(regrets["AFD" if minimax == "PD" else "PD"]) + 1e-15
    # A per-scenario oracle cannot do worse than any fixed choice.
    oracle_mean = sum(max(s.values()) for s in samples) / N
    assert oracle_mean + 1e-15 >= max(means.values())
    return {
        "study": "synthetic dimensionless sensitivity; not calibrated hardware results",
        "seed": SEED,
        "scenario_count": N,
        "sampling": "independent log-uniform resource ratios; uniform pooling loss",
        "ranges": {
            "kv_pressure": [0.35, 3.0], "ffn_pressure": [0.35, 3.0],
            "prefill_pressure": [0.35, 3.0], "memory_compute_ratio": [0.4, 2.5],
            "fabric_ratio": [0.08, 2.0], "pooling_loss": [0.0, 0.25],
            "role_specific_match_factors": [0.65, 1.8]
        },
        "mean_surrogate_utility": means,
        "fraction_best_including_ties": {n: wins[n] / N for n in names},
        "mean_regret": mean_regret,
        "p95_regret": p95_regret,
        "worst_regret": {n: max(regrets[n]) for n in names},
        "minimax_regret_fixed_choice": minimax,
        "oracle_adaptive_mean_utility": oracle_mean,
        "fixed_best_mean_utility": max(means.values()),
        "oracle_minus_fixed_mean": oracle_mean - max(means.values()),
        "tie_count": ties,
        "assertions": ["nonnegative regret", "minimax regret no worse than alternatives", "scenario oracle no worse than fixed policy"]
    }


if __name__ == "__main__":
    out = summarize()
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
