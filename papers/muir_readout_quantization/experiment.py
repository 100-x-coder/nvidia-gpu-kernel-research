#!/usr/bin/env python3
"""CPU reference for sensitivity-aware block quantization of Muir readout gains.

Controlled synthetic inputs only; this is not a trained-model or accelerator benchmark.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path

D, DV, TILE, SEEDS = 128, 32, 8, 64
BUDGET_BITS_PER_TILE = 4


def norm(xs):
    return math.sqrt(sum(x*x for x in xs))


def quantize_tile(gamma, bits):
    levels = (1 << (bits - 1)) - 1
    peak = max(map(abs, gamma), default=0.0)
    if peak == 0.0:
        return [0.0 for _ in gamma], 0.0
    step = peak / levels
    return [max(-levels, min(levels, round(x / step))) * step for x in gamma], step


def one_case(seed):
    rng = random.Random(seed)
    gamma, q, state = [], [], []
    # Tile-level dynamic ranges and row energies make sensitivity heterogeneous.
    for tile in range(D // TILE):
        gain_scale = 10.0 ** rng.uniform(-1.1, 0.8)
        energy_scale = 10.0 ** rng.uniform(-1.0, 1.0)
        for _ in range(TILE):
            gamma.append(rng.uniform(-1.0, 1.0) * gain_scale)
            q.append(rng.gauss(0, 1))
            row_scale = energy_scale * 10.0 ** rng.uniform(-0.2, 0.2)
            state.append([rng.gauss(0, row_scale / math.sqrt(DV)) for _ in range(DV)])
    # y = (gamma * q)^T S, one output coordinate per value channel.
    def read(g):
        return [sum(g[i] * q[i] * state[i][v] for i in range(D)) for v in range(DV)]
    target = read(gamma)
    groups = [list(range(i, i + TILE)) for i in range(0, D, TILE)]
    sensitivities, peaks = [], []
    for ids in groups:
        sensitivities.append(norm([q[i] for i in ids]) * math.sqrt(sum(x*x for i in ids for x in state[i])))
        peaks.append(max(abs(gamma[i]) for i in ids))
    # Bound each tile's output perturbation by ||q_tile||_2 ||S_tile||_F
    # times its maximum gain quantization error (<= peak / levels).
    def bound(bits):
        return sum(sensitivities[t] * peaks[t] / ((1 << (bits[t]-1)) - 1) for t in range(len(groups)))
    uniform_bits = [BUDGET_BITS_PER_TILE] * len(groups)
    allocated_bits = [2] * len(groups)
    # Greedy marginal reduction of the certified bound under the same bit budget.
    for _ in range((BUDGET_BITS_PER_TILE - 2) * len(groups)):
        choices = []
        for t in range(len(groups)):
            if allocated_bits[t] < 8:
                trial = allocated_bits.copy(); trial[t] += 1
                choices.append((bound(allocated_bits) - bound(trial), -t, t))
        allocated_bits[max(choices)[2]] += 1
    def decoded(bits):
        result = [0.0] * D
        for ids, b in zip(groups, bits):
            values, _ = quantize_tile([gamma[i] for i in ids], b)
            for i, x in zip(ids, values): result[i] = x
        return result
    outputs = {}
    for name, bits in (("uniform", uniform_bits), ("sensitivity_allocated", allocated_bits)):
        approx = read(decoded(bits))
        err = norm([a-b for a,b in zip(approx,target)])
        outputs[name] = {"output_l2_error": err, "bound": bound(bits), "bits_per_tile": bits}
        if err > bound(bits) + 1e-10:
            raise AssertionError(f"bound failed at seed {seed}: {name} {err} > {bound(bits)}")
    outputs["full_precision"] = {"output_l2_error": 0.0}
    return outputs


def main():
    cases = [one_case(20261109 + i) for i in range(SEEDS)]
    names = ("uniform", "sensitivity_allocated")
    summary = {}
    for name in names:
        errors = [c[name]["output_l2_error"] for c in cases]
        bounds = [c[name]["bound"] for c in cases]
        summary[name] = {"mean_output_l2_error": sum(errors)/SEEDS,
                         "median_output_l2_error": sorted(errors)[SEEDS//2],
                         "mean_certified_bound": sum(bounds)/SEEDS,
                         "bound_violations": 0}
    improvements = [1-c["sensitivity_allocated"]["output_l2_error"]/max(c["uniform"]["output_l2_error"],1e-30) for c in cases]
    result = {"experiment":"synthetic_muir_readout_gain_quantization_cpu_reference",
      "seed_start":20261109,"seeds":SEEDS,"state_rows":D,"value_width":DV,"tile_rows":TILE,
      "average_bits_per_tile":BUDGET_BITS_PER_TILE,"bits_per_tile_minimum":2,
      "source":"Muir arXiv:2611.08376v1; proposed quantization extension is not in the source paper",
      "limitations":"Synthetic controlled CPU study; no trained checkpoints, corpus, GPU, or latency measurement.",
      "summary":summary,"mean_relative_error_reduction_vs_uniform":sum(improvements)/SEEDS,
      "allocated_bits_seed_0":cases[0]["sensitivity_allocated"]["bits_per_tile"]}
    Path(__file__).with_name("results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    print("PASS: output perturbation stayed below the analytic bound in all cases")
if __name__ == "__main__": main()
