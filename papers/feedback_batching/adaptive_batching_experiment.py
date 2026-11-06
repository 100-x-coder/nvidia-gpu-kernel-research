#!/usr/bin/env python3
"""Deterministic CPU simulation of online batch sizing under workload phase drift.

This is a sensitivity experiment, not a GPU benchmark. It isolates the effect of
stale versus updated memory-footprint estimates in a manager/worker pipeline.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path

SEED = 20261106
WORKERS = 4
MEMORY_BUDGET = 96.0  # normalized memory units per worker
LAUNCH_COST = 2.0     # normalized time units per batch


def make_phase(small_fraction: float, n: int, rng: random.Random) -> list[tuple[str, float, float]]:
    """Return (class, memory, work) items; light and heavy classes are distinct."""
    light_n = round(n * small_fraction)
    items = [("light", 1.0, 1.0) for _ in range(light_n)]
    items += [("heavy", 8.0, 6.0) for _ in range(n - light_n)]
    rng.shuffle(items)
    return items


def run_policy(phases: list[list[tuple[str, float, float]]], adaptive: bool) -> dict:
    # Each worker owns a queue. Class-specific batching mirrors the base study.
    # Static chooses capacity from the initial phase's class profile and retains it.
    # Adaptive updates the observed mean footprint per class after every phase and
    # reserves 15% headroom against within-class footprint variation.
    estimated = {"light": 1.0, "heavy": 8.0}
    history = []
    total_items = 0
    total_time = 0.0
    phase_results = []
    for phase_idx, items in enumerate(phases):
        by_class = {"light": [], "heavy": []}
        for item in items:
            by_class[item[0]].append(item)
        phase_time = 0.0
        phase_count = 0
        phase_batches = 0
        for cls in ("light", "heavy"):
            group = by_class[cls]
            if not group:
                continue
            # Reserve headroom only in adaptive mode. Static remains optimistic,
            # matching a tuner calibrated against the first phase.
            observed = estimated[cls]
            guard = 0.85 if adaptive else 1.0
            batch_cap = max(1, math.floor(MEMORY_BUDGET * guard / observed))
            batches = [group[i:i + batch_cap] for i in range(0, len(group), batch_cap)]
            # Greedy list scheduling approximates concurrent workers; each batch
            # occupies one worker for launch plus its per-item service cost.
            loads = [0.0] * WORKERS
            for batch in batches:
                worker = min(range(WORKERS), key=loads.__getitem__)
                work = sum(x[2] for x in batch)
                loads[worker] += LAUNCH_COST + work
                phase_count += len(batch)
                phase_batches += 1
            phase_time = max(phase_time, max(loads))
            if adaptive:
                # Online update from the just-completed class observations.
                # Exponential smoothing prevents one phase from causing a jump.
                sample = sum(x[1] for x in group) / len(group)
                estimated[cls] = 0.5 * estimated[cls] + 0.5 * sample
        total_items += phase_count
        total_time += phase_time
        phase_results.append({
            "phase": phase_idx + 1,
            "light_fraction": round(sum(x[0] == "light" for x in items) / len(items), 3),
            "items": phase_count,
            "simulated_time": round(phase_time, 3),
            "throughput_items_per_time": round(phase_count / phase_time, 3),
            "batches": phase_batches,
            "estimates_after_phase": dict(estimated),
        })
    return {
        "total_items": total_items,
        "total_simulated_time": round(total_time, 3),
        "aggregate_throughput_items_per_time": round(total_items / total_time, 3),
        "phases": phase_results,
    }


def main() -> None:
    rng = random.Random(SEED)
    # Intentional shift from light-heavy-light to test stale calibration.
    phases = [make_phase(p, 2400, rng) for p in (0.9, 0.25, 0.9)]
    static = run_policy(phases, adaptive=False)
    adaptive = run_policy(phases, adaptive=True)
    result = {
        "experiment": "synthetic workload-shift batching sensitivity",
        "seed": SEED,
        "units": "normalized; not measured seconds or GPU throughput",
        "configuration": {"workers": WORKERS, "per_worker_memory_units": MEMORY_BUDGET,
                          "batch_launch_cost_units": LAUNCH_COST,
                          "items_per_phase": 2400, "light_work": 1.0,
                          "heavy_work": 6.0, "light_memory": 1.0,
                          "heavy_memory": 8.0},
        "policies": {"static_first_phase_calibration": static,
                     "online_smoothed_memory_estimate": adaptive},
    }
    result["adaptive_over_static_speedup"] = round(
        adaptive["aggregate_throughput_items_per_time"] /
        static["aggregate_throughput_items_per_time"], 4)
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
