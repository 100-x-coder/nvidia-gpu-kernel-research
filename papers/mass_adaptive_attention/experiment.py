#!/usr/bin/env python3
"""CPU reference study of query-conditioned history promotion for mixed-D attention."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

D = 32
D_LOW = 8
TOKENS = 64
QUERIES = 24
SEEDS = 40
PROTECTED = (4, 8, 16)


def dot(a: list[float], b: list[float], width: int = D) -> float:
    return sum(a[i] * b[i] for i in range(width))


def softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(x - peak) for x in scores]
    z = sum(exps)
    return [x / z for x in exps]


def norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def attention(q: list[float], keys: list[list[float]], values: list[list[float]]) -> list[float]:
    probs = softmax([dot(q, k) / math.sqrt(D) for k in keys])
    return [sum(probs[t] * values[t][j] for t in range(len(keys))) for j in range(D)]


def relative_error(actual: list[float], target: list[float]) -> float:
    denom = max(norm(target), 1e-12)
    return norm([a - b for a, b in zip(actual, target)]) / denom


def draw(rng: random.Random, sigma: float) -> list[float]:
    return [rng.gauss(0.0, sigma) for _ in range(D)]


def make_case(seed: int, regime: str):
    rng = random.Random(seed)
    keys = [draw(rng, 1.0) for _ in range(TOKENS)]
    values = [draw(rng, 1.0) for _ in range(TOKENS)]
    queries = [draw(rng, 1.0) for _ in range(QUERIES)]
    if regime == "diffuse":
        # Low-rank heads share a stable subspace; this is a controlled construction.
        for t in range(TOKENS):
            for j in range(D_LOW, D):
                keys[t][j] *= 0.15
                values[t][j] *= 0.25
        for q in queries:
            for j in range(D_LOW, D):
                q[j] *= 0.15
    elif regime == "old_spike":
        # Each query has its own distinct high-score token in the old prefix.
        for qi, q in enumerate(queries):
            t = qi
            keys[t] = [0.0] * D
            for j in range(D):
                keys[t][j] = q[j] * 2.7
            values[t] = [rng.gauss(0.0, 2.0) for _ in range(D)]
    elif regime != "isotropic":
        raise ValueError(regime)
    return queries, keys, values


def case_errors(seed: int, regime: str) -> dict[str, dict[str, float]]:
    queries, keys, values = make_case(seed, regime)
    low_keys = [[x if j < D_LOW else 0.0 for j, x in enumerate(k)] for k in keys]
    low_values = [[x if j < D_LOW else 0.0 for j, x in enumerate(v)] for v in values]
    errors = {name: {str(r): 0.0 for r in PROTECTED} for name in ("mass", "recency", "random", "oracle")}
    errors["all_low"] = {"0": 0.0}
    errors["full"] = {"0": 0.0}
    for qi, q in enumerate(queries):
        target = attention(q, keys, values)
        low_scores = [dot(q, k) / math.sqrt(D) for k in low_keys]
        low_probs = softmax(low_scores)
        errors["all_low"]["0"] += relative_error(attention(q, low_keys, low_values), target)
        errors["full"]["0"] += relative_error(target, target)
        full_probs = softmax([dot(q, k) / math.sqrt(D) for k in keys])
        for r in PROTECTED:
            candidates = {
                "mass": sorted(range(TOKENS), key=lambda t: (-low_probs[t], t))[:r],
                "recency": list(range(TOKENS - r, TOKENS)),
                "random": random.Random(seed * 1009 + qi * 31 + r).sample(range(TOKENS), r),
                "oracle": sorted(range(TOKENS), key=lambda t: (-full_probs[t], t))[:r],
            }
            for name, ids in candidates.items():
                chosen = set(ids)
                mixed_k = [keys[t] if t in chosen else low_keys[t] for t in range(TOKENS)]
                mixed_v = [values[t] if t in chosen else low_values[t] for t in range(TOKENS)]
                out = attention(q, mixed_k, mixed_v)
                errors[name][str(r)] += relative_error(out, target)
    count = len(queries)
    for method in errors.values():
        for k in method:
            method[k] /= count
    return errors


def main() -> None:
    results = {
        "experiment": "query_conditioned_mixed_dimension_reference",
        "seed_start": 20261108,
        "seeds_per_regime": SEEDS,
        "queries_per_seed": QUERIES,
        "tokens": TOKENS,
        "full_dimension": D,
        "low_dimension": D_LOW,
        "protected_tokens": list(PROTECTED),
        "metrics": "mean per-query relative L2 attention-output error versus full-dimensional attention",
        "workload_note": "controlled synthetic vectors; not NanoKV codecs, model traces, or video benchmarks",
        "results": {},
    }
    for regime in ("diffuse", "isotropic", "old_spike"):
        aggregate: dict[str, dict[str, float]] = {}
        for seed in range(20261108, 20261108 + SEEDS):
            one = case_errors(seed, regime)
            for method, values in one.items():
                dest = aggregate.setdefault(method, {k: 0.0 for k in values})
                for k, v in values.items():
                    dest[k] += v / SEEDS
        results["results"][regime] = aggregate
    Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("PASS: deterministic reference computation completed for 3 regimes x 40 seeds x 24 queries")


if __name__ == "__main__":
    main()
