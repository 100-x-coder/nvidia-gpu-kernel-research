#!/usr/bin/env python3
"""Synthetic CPU experiment for aggregation-gap certificates; no GPU timings."""
import json, random, statistics
from pathlib import Path
SEED, N_UNITS, TRIALS = 20261113, 8, 10000
rng = random.Random(SEED)
rows = []
for _ in range(TRIALS):
    loads = [rng.lognormvariate(0.0, 1.0) for _ in range(N_UNITS)]
    total = sum(loads)
    rows.append((total/N_UNITS, sum(x*x for x in loads)/total, max(loads)))
means = [z/a for a, _, z in rows]
certs = [z/b for _, b, z in rows]
result = {"experiment":"synthetic independent per-unit service loads; not GPU execution", "seed":SEED,"trials":TRIALS,"units":N_UNITS,"load_distribution":"Python random.lognormvariate(mu=0, sigma=1)", "mean_load_baseline":{"mean_actual_over_bound":statistics.mean(means),"median_actual_over_bound":statistics.median(means),"p95_actual_over_bound":sorted(means)[int(.95*TRIALS)-1]}, "second_moment_certificate":{"formula":"sum(load_i^2)/sum(load_i)","violations":sum(b>z+1e-12 for _,b,z in rows),"mean_actual_over_bound":statistics.mean(certs),"median_actual_over_bound":statistics.median(certs),"p95_actual_over_bound":sorted(certs)[int(.95*TRIALS)-1]}, "paired_examples":[{"loads":[10,0],"aggregate_mean":5,"actual_max":10,"certificate":10},{"loads":[5,5],"aggregate_mean":5,"actual_max":5,"certificate":5}], "interpretation":"The certificate is a lower bound on maximum load, not a point predictor; ratios quantify looseness."}
out=Path(__file__).with_name("results.json"); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
