# Second-Moment Certificates for Aggregation-Induced GPU Load Blind Spots

**Lee Si Yuan**

Independent Researcher

`siyuanlee.cs@gmail.com`

## Abstract

Analytical GPU performance models need to distinguish fixed-latency stalls from queueing stalls and to account for demand on individual hardware units. ForeSight addresses the latter by recovering per-unit counters with UnitSight. We add a lightweight diagnostic to that per-unit demand vector: for nonnegative loads $x_i$, the ratio $C=\sum_i x_i^2 / \sum_i x_i$ is a certified lower bound on the maximum unit load. Unlike the mean $\sum_i x_i/n$, this certificate increases with concentration and is exact when all demand is on one unit. A Lean 4 proof checks the two-unit natural-number case and an equal-aggregate counterexample. A seeded CPU experiment over 10,000 synthetic eight-unit log-normal workloads observed no bound violations; the mean ratio of actual maximum to certificate was 1.56, compared with 2.90 for the mean-load baseline. The experiment establishes neither GPU accuracy nor queueing-delay prediction. The result is an exploratory diagnostic for per-unit profiling, not a replacement for ForeSight's schedule replay or queueing model.

## 1. Motivation and scope

ForeSight observes that queueing stalls vary with pressure on each hardware unit, and UnitSight recovers per-unit counters that ordinary profiler outputs aggregate. The source paper reports an 8.3% kernel-time MAPE across configurations of nine NVIDIA GPUs and 2.0% end-to-end MAPE for its evaluated vLLM inference workload [Wang et al., 2026]. Here we ask a narrower question: given nonnegative per-unit demand, can a scalar concentration statistic provide a cheap, mathematically safe diagnostic of load imbalance?

This note studies the maximum normalized demand $M=\max_i x_i$ as a proxy for a bottlenecking unit. It does not model service rates, inter-unit routing, CTA scheduling, temporal burstiness, or queueing disciplines. It does not reinterpret the source paper's reported results. The extension is a lower-bound certificate on maximum demand, not a runtime predictor.

The source paper's identifier was not found in prior paper artifacts or paper-specific commits before the 2026-11-11 daily feed entry. The same entry appears in `latest` as the feed's alias. This audit covers the repository history and session context only; it cannot prove that the work was never accessed elsewhere. The paper's arXiv v1 text was consulted for this study.

## 2. Second-moment lower bound

For $n$ units with loads $x_i\ge 0$ and nonzero total demand $S=\sum_i x_i$, define

$$C=\frac{\sum_i x_i^2}{S}, \qquad \mu=\frac{S}{n}, \qquad M=\max_i x_i.$$

Since each $x_i\le M$, multiplication by nonnegative $x_i$ gives $x_i^2\le Mx_i$. Summing yields $\sum_i x_i^2\le MS$, hence

$$\mu \le C \le M.$$

The first inequality follows from $n\sum_i x_i^2\ge (\sum_i x_i)^2$ (Cauchy-Schwarz). Thus $C$ is at least the average load and at most the maximum. For equal loads, $C=\mu=M$; for a single loaded unit, $C=M=S$. In contrast, the aggregate mean alone is identical for load vectors $(10,0)$ and $(5,5)$, while their maximum loads are 10 and 5 respectively. The statistic is a load-domain certificate; it does not imply the same inequality for queueing delay or execution time.

In a profiler pipeline that exposes per-unit counts, this value can be reported beside the aggregate mean and maximum to summarize concentration without retaining the full vector. Its direct interpretation requires counts to be comparable across units and nonnegative. If units have different capacities, normalize loads by a defensible service capacity before computing the statistic; the resulting bound then applies to normalized utilization only.

## 3. CPU experiment

The standard-library script generates 10,000 independent eight-unit load vectors with Python's `random.lognormvariate(0,1)` and seed `20261113`. For each vector, it computes the mean lower bound $\mu$, second-moment certificate $C$, and actual synthetic maximum $M$. The reported ratio is $M/$bound; lower ratios indicate a tighter lower bound. The experiment checks the certificate numerically with a tolerance of $10^{-12}$. No GPU kernel, profiler, or queueing simulator is run.

| Quantity | Mean $M/$bound | Median | 95th percentile |
|---|---:|---:|---:|
| Aggregate mean $\mu$ | 2.898 | 2.691 | 4.786 |
| Second-moment certificate $C$ | 1.561 | 1.569 | 1.780 |

Observed numerical certificate violations: **0 / 10,000**. This is a reproducibility check of the arithmetic implementation, not evidence of hardware measurement accuracy. The paired vectors $(10,0)$ and $(5,5)$ both have mean 5, but the certificate equals their respective maxima (10 and 5), illustrating how the second moment distinguishes concentrations hidden by the sum.

## 4. Formal scope, limitations, and next step

`papers/formal/AggregationGapClaims.lean` compiles with Lean 4.35.0. It proves (a) the two natural-number load vectors above have equal sums and different maxima, and (b) for two natural-number loads $x,y$, $x^2+y^2\le\max(x,y)(x+y)$. The general finite-vector inequality in Section 2 follows from the termwise argument given there, but this repository's Lean file does not formalize arbitrary vector lengths or real-valued loads. The proof says nothing about GPU performance, counters, or queueing.

Key limitations are the synthetic i.i.d. load distribution, equal unit capacities, lack of service-time variation, and use of maximum demand rather than measured runtime. A meaningful hardware study would compute the statistic from validated per-SM, per-L2-slice, or per-channel counters and test whether it predicts held-out imbalance or improves an existing model's calibration. Such a study requires a supported NVIDIA GPU and must compare predictions with measured kernel times. This environment has no `nvidia-smi` executable or available GPU device; accordingly, no hardware result is claimed.

## Reproducibility

From this directory run `python3 experiment.py`; the script prints and writes `results.json`. From `papers/formal`, run `~/.elan/bin/lake env lean AggregationGapClaims.lean`. The seeded experiment uses only the Python standard library.

## References

Wang, Ziwen, Qi Luo, Kunlin Li, Changxi Liu, Dongsheng Wang, and Yun Chen. 2026. *Exploiting What NVIDIA Profilers Hide: Fast and Accurate Analytical GPU Performance Modeling*. arXiv:2611.13762v1. https://arxiv.org/abs/2611.13762v1.
