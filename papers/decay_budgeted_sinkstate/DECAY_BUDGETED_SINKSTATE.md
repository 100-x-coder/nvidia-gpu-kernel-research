# Decay-Aware Error Budgets for Recurrent-State SVD

**Lee Si Yuan**

Independent Researcher

[siyuanlee.cs@gmail.com](mailto:siyuanlee.cs@gmail.com)

## Abstract

SinkState stores linear-attention recurrent states in adaptive low-rank form and truncates components using a relative singular-value threshold. A relative threshold does not directly bound state or readout error accumulated over repeated updates. This note adds a decay-aware error budget for the scalar-decay recurrence. If the current Frobenius state-error certificate is $b_{t-1}$, the decay is $\alpha_t\in[0,1]$, and the exact low-rank truncation discards Frobenius norm at most $\delta_t$, then $b_t=\alpha_t b_{t-1}+\delta_t$ is a certificate. To preserve a fixed budget $B$, choose rank so that $\delta_t\le B-\alpha_t b_{t-1}$. It follows that every readout error is at most $B\|q_t\|_2$. A reproducible exact-rational CPU experiment checks 5,120 steps on synthetic diagonal states with zero bound violations; the relative-threshold baseline has a higher mean trial-wise maximum coordinate-readout error and lower retained rank. These results test the inequality on a restricted surrogate. They do not establish model quality, GPU speed, or that SinkState's local/frozen update meets the exact-truncation premise.

## Source and prior-access audit

The source is Haolin Jiang et al., “SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving,” arXiv:2611.14246v1, submitted November 10, 2026 ([paper](https://arxiv.org/abs/2611.14246v1)). It first appears in `data/daily/2026-11-11.md` and is carried into the November 12 snapshot. Before this study, its identifier and title were absent from the paper index and paper-specific Git history. The daily feed held title/abstract metadata; the full v1 paper was consulted for this extension. This audit covers repository history and available session context, not external access logs.

## Idea and bound

Let $H_t=\alpha_t H_{t-1}+u_tv_t^\top$ and suppose the compressed recurrence applies the same scalar decay and exact rank truncation to its current approximation $\widehat H_{t-1}$. Before truncation, the discrepancy from the exact state is $\alpha_t(H_{t-1}-\widehat H_{t-1})$, because the same rank-one update is added to both. If truncation adds a residual $R_t$ with $\|R_t\|_F\le\delta_t$, the triangle inequality gives

\[
\|H_t-\widehat H_t\|_F\le \alpha_t\|H_{t-1}-\widehat H_{t-1}\|_F+\delta_t.
\]

Maintaining a certified upper bound $b_t=\alpha_t b_{t-1}+\delta_t$ yields a per-state invariant. For a fixed target $B$, pick the smallest rank whose discarded singular-value tail satisfies $\sum_{i>r}\sigma_i^2\le (B-\alpha_t b_{t-1})^2$. Then $b_t\le B$. Since $\|Aq\|_2\le\|A\|_F\|q\|_2$, the readout discrepancy is at most $B\|q_t\|_2$.

This construction assumes scalar decay, an exact spectrum (or a certified upper bound on its tail), the same update in the exact and compressed recurrences, and a truncation residual measured against the pre-truncation approximation. `papers/formal/DecayBudgetClaims.lean` proves the scalar one-step and sequence invariant with Lean 4.35.0 / Mathlib 4.35.0. It does not formalize matrix norms, the SVD residual formula, or an implementation. SinkState uses local top-subspace updates and frozen factors; the inequality does not certify that implementation without additional bounds for its local update and frozen tail.

## Exact CPU experiment

Run `python3 experiment.py`. It uses Python's standard library and `fractions.Fraction`: 32 seeded trials, 160 steps per trial, width 16, scalar decays $3/4$ or $9/10$, and synthetic diagonal rank-one writes. In this diagonal case, singular values are absolute diagonal entries and exact truncation is sorting, so all state updates and norm comparisons are rational. The baseline drops components below one tenth of the largest singular value. The proposed rule uses $B=1/2$ and drops the largest possible tail that satisfies the per-step budget.

| Measure | Relative threshold | Decay-aware budget |
|---|---:|---:|
| Mean retained rank | 10.79 | 12.28 |
| Mean over trials of worst coordinate-readout error | 3.428 | 0.116 |
| Maximum observed error for proposed policy | — | 0.130 |
| Steps checked / budget violations | — | 5,120 / 0 |

The higher rank is the cost of the stated error target in this workload. The error number uses basis-vector queries and is not a language-model metric. The bound applies to all queries through the Frobenius norm, with query error scaling by query norm.

## Limitations and next test

The run uses diagonal matrices, exact arithmetic, synthetic updates, and an exact global truncation oracle. It omits randomized SVD error, roundoff, local/frozen-factor drift, buffer effects, compaction work, memory layout, and GPU execution. No NVIDIA GPU or CUDA is available. Lean 4.35.0 / Mathlib 4.35.0 checks the scalar recurrence invariant, while the experiment checks the finite exact-rational surrogate. Neither validates the SVD implementation or SinkState's local/frozen update.

The next test should instrument the actual SinkState state-update path to compute a certified upper bound on local-update and frozen-tail residuals, then evaluate the resulting rank and readout-error trade-off on the same recurrent checkpoints and GPU as the original study. From `papers/formal`, reproduce the scalar proof with `~/.elan/bin/lake env lean DecayBudgetClaims.lean`.

## Reference

Haolin Jiang, Weihang Li, Zhongzhu Zhou, Jinwen Pan, Santosh Pandey, and Hang Liu. 2026. *SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving*. arXiv:2611.14246v1. https://arxiv.org/abs/2611.14246v1.
