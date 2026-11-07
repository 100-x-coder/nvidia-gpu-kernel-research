# Adaptive Contiguous Decay Grouping for Gated Linear Attention: A CPU Reference Study

**Author:** Lee Si Yuan  
**Group:** Independent Researcher  
**Email:** siyuanlee.cs@gmail.com  
**Base paper:** Gagik Amirkhanyan and Nuojin Cheng, *Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs*, arXiv:2611.06828v1 (2026). [Paper](https://arxiv.org/abs/2611.06828v1)

## Abstract

Systolic Linear Attention (SLA) groups per-channel forget gates to replace vector rescaling with tile-friendly scalar operations. A fixed group can incur avoidable state error when gate values vary within a tile. We study an adaptive, order-preserving refinement: partition each physical channel tile into a budgeted number of contiguous subtiles, assign each subtile its mean gate, and select boundaries by dynamic programming to minimize squared gate error. A deterministic CPU reference recurrence at 256 key channels, 32 value channels, and 48 steps evaluates four synthetic gate patterns. Two groups per 64-channel tile reduce final state error by 82.1% for a one-outlier pattern and eliminate error for the constructed bimodal pattern; reductions are smaller for clustered and smoothly varying gates. We verify a one-step Frobenius perturbation bound numerically. This is an algorithmic feasibility study only: it provides no evidence about model quality, TPU/GPU kernel speed, or the cost of boundary selection and gate dispatch.

## 1. Motivation and contribution

SLA changes gated-delta attention to better match TPU systolic matrix units through wide shared keys, tile-grouped forget gates, and matrix-based chunk solves. Grouping gates approximates the full per-channel decay and can change the recurrent state. This extension asks whether subdividing a fixed physical tile according to its gate pattern can reduce that approximation error without reordering channels.

We propose **adaptive contiguous decay grouping (ACDG)**. For each 64-channel tile and token, choose at most $K$ contiguous segments. Each segment uses the arithmetic mean of its channel gates. A dynamic program finds boundaries minimizing squared gate residuals. Channel order is preserved. Keys, values, and the write update are unchanged. This is a reference algorithm, not a production implementation.

## 2. Method and error bound

For incoming state $S\in\mathbb{R}^{d_k\times d_v}$ and channel-wise forget gate $f\in(0,1]^{d_k}$, exact decay is $D_f S$, where $D_f=\operatorname{diag}(f)$. A partition $\mathcal{P}$ assigns $\bar f_g=|g|^{-1}\sum_{i\in g}f_i$ to each contiguous segment $g$. Approximate decay applies $D_{\bar f}S$, followed by the same rank-one write as the exact recurrence.

The segmentation minimizes

$$\min_{\mathcal{P}:|\mathcal{P}|\le K}\sum_{g\in\mathcal{P}}\sum_{i\in g}(f_i-\bar f_g)^2.$$

Prefix sums give each segment cost in constant time; dynamic programming takes $O(Km^2)$ operations for a tile width $m$. This per-token metadata work is included in the CPU reference, but no accelerator execution cost is measured.

For any fixed state, decay-only perturbation obeys

$$\|(D_f-D_{\bar f})S\|_F \le \|f-\bar f\|_\infty\,\|S\|_F.$$

This follows from the induced 2-norm of a diagonal matrix. It bounds one step, not total sequence error, because later gates and writes propagate earlier perturbations. The program checks this inequality at every tested step using a floating-point tolerance. `papers/formal/SLAClaims.lean` machine-checks the scalar coordinatewise inequality used in the argument; finite-matrix aggregation and floating-point behavior are not Lean-formalized.

## 3. CPU reference experiment

The script uses Python's standard library, seed `20261107`, key width $d_k=256$, value width $d_v=32$, physical tile width 64, and 48 steps. Exact and approximate recurrences share the same initial state, keys, values, and full gate traces. We compare $K\in\{1,2,4,8,16\}$ contiguous segments per tile. Four controlled patterns test clustered noise, one low outlier per tile, a smooth within-tile ramp, and a two-level bimodal pattern. They are synthetic constructions, not model or benchmark data.

| Gate pattern | Final error, K=1 | Final error, K=2 | Reduction at K=2 |
|---|---:|---:|---:|
| Clustered | 0.02845 | 0.02710 | 4.7% |
| One outlier | 0.07712 | 0.01380 | 82.1% |
| Smooth ramp | 0.16745 | 0.10310 | 38.4% |
| Bimodal | 1.68061 | 0.00000 | 100% |

Final error is the Frobenius norm between the approximate and exact final states from one seeded trace per pattern. The error decreases with the segment budget in these tested traces. Two segments capture the constructed isolated outlier and exact two-level pattern. The smooth pattern retains nonzero error, and the clustered pattern gains little at $K=2$. The zero bimodal error is specific to a gate pattern that exactly matches the selected segmentation, up to floating-point arithmetic. These observations do not establish generalization or quality in a trained model.

## 4. Limitations and next steps

The study uses four synthetic traces and does not measure retrieval accuracy, perplexity, gradient behavior, low-precision stability, energy, memory traffic, or latency. Dynamic programming and boundary generation may cost more than saved elementwise work; per-token boundaries may also make dispatch and fusion harder. The available environment exposes an 8-core CPU but no NVIDIA GPU, CUDA compiler, or TPU runtime. No accelerator claims are tested. A follow-up should benchmark a fused implementation against exact per-channel and fixed-tile grouping on real workloads, include metadata overhead, and choose the boundary policy on held-out sequences.

## 5. Conclusion

Budgeted contiguous gate segmentation is a simple reference extension to fixed tile grouping. It can sharply reduce approximation error when gate structure aligns with a small number of contiguous segments, but gains are modest for clustered gates. The results justify an accelerator evaluation; they do not establish that the added selection work is worthwhile in a production kernel.

## Reproduction

Run `python3 experiment.py`. The script writes `results.json`, requires no third-party packages, fixes the seed, checks the local perturbation bound on every step, and asserts monotonic final-state error across the tested segment budgets. `paper.tex` uses the ACL style (`acl.sty`); `adaptive_contiguous_decay_grouping.pdf` is generated from that source with Tectonic. The Lean lemma is checked with `cd papers/formal && lake env lean SLAClaims.lean`.

## Reference

Amirkhanyan, Gagik, and Nuojin Cheng. 2026. “Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs.” arXiv:2611.06828v1. [https://arxiv.org/abs/2611.06828v1](https://arxiv.org/abs/2611.06828v1).
