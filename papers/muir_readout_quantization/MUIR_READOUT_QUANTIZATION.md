# Sensitivity-Allocated Block Quantization for Matched-Filter Readout

**Lee Si Yuan**

Independent Researcher

<siyuanlee.cs@gmail.com>

## Abstract

Muir introduces a matched-filter readout gain that scales the query before reading its recurrent state. We study a deployment extension that quantizes this gain in fixed-size channel tiles under a fixed nominal bit budget. Our greedy allocator gives the next bit to the tile with the largest reduction in a conservative output-error bound. For a CPU reference recurrence with 128 state rows, 32 value channels, 8-row tiles, and 64 controlled synthetic seeds, sensitivity allocation reduced mean output L2 error by 77.1% relative to uniform four-bit tile quantization at the same gain-code bit count. An analytic per-tile bound held in every trial and a Lean 4 proof checks its weighted-sum form. The result is a synthetic feasibility result: scale metadata is excluded from the bit budget, and we measure neither trained-model quality nor GPU latency.

## 1. Motivation and scope

Muir uses a channel-wise gain on the query to calibrate how recurrent state rows contribute to output. This paper asks a systems question about that readout: if the gain is quantized in tiles, should every tile use the same number of bits? The proposal is a sensitivity-aware bit allocator, not a change to Muir's learned gain or recurrence.

No earlier Muir research artifact or extension appears in the papers collection or paper-specific Git history. The automated daily index had already stored Muir's title and abstract; this work reads and extends its full arXiv v1 text. The source is Gangavarapu et al., [Muir, arXiv:2611.08376v1](https://arxiv.org/abs/2611.08376v1).

## 2. Method

For query coordinates $q_j$, state rows $S_j\in\mathbb{R}^{d_v}$, and learned gains $\gamma_j$, the readout is

$$y=\sum_j \gamma_j q_j S_j.$$

Partition channels into tiles $G_t$. In a tile with $b_t\ge2$ signed quantization bits, use $L_t=2^{b_t-1}-1$ positive levels and scale $a_t=\max_{j\in G_t}|\gamma_j|/L_t$. Symmetric round-to-nearest quantization gives $|\hat\gamma_j-\gamma_j|\le a_t/2$; the implementation's conservative bound uses the looser $\max|\gamma|/L_t$ and accommodates ties and floating arithmetic.

By triangle and Cauchy--Schwarz,

$$\|\hat y-y\|_2 \le \sum_t \max_{j\in G_t}|\hat\gamma_j-\gamma_j|\,\|q_{G_t}\|_2\,\|S_{G_t}\|_F
\le \sum_t \frac{\max_{j\in G_t}|\gamma_j|}{2^{b_t-1}-1}\,\|q_{G_t}\|_2\,\|S_{G_t}\|_F.$$

The reference allocator starts each tile at two bits and spends the remaining nominal gain-code budget one bit at a time. At each step it increments the tile that most reduces the bound above. It is a greedy heuristic; we do not claim global optimality. Scale metadata and allocation-computation cost are not included in the nominal bit budget.

## 3. CPU reference study

The standard-library script generates independent controlled synthetic cases. Each has 128 rows, value width 32, and 16 tiles of 8 rows. Tile gain ranges and state-row energies are drawn across log-uniform scales to induce heterogeneous sensitivity. We compare uniform four-bit quantization with adaptive tile bit widths constrained to an average of four bits per gain (512 gain-code bits per vector), excluding scale metadata, over 64 fixed seeds. We check the computed output error against the analytic bound in each run.

| Method | Mean output L2 error | Median output L2 error | Mean analytic bound | Bound violations |
|---|---:|---:|---:|---:|
| Uniform, 4 bits/gain | 2.3682 | 1.7889 | 50.5491 | 0/64 |
| Sensitivity allocated, 2–8 bits/gain | 0.4362 | 0.3554 | 16.4972 | 0/64 |

Mean measured error fell 77.1% under the adaptive allocation in these synthetic cases. This does not establish a model-quality improvement or a speedup. The bound is conservative, and this CPU script does not measure scale metadata, fused-kernel overhead, quantized arithmetic, or accelerator latency.

## 4. Formal check

`papers/formal/MuirReadoutClaims.lean` proves that if each scalar approximation error is bounded by a radius, then any nonnegative weighted signed sum is bounded by the corresponding weighted radii. It also proves monotonicity of the peak-over-levels bound when level count increases. These lemmas verify the inequalities used by the bound; they do not prove the measurements, greedy allocator optimality, quantizer execution, or hardware behavior.

## 5. Limitations and next experiment

This study uses synthetic vectors and states, not Muir checkpoints, language data, or learned readout gains. The gain-code budget omits scale values, alignment, tile descriptors, and any extra read bandwidth. There is no NVIDIA GPU, CUDA runtime, or accelerator compiler on the available host. The next test should use released model activations, include scale metadata in an actual storage format, and benchmark a fused quantize/read kernel against full-precision and uniform-quantized baselines on target hardware while measuring task quality.

## Reproducibility

Run `python3 papers/muir_readout_quantization/experiment.py` from the repository root. It uses only Python's standard library, fixes seeds 20261109–20261172, writes `results.json`, and asserts all 64 measured output errors satisfy the bound. Lean command: `cd papers/formal && lake env lean MuirReadoutClaims.lean`. LaTeX source and the compiled ACL-style PDF are supplied alongside this Markdown file.

## Reference

Tushaar Gangavarapu, Gil Tabak, and Clemens JS Schaefer. 2026. Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout. arXiv:2611.08376v1. <https://arxiv.org/abs/2611.08376v1>.
