# Range-Certified Dot-Product Tiling for Small-Field GPU Proofs

**Lee Si Yuan**<br>
Independent Researcher<br>
`siyuanlee.cs@gmail.com`

## Abstract

Small-field proofs of integer inference must ensure that a field equation cannot hide a nonzero integer residual through modular wraparound. Sable gives gadget-specific range bounds for this purpose. We study a complementary kernel-planning extension: use verifier-enforced, output-row-specific operand intervals to calculate each dot-product tile's largest safe reduction length, subject to both the field residual bound and signed-int32 accumulation. The planner uses interval products and checks the entire residual interval, including output values. A Lean 4 development proves that these endpoint checks imply integer equality whenever the field relation holds. In a seeded synthetic workload of 512 output rows and 100,000 reduction terms, per-row bounds reduce the counted row-chunk total from 1,024 to 695 (32.1%); 329 rows need no split. This is a range-planning result, not a GPU speedup: the channel intervals are synthetic, must be enforced by the real circuit, and no GPU or proof implementation was available.

## Source work and selection audit

This extension is motivated by Tomer Solberg's *Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models*, arXiv:2611.14083v1. Sable defines a deterministic integer transformer and a wrap-free constraint system over the 31-bit KoalaBear prime, whose value it gives as `2^31 - 2^24 + 1`. It emphasizes that every gadget needs explicit bounds, and reports a bound below `2^29` for a 2^15-term linear-layer accumulator. The paper also describes how a modular relation can be unsound if its integer residual is not bounded below the field modulus.

The identifier does not appear in prior paper artifacts or prior paper-specific commits in this repository; its first repository appearance is the 2026-11-11 automated daily feed. The feed contains metadata and an abstract. The full arXiv v1 text was consulted for this extension. This selection audit is limited to the repository history and available session context; it cannot establish that no one accessed the paper outside them.

## Method

Let each activation `x_i` lie in `[xlo_i, xhi_i]`, and each weight `w_i` lie in `[wlo_i, whi_i]`. Interval multiplication gives

```text
p_i_lo = min(xlo_i*wlo_i, xlo_i*whi_i, xhi_i*wlo_i, xhi_i*whi_i)
p_i_hi = max(xlo_i*wlo_i, xlo_i*whi_i, xhi_i*wlo_i, xhi_i*whi_i).
```

For a tile of `n` terms, sum the term intervals (and any bias interval) to obtain `[sumLo, sumHi]`. Separately bound the committed output `z` by `[outLo, outHi]`. The integer residual `r = sum - z` is enclosed by

```text
[sumLo - outHi, sumHi - outLo].
```

A sufficient no-wrap certificate is the pair of strict checks

```text
-p < sumLo - outHi
sumHi - outLo < p.
```

If the circuit also establishes `sum - z = p*k` for an integer `k`, these checks force `k = 0`, hence `sum = z` over the integers. The Lean theorem `interval_field_equality` proves this implication. A separate hardware check constrains the raw accumulator interval to signed int32. The current planner accepts repeated operand intervals per row; the interval equations themselves also apply to non-identical term intervals.

The planner selects the largest tile length that passes both the strict field-residual bound and the signed-int32 bound. To make row-specific intervals usable in a real proof, the circuit must enforce them (for example, through range lookups or public quantization metadata bound to the committed weights). Merely measuring an observed maximum and treating it as a proof precondition would be unsound.

## CPU reference experiment

The standard-library experiment fixes seed `20261112`, uses the KoalaBear prime, signed-int32 accumulator bounds, activation interval `[-127,127]`, and 512 synthetic output rows. Each row receives a declared symmetric weight interval `[-m,m]`, with `m` drawn uniformly from integers 8 through 127. The full reduction length is 100,000. Baseline planning applies the global `[-127,127]` weight range to every row; the interval-aware planner derives each row's safe tile length. It counts the row chunks implied by these plans, without modeling GPU occupancy, memory traffic, or proof-system cost.

| Check / metric | Result |
|---|---:|
| Global safe chunk length | 66,052 terms |
| Per-row safe chunk length, min / median / max | 66,052 / 100,000 / 100,000 |
| Rows that safely fit in one 100,000-term tile | 329 / 512 |
| Baseline row-chunk count | 1,024 |
| Interval-aware row-chunk count | 695 |
| Row-chunk count reduction | 32.1% |
| Exact strict-boundary test | next term rejected |
| Concrete deterministic dot products checked | 4,096 terms |

All assertions pass. The synthetic result shows that tighter enforced ranges can reduce the number of chunks required by this planner. It does not show fewer GPU instructions or a faster proof: variable row lengths can introduce masks, divergence, padding, or launch complications, and this experiment does not model those costs.

## Formal verification

`papers/formal/RangeCertifiedDotClaims.lean` proves three statements over integers: a bounded multiple of `p` must be zero; endpoint bounds imply every permitted sum/output residual lies strictly within `(-p,p)`; and combining those facts with a modular equality recovers integer equality. The proof does not establish that an implementation's circuit correctly enforces the intervals, that the interval metadata matches deployed weights, or that GPU accumulation follows the modeled signed-int32 operation. Those remain obligations for a complete implementation.

## Limitations and next experiment

No NVIDIA GPU or CUDA toolchain is available here. The experiment is synthetic CPU range analysis: it does not run a GPU kernel, Sable prover, transformer, or profiler. It does not establish the declared per-row weight intervals for any real checkpoint. The next useful study would bind quantized checkpoint metadata to circuit range constraints, generate row-specific tiles in a real proving kernel, and compare proof size, prover time, GPU occupancy, and output equality against a uniform-bound baseline on the same models and GPU. Until then, the result is a testable planning hypothesis.

## Reproduction

From the repository root, run:

```sh
python3 papers/range_certified_dot_products/experiment.py
```

The script writes `results.json` and asserts interval corner cases, the strict modulus boundary, signed-int32 safety, and deterministic concrete sums. Check the Lean proofs with:

```sh
cd papers/formal
lake build Mathlib.Tactic.Linarith && lake env lean RangeCertifiedDotClaims.lean
```

## References

- Tomer Solberg. 2026. *Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models*. arXiv:2611.14083v1. <https://arxiv.org/abs/2611.14083v1>.
