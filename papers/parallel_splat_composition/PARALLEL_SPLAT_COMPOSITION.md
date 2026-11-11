# Ordered Parallel Composition for Tiled Splat Rendering

**Lee Si Yuan**<br>
Independent Researcher<br>
`siyuanlee.cs@gmail.com`

## Abstract

Ryu and Yoo reduce the long tail in 3D Gaussian Splatting (3DGS) by scheduling heavy tiles early and splitting exceptionally heavy, depth-ordered tile lists into chunks. Each chunk yields a premultiplied color and remaining transmittance, which are merged in depth order. This note studies a narrow extension: represent every chunk as an affine compositing transform and combine adjacent transforms with an order-preserving balanced tree. The real-arithmetic operation is associative, so the tree preserves the exact front-to-back result while reducing the reduction dependency depth from `k - 1` to `ceil(log2 k)` for `k` chunks. A Lean 4 proof checks associativity and the identity element. A deterministic CPU reference evaluates exact rational composition and emulated float32 reassociation over 840 chunk configurations. Tree and serial composition agree exactly in all rational cases. The largest observed float32 color error against the exact result is `1.02e-7`; the largest tree-versus-serial color delta is `1.79e-7`. This supports algebraic feasibility and a small synthetic numerical audit only. It does not establish GPU speedup, end-to-end rendering quality, or device-specific floating-point behavior.

## Source work and selection audit

The extension follows Ryu and Yoo, “Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs,” arXiv:2611.12319v1. Their paper already partitions heavy tiles into contiguous depth chunks and combines the partial colors and transmittances. It reports GPU measurements on a GB10 and RTX A6000. The source was present in the repository's 2026-11-10 automated daily feed. Before selecting it, I checked the paper index and paper-specific Git history: neither contained the identifier, a citation, nor a previous extension. The feed had captured metadata; I consulted the full paper for this work. This audit establishes only that no prior access artifact was found in this repository history and available session context.

## Method

For one pixel and one color channel, a contiguous depth interval is summarized by a pair `A = (C_A, T_A)`, where `C_A` is the interval's premultiplied color contribution assuming unit incoming transmittance, and `T_A` is its remaining transmittance. RGB composition applies the same scalar transmittance to each channel. If interval `A` is in front of interval `B`, their combined summary is

```text
A ⊙ B = (C_A + T_A C_B, T_A T_B).
```

The identity is `(0, 1)`. Expanding either parenthesization gives

```text
(A ⊙ B) ⊙ D = (C_A + T_A C_B + T_A T_B C_D, T_A T_B T_D)
A ⊙ (B ⊙ D) = (C_A + T_A C_B + T_A T_B C_D, T_A T_B T_D).
```

Thus the operation is associative over real arithmetic. Given `k` contiguous chunk summaries, a balanced, order-preserving tree can combine them in `ceil(log2 k)` dependency levels, instead of a left fold's `k - 1` levels. Both use `k - 1` pairwise compositions. The tree changes only the reduction schedule: leaves remain contiguous depth intervals and left subtrees always precede right subtrees. It does not authorize arbitrary chunk permutation.

This is a reduction-stage extension, not a new 3DGS renderer. It can be paired with the source paper's scheduling and chunking policy. Whether the shorter dependency chain matters depends on the renderer's current reduction kernel, available parallelism, synchronization, scratch traffic, and launch costs; the source's GPU results do not measure this tree.

## CPU reference experiment

The standard-library script uses seed `20261111`, 120 synthetic tiles, and chunk counts `{2, 3, 4, 8, 16, 32, 64}`. Each tile has 16–256 alpha layers. Alpha and color values are rational multiples of `1/100` in `[0,1]`. It computes chunk summaries with exact fractions, compares a left fold with a recursively balanced tree, and separately checks both against direct layer-by-layer compositing. Float32 arithmetic is emulated by rounding after multiplication and addition in the composition operation.

| Check | Result |
|---|---:|
| Exact tree equals exact serial composition | 840 / 840 configurations |
| Exact chunk composition equals direct layer scan | all cases |
| Maximum float32 tree color absolute error vs exact | `1.02e-7` |
| Maximum float32 transmittance absolute error vs exact | `7.25e-13` |
| Maximum float32 tree color delta vs serial chunk reduction | `1.79e-7` |
| Modeled pair-composition count, serial / tree | `k-1 / k-1` |
| Modeled dependency depth, serial / tree | `k-1 / ceil(log2 k)` |

The experiment is intentionally a correctness and numeric-order check, not a timing test. Since alpha and color remain in `[0,1]`, its numerical results do not cover HDR colors, negative intermediate values, compiler reassociation, fused multiply-add, or all possible float32 inputs.

## Formal check

`papers/formal/SplatCompositionClaims.lean` defines the segment pair over `ℝ` and proves associativity and both identity laws using Lean 4 and Mathlib. This verifies the exact algebra behind the tree reduction. It does not prove the implementation, interval partitioning, floating-point behavior, or a performance claim.

## Limitations and next experiment

No NVIDIA GPU, CUDA compiler/runtime, or 3DGS dependencies are available in this environment. The float32 emulation is host-side and does not reproduce device compiler or instruction behavior. There is no GPU latency, occupancy, scratch-memory, image-quality, or end-to-end renderer measurement here. A useful next experiment is to implement serial and balanced reduction variants in the same 3DGS renderer, retain identical chunk boundaries, validate outputs against the serial renderer on a real scene suite, and measure the reduction kernel and full alpha-blending stage on multiple GPU architectures. This extension should be treated as a testable kernel hypothesis until that comparison is run.

## Reproduction

From this directory, run `python3 experiment.py`. It writes `results.json` and asserts exact rational equality for every tested configuration. From `papers/formal`, run `PATH="$HOME/.elan/bin:$PATH" lake env lean SplatCompositionClaims.lean`.

## References

- Junha Ryu and Hoi-Jun Yoo. 2026. *Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs*. arXiv:2611.12319v1. <https://arxiv.org/abs/2611.12319v1>.
- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics 42(4). <https://doi.org/10.1145/3592433>.
