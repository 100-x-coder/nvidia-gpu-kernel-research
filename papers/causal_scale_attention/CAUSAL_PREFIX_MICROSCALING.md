# Causal Prefix Microscaling for Low Precision Attention

**Author:** Lee Si Yuan (*Independent Researcher*, `siyuanlee.cs@gmail.com`)  
**Date:** November 2026  
**Base work:** Velikonivtsev et al., FLAMIA ([arXiv:2611.03344v1](https://arxiv.org/abs/2611.03344v1))  
**Evaluation:** CPU reference quantization study; no GPU kernel timing  
**Code:** [`scale_leakage_experiment.py`](./scale_leakage_experiment.py)  
**Results:** [`results.json`](./results.json)

## Abstract

Microscaled attention can inherit a causal dependency through a scale shared by past and future keys. FLAMIA analyzes this effect and limits it with block-local scales, per-key scaling in selected causal tiles, and other safeguards. This short extension studies a stricter option for scale groups that cross a causal boundary: assign each token the power-of-two scale computed from the causal prefix ending at that token. The resulting scale sequence is monotone within a block and can be run-length encoded when its exponent stays constant. A deterministic CPU stress study with E4M3-like rounding found that 32-token shared scales changed 24.9% of compared earlier reconstructions after a future outlier; prefix-derived scales changed none. Background RMSE was 0.01376 for shared scales and 0.000796 for prefix-derived scales, at 2.7 times as many scale runs. These results establish a numerical tradeoff in this reference model, not Blackwell performance or end-to-end training quality.

## 1. Motivation and scope

FLAMIA describes how a future key can affect a visible key through a shared, data-dependent quantization exponent, even when attention applies a causal mask. Its design confines or removes several such dependencies, including by using per-key scales for values on diagonal tiles. The remaining within-block effects motivate asking whether an online causal scale can preserve more precision for low-magnitude tokens before an outlier arrives.

This paper evaluates one narrow alternative: for token position `i` within a scale block, compute its scale from the maximum absolute value in the block prefix ending at `i`. No later token can alter that token's quantized reconstruction. This is a numerical reference proposal. It does not replace FLAMIA's kernel design, and it has not been implemented on NVIDIA hardware.

## 2. Prefix-Causal Microscaling

Let a block contain values `x_0, …, x_(B-1)`, and let `M` be the largest finite positive code in the E4M3 codebook. Define the prefix scale

```text
s_i = 1, if max_(0 <= j <= i) |x_j| = 0
s_i = 2^ceil(log2(max_(0 <= j <= i) |x_j| / M)), otherwise
q_i = s_i * Q_E4M3(x_i / s_i)
```

where `Q_E4M3` rounds to the nearest finite code. The scale used for token `i` depends only on tokens at positions `j <= i`. Therefore, appending a future token cannot change `q_i`. Since the prefix maximum is nondecreasing, the exponent changes only when a new prefix maximum crosses a power-of-two threshold. The scale metadata can be represented as exponent change points, subject to an implementation that can efficiently consume those runs.

The proposal trades one exponent per fixed block for a variable number of exponent runs. In the worst case, the scale changes at every token. For attention kernels, scale metadata and the algebra needed to fold per-key scales into score or value products may add work. Whether the added work can be hidden behind tensor-core operations is an open hardware question.

## 3. CPU reference experiment

The script [`scale_leakage_experiment.py`](./scale_leakage_experiment.py) builds a finite E4M3-like codebook and rounds values using power-of-two shared scales. It generates 64 seeded sequences of 4,096 values. Background values are Gaussian with standard deviation 0.03; one large outlier between 4,096 and 65,536 is inserted every 128 tokens, at a position inside common 32-token groups.

For fixed groups of 1, 2, 4, 8, 16, and 32 tokens, the script measures reconstruction error and whether adding the next token changes the reconstruction of an earlier token in the same scale group. It compares these with 32-token blocks using prefix-derived scales. Background RMSE excludes inserted outlier values (`|x| >= 1`), so it isolates damage to ordinary small activations. Scale runs count consecutive tokens with the same scale exponent.

This deliberately adversarial distribution tests dependency mechanics. Its synthetic values, approximate codebook, and scalar CPU execution do not model all FLAMIA quantization details or GPU timing.

## 4. Results

| Scale policy | Background RMSE | Earlier-value change rate | Scale runs per sequence |
|---|---:|---:|---:|
| 1 token | 0.000796 | 0.00% | 4,096 |
| 2 tokens | 0.002567 | 1.56% | 2,048 |
| 4 tokens | 0.004318 | 3.11% | 1,024 |
| 8 tokens | 0.006476 | 6.21% | 512 |
| 16 tokens | 0.009554 | 12.44% | 256 |
| 32-token shared scale | 0.013762 | 24.87% | 128 |
| 32-token block with prefix scales | 0.000796 | 0.00% | 347.6 average |

In this stress case, prefix scaling reduced background RMSE by 94.2% relative to a shared 32-token scale and removed observed future-induced reconstruction changes. It used 2.72 times as many scale runs. Fixed group sizes showed a smooth precision and dependency tradeoff as the scale-sharing span increased.

The full-precision-inclusive RMSE is dominated by the deliberately enormous outliers, so it does not distinguish the policies well; the background-only metric is more informative for this particular experiment. All generated values and measurements are available in `results.json`.

## 5. Limitations and next evaluation

The quantizer is an E4M3-like codebook implementation, not a bit-for-bit reproduction of FLAMIA's conversion path. The experiment is not an attention forward/backward test and says nothing about gradient consistency, model loss, throughput, or memory traffic on B200/B300. It also does not include attention's per-key scale folding or packed-document scheduling.

The next useful checks are: reproduce the exact FLAMIA scale conversion and range rules; test the method on real Q/K/V activation traces; verify causal output and gradient invariance through a full attention reference; then measure metadata overhead and fused-kernel latency on Blackwell. Until those checks are done, the result should be treated as a CPU numerical observation, not a GPU optimization claim.

## References

1. F. Velikonivtsev, M. Koltakov, and D. Kuznedelev, “FLAMIA: FLAsh MIcroscaling Attention for Fast and Causally Safe Training on Blackwell,” arXiv:2611.03344v1, 2026. [arXiv](https://arxiv.org/abs/2611.03344)

## Code and formal check

The [code attachment](./code_attachment.zip) includes this experiment and its output. The Lean theorem `prefix_summary_independent_of_suffix` proves the abstract prefix-only property; it does not formalize floating-point rounding or the full quantizer. See [the formal verification notes](../formal/FORMAL_VERIFICATION.md).
