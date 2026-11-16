# Certified Omitted-Band Screening for Batched Matched Filtering

**Lee Si Yuan**<br>
*Independent Researcher*<br>
[siyuanlee.cs@gmail.com](mailto:siyuanlee.cs@gmail.com)

## Abstract

Nitz et al. combine fused matched filtering with hierarchical, statistically controlled screening. This extension tests a complementary gate: certify that an omitted frequency band cannot raise a coarse correlation peak above the detection threshold. For each data–template pair, triangle inequality and Cauchy–Schwarz bound the omitted inverse-transform residual by the product of data and template tail spectral energies divided by transform length. That gives a zero-false-dismissal screening implication in exact arithmetic, conditional on correct spectra and normalization.

A seeded CPU reference experiment on an Apple M2 evaluates 512 Gaussian noise pairs and 128 template-aligned injections at transform length 128. In a defined low-band-dominant synthetic spectrum, the gate safely skips 438/512 (85.5%) noise pairs and skips none of the 128 above-threshold injections. A scalar Python direct-DFT harness took 0.05028 s for screen-then-full evaluation versus 0.15662 s for a full transform over 48 pairs (3.11×). This is a toy implementation result, not an optimized FFT or accelerator benchmark. A Lean theorem verifies the implication from a supplied residual bound to safe screening.

## Provenance and prior-access check

The source paper selected from `data/daily/2026-11-14.md` is Alexander H. Nitz, Keisi Kacanja, and Collin D. Capano, [“Peak Performance in Matched Filtering: Fused Correlation and Statistically Controlled Screening on CPUs and GPUs”](https://arxiv.org/abs/2611.17229v1), arXiv:2611.17229v1 (submitted 2026-11-12). The daily archive already contained its title and abstract. Before using the full paper, I checked the existing paper files and paper-specific Git history: neither had an earlier extension or citation for this source. I consulted the paper for this task. This check cannot establish whether it was read outside the repository earlier.

## Method

Let `X[k]` and `H[k]` be length-`N` DFTs for real data and a real template. Retain a conjugate-symmetric bin set `L`; let `T` be the omitted bins. With `C[k] = X[k] conjugate(H[k])`, split the inverse correlation into retained and omitted components:

\[
r[t] = r_L[t] + r_T[t], \qquad
r_T[t] = \frac{1}{N}\sum_{k\in T}X[k]\overline{H[k]}e^{2\pi i kt/N}.
\]

For every lag, triangle inequality followed by Cauchy–Schwarz gives

\[
|r_T[t]| \leq \frac{1}{N}\sum_{k\in T}|X[k]||H[k]|
\leq \frac{1}{N}
\sqrt{\sum_{k\in T}|X[k]|^2}
\sqrt{\sum_{k\in T}|H[k]|^2}
=: B_T.
\]

Therefore `max_t |r[t]| ≤ max_t |r_L[t]| + B_T`. If the right-hand side is strictly below threshold `τ`, the pair cannot have a full-band peak at or above `τ` and may be skipped. For a template bank, template tail energy can be precomputed once per template and data tail energy once per data segment. The retained-band correlation still needs an efficient implementation; a direct reference computation is not evidence of an accelerated kernel.

The Lean theorem at [FrequencyScreenClaims.lean](../formal/PaperClaims/FrequencyScreenClaims.lean) proves the final real-valued implication `|low + high| < threshold`, given `|high| ≤ bound` and `|low| + bound < threshold`. It does not formalize the DFT decomposition, Cauchy–Schwarz bound derivation, floating-point error, or the experiment implementation. For a real kernel, tail energies and the bound need conservative rounding, and transform/accumulation error must be included in the envelope.

## Experiment

Run from the repository root:

```bash
python3 papers/certified_frequency_screening/experiment.py
```

The experiment uses the Python standard library and seed `20261116`. It generates Hermitian spectra at `N=128`; retained bins have 8× the amplitude of omitted bins. It calibrates a 90th-percentile peak threshold using 96 noise pairs, evaluates 512 independent noise pairs, and evaluates 128 template-aligned injections at amplitude 4.0. For each of the 640 evaluation pairs it checks the energy bound against the full direct-DFT correlation, allowing a small floating-point tolerance. The time comparison uses 48 pairs and direct scalar Python DFTs.

## Results

| Measurement | Result |
|---|---:|
| Noise pairs safely skipped | 438/512 (85.5%) |
| Above-threshold injections | 128/128 |
| Above-threshold injections skipped | 0/128 |
| Median noise residual bound | 0.012816 |
| Certificate checks | 640/640 passed |
| Full direct DFT, 48 pairs | 0.15662 s |
| Screen then full if needed | 0.05028 s (3.11× in this harness) |

The script writes the complete run metadata to [results.json](results.json). The output was generated on an Apple M2 CPU under Asahi Linux. No NVIDIA GPU was available or used.

## Limits and next steps

The energy profile is synthetic and deliberately favors low-band screening. The threshold quantile is estimated from only 96 synthetic noise samples; it is not a calibrated false-alarm or false-dismissal rate. The floating-point check is numerical, while the Lean theorem is over exact real numbers under a supplied bound. Timings compare partial and full scalar Python direct DFTs; they do not predict performance against an optimized FFT or any GPU. There is no detector data, nonstationary PSD study, waveform-bank integration, or production search pipeline.

The next tests should measure tail-energy distributions on real detector data and representative waveform banks, evaluate template/data energy reuse over realistic batch sizes, sweep cutoff and threshold, add directed-rounding error margins, compare with a high-precision reference, and implement the gate with an optimized CPU or GPU kernel.

## References

Nitz, Alexander H., Keisi Kacanja, and Collin D. Capano. 2026. “Peak Performance in Matched Filtering: Fused Correlation and Statistically Controlled Screening on CPUs and GPUs.” arXiv:2611.17229v1. https://arxiv.org/abs/2611.17229v1.
