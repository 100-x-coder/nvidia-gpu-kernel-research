#!/usr/bin/env python3
"""CPU reference experiment for deterministic residual-bound screening.

Uses only Python's standard library. This validates numerical safety and the
screening decision on synthetic real-signal spectra; timings are scalar Python
DFTs and are not representative of optimized FFT/GPU performance.
"""
from __future__ import annotations
import cmath
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

SEED = 20261116
N = 128
CUTOFF = 12  # retain DC and positive bins 1..11 (with conjugate mirrors)
NOISE_TRAIN = 96
NOISE_EVAL = 512
SIGNAL_EVAL = 128
THRESHOLD_QUANTILE = 0.90


def real_spectrum(rng: random.Random, n: int, scale: float = 1.0) -> list[complex]:
    """Draw Hermitian spectrum for a real signal, with independent bins."""
    x = [0j] * n
    def spectral_scale(k: int) -> float:
        # Controlled low-band-dominant regime: tail amplitudes are 12.5%.
        return scale * (1.0 if min(k, n-k) < CUTOFF else 0.125)
    x[0] = complex(rng.gauss(0, spectral_scale(0)), 0)
    x[n // 2] = complex(rng.gauss(0, spectral_scale(n//2)), 0)
    for k in range(1, n // 2):
        bin_scale = spectral_scale(k)
        z = complex(rng.gauss(0, bin_scale / math.sqrt(2)), rng.gauss(0, bin_scale / math.sqrt(2)))
        x[k], x[n-k] = z, z.conjugate()
    return x


def full_corr(x: list[complex], h: list[complex]) -> list[complex]:
    n = len(x)
    c = [x[k] * h[k].conjugate() for k in range(n)]
    return [sum(c[k] * cmath.exp(2j * math.pi * k * t / n) for k in range(n)) / n
            for t in range(n)]


def low_corr(x: list[complex], h: list[complex], cutoff: int) -> list[complex]:
    n = len(x)
    bins = list(range(cutoff)) + list(range(n-cutoff+1, n))
    c = [(k, x[k] * h[k].conjugate()) for k in bins]
    return [sum(v * cmath.exp(2j * math.pi * k * t / n) for k, v in c) / n
            for t in range(n)]


def residual_bound(x: list[complex], h: list[complex], cutoff: int) -> float:
    """triangle-inequality/Cauchy bound on omitted inverse-DFT residual at every lag."""
    n = len(x)
    tail = list(range(cutoff, n-cutoff+1))
    ex = sum(abs(x[k])**2 for k in tail)
    eh = sum(abs(h[k])**2 for k in tail)
    return math.sqrt(ex * eh) / n


def peaks(x: list[complex], h: list[complex]) -> list[complex]:
    return full_corr(x, h)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, math.ceil(q * len(ordered)) - 1)]


def assess(x: list[complex], h: list[complex], cutoff: int, threshold: float):
    lo = low_corr(x, h, cutoff)
    b = residual_bound(x, h, cutoff)
    upper = max(map(abs, lo)) + b
    skipped = upper < threshold
    exact_peak = max(map(abs, full_corr(x, h)))
    assert exact_peak <= upper + 2e-10 * max(1.0, exact_peak)
    if skipped:
        assert exact_peak < threshold + 2e-10 * max(1.0, threshold)
    return skipped, exact_peak, upper, b


def main() -> None:
    rng = random.Random(SEED)
    templates = [real_spectrum(rng, N) for _ in range(NOISE_TRAIN + NOISE_EVAL + SIGNAL_EVAL)]
    training = []
    for i in range(NOISE_TRAIN):
        x = real_spectrum(rng, N)
        training.append(max(map(abs, full_corr(x, templates[i]))))
    threshold = percentile(training, THRESHOLD_QUANTILE)

    noise_rows = []
    for i in range(NOISE_TRAIN, NOISE_TRAIN + NOISE_EVAL):
        x = real_spectrum(rng, N)
        noise_rows.append(assess(x, templates[i], CUTOFF, threshold))

    signal_rows = []
    for i in range(NOISE_TRAIN + NOISE_EVAL, len(templates)):
        h = templates[i]
        noise = real_spectrum(rng, N)
        # Inject a template-aligned real signal with a fixed amplitude.
        x = [noise[k] + 4.0 * h[k] for k in range(N)]
        signal_rows.append(assess(x, h, CUTOFF, threshold))

    # Measure this reference implementation end-to-end on a fixed subset.
    bench = [(real_spectrum(rng, N), real_spectrum(rng, N)) for _ in range(48)]
    t0 = time.perf_counter()
    for x, h in bench:
        full_corr(x, h)
    full_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    for x, h in bench:
        skip, _, _, _ = assess(x, h, CUTOFF, threshold)
        # assess computes the exact answer for validation; remove its cost below
        # by recording only decision statistics in production-style timing.
    validation_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    for x, h in bench:
        lo = low_corr(x, h, CUTOFF)
        b = residual_bound(x, h, CUTOFF)
        if max(map(abs, lo)) + b >= threshold:
            full_corr(x, h)
    screened_seconds = time.perf_counter() - t0

    skipped_noise = sum(row[0] for row in noise_rows)
    skipped_signals = sum(row[0] for row in signal_rows)
    max_ub_gap = max(row[2] - row[1] for row in noise_rows + signal_rows)
    median_bound = statistics.median(row[3] for row in noise_rows)
    result = {
        "study": "deterministic omitted-band certificate; synthetic CPU reference only",
        "source_paper": "Nitz et al., arXiv:2611.17229v1 (2026)",
        "seed": SEED,
        "host": platform.platform(),
        "cpu": "Apple M2 (Asahi Linux; 8 reported logical CPUs)",
        "accelerator": "none used; no NVIDIA GPU available",
        "transform_length": N,
        "retained_positive_bins_including_dc": CUTOFF,
        "threshold_training_noise_pairs": NOISE_TRAIN,
        "threshold_empirical_quantile": THRESHOLD_QUANTILE,
        "threshold": threshold,
        "noise_evaluation_pairs": NOISE_EVAL,
        "noise_pairs_safely_skipped": skipped_noise,
        "noise_skip_fraction": skipped_noise / NOISE_EVAL,
        "signal_evaluation_pairs": SIGNAL_EVAL,
        "injected_signal_amplitude": 4.0,
        "above_threshold_signal_pairs": sum(row[1] >= threshold for row in signal_rows),
        "above_threshold_signals_safely_skipped": sum(row[0] and row[1] >= threshold for row in signal_rows),
        "signal_pairs_safely_skipped": skipped_signals,
        "max_upper_bound_minus_exact_peak_noise_and_signal": max_ub_gap,
        "median_noise_residual_bound": median_bound,
        "all_pairwise_certificate_checks_passed": True,
        "reference_timing_seconds_48_pairs": {
            "full_scalar_dft": full_seconds,
            "screen_then_full_if_needed": screened_seconds,
            "validation_full_answer_included": validation_seconds,
            "screened_over_full_speedup": full_seconds / screened_seconds,
        },
        "limitations": [
            "synthetic Gaussian spectra with 8x low-band to tail amplitude ratio and template-aligned injections",
            "threshold quantile is a small empirical calibration, not a false-dismissal guarantee",
            "certificate safety is deterministic conditional on exact arithmetic; this implementation uses floating point with explicit numerical tolerance",
            "scalar Python DFT timing is only a harness check, not an optimized FFT, GPU, or production benchmark",
            "no nonstationary PSD drift, real detector data, or end-to-end search pipeline",
        ],
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
