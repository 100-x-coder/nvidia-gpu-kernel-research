#!/usr/bin/env python3
"""CPU reference study of causal leakage from shared microscaling exponents.

This is a numerical dependency experiment, not an FP8 hardware benchmark.
It uses a small E4M3-like finite codebook and power-of-two shared scales.
"""
from __future__ import annotations

import json
import math
import random
from bisect import bisect_left
from pathlib import Path


def e4m3_codebook() -> list[float]:
    values = {0.0}
    for sign in (-1.0, 1.0):
        for exponent in range(16):
            for mantissa in range(8):
                if exponent == 15 and mantissa == 7:  # reserve NaN code
                    continue
                if exponent == 0:
                    v = (mantissa / 8.0) * 2.0**-6
                else:
                    v = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
                values.add(sign * v)
    return sorted(values)


CODEBOOK = e4m3_codebook()
MAX_CODE = max(CODEBOOK)


def nearest_code(x: float) -> float:
    i = bisect_left(CODEBOOK, x)
    if i == 0:
        return CODEBOOK[0]
    if i == len(CODEBOOK):
        return CODEBOOK[-1]
    return CODEBOOK[i - 1] if x - CODEBOOK[i - 1] <= CODEBOOK[i] - x else CODEBOOK[i]


def power2_scale(max_abs: float) -> float:
    if max_abs == 0.0:
        return 1.0
    exponent = math.ceil(math.log2(max_abs / MAX_CODE))
    return 2.0**exponent


def quantize_group(values: list[float]) -> tuple[list[float], float]:
    scale = power2_scale(max(abs(x) for x in values))
    return [nearest_code(x / scale) * scale for x in values], scale


def quantize(values: list[float], group_size: int) -> tuple[list[float], list[float]]:
    result: list[float] = []
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        block = values[start : start + group_size]
        q, scale = quantize_group(block)
        result.extend(q)
        scales.append(scale)
    return result, scales


def quantize_causal_prefix(values: list[float], group_size: int = 32) -> tuple[list[float], list[float]]:
    """Use a within-block scale derived only from each token's causal prefix."""
    result: list[float] = []
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        prefix_max = 0.0
        for value in values[start : start + group_size]:
            prefix_max = max(prefix_max, abs(value))
            scale = power2_scale(prefix_max)
            result.append(nearest_code(value / scale) * scale)
            scales.append(scale)
    return result, scales


def study(seed: int = 20261104, length: int = 4096, trials: int = 64) -> dict:
    rng = random.Random(seed)
    # Background values are intentionally small; each block receives a rare
    # large activation to stress scale changes near causal boundaries.
    sequences: list[list[float]] = []
    for _ in range(trials):
        x = [rng.gauss(0.0, 0.03) for _ in range(length)]
        # Place each outlier inside (rather than at the end of) common scale
        # groups so it can reveal the dependency on earlier positions.
        for i in range(95, length, 128):
            x[i] = rng.choice((-1.0, 1.0)) * rng.uniform(4096.0, 65536.0)
        sequences.append(x)

    rows = []
    for group_size in (1, 2, 4, 8, 16, 32):
        changed = 0
        compared = 0
        sq_error = 0.0
        background_sq_error = 0.0
        background_n = 0
        n = 0
        for original in sequences:
            quantized, _ = quantize(original, group_size)
            sq_error += sum((a - b) ** 2 for a, b in zip(original, quantized))
            for a, b in zip(original, quantized):
                if abs(a) < 1.0:
                    background_sq_error += (a - b) ** 2
                    background_n += 1
            n += len(original)
            # Every position is evaluated under its causal prefix. Appending
            # later tokens must not change a past token's reconstruction.
            for start in range(0, len(original), group_size):
                block = original[start : start + group_size]
                full_q, _ = quantize_group(block)
                for i in range(len(block)):
                    if i + 1 < len(block):
                        prefix_q, _ = quantize_group(block[: i + 1])
                        changed += prefix_q[-1] != full_q[i]
                        compared += 1
        rows.append(
            {
                "scale_group_tokens": group_size,
                "rmse": math.sqrt(sq_error / n),
                "background_rmse": math.sqrt(background_sq_error / background_n),
                "adjacent_future_change_rate": changed / compared if compared else 0.0,
                "scale_groups_per_sequence": math.ceil(length / group_size),
            }
        )
    changed = 0
    compared = 0
    sq_error = 0.0
    background_sq_error = 0.0
    background_n = 0
    n = 0
    scale_runs = 0
    for original in sequences:
        q, scales = quantize_causal_prefix(original)
        sq_error += sum((a - b) ** 2 for a, b in zip(original, q))
        for a, b in zip(original, q):
            if abs(a) < 1.0:
                background_sq_error += (a - b) ** 2
                background_n += 1
        n += len(original)
        scale_runs += 1 + sum(a != b for a, b in zip(scales, scales[1:]))
        # Recompute every prefix independently; the reconstruction already
        # assigned to a token must remain invariant as later tokens arrive.
        for start in range(0, len(original), 32):
            block = original[start : start + 32]
            for i in range(1, len(block)):
                prefix_q, _ = quantize_causal_prefix(block[: i + 1], group_size=32)
                changed += q[start + i] != prefix_q[-1]
                compared += 1
    rows.append(
        {
            "scale_group_tokens": "causal_prefix_32_token_blocks",
            "rmse": math.sqrt(sq_error / n),
            "background_rmse": math.sqrt(background_sq_error / background_n),
            "adjacent_future_change_rate": changed / compared,
            "scale_groups_per_sequence": scale_runs / trials,
        }
    )
    return {
        "description": "CPU E4M3-like quantization dependency simulation; not GPU timing",
        "seed": seed,
        "length": length,
        "trials": trials,
        "rows": rows,
    }


if __name__ == "__main__":
    result = study()
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
