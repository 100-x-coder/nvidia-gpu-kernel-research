#!/usr/bin/env python3
"""CPU reference experiment for order-preserving splat chunk composition.

This checks the real-arithmetic monoid law with exact Fraction values and
measures float32 reassociation error on deterministic synthetic alpha layers.
It is not a GPU renderer or a GPU performance benchmark.
"""
from __future__ import annotations

import json
import random
import struct
from fractions import Fraction
from pathlib import Path


Pair = tuple[float, float]  # (premultiplied contribution, remaining transmittance)
ExactPair = tuple[Fraction, Fraction]


def compose(a: Pair, b: Pair) -> Pair:
    """Apply front segment a, then back segment b."""
    return (a[0] + a[1] * b[0], a[1] * b[1])


def compose_exact(a: ExactPair, b: ExactPair) -> ExactPair:
    return (a[0] + a[1] * b[0], a[1] * b[1])


def reduce_tree(items: list[Pair]) -> Pair:
    """Balanced tree whose leaves and subtrees remain in depth order."""
    if not items:
        return (0.0, 1.0)
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return compose(reduce_tree(items[:mid]), reduce_tree(items[mid:]))


def reduce_tree_exact(items: list[ExactPair]) -> ExactPair:
    if not items:
        return (Fraction(0), Fraction(1))
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return compose_exact(reduce_tree_exact(items[:mid]), reduce_tree_exact(items[mid:]))


def f32(x: float) -> float:
    return struct.unpack("!f", struct.pack("!f", x))[0]


def compose_f32(a: Pair, b: Pair) -> Pair:
    return (f32(a[0] + f32(a[1] * b[0])), f32(a[1] * b[1]))


def reduce_tree_f32(items: list[Pair]) -> Pair:
    if not items:
        return (0.0, 1.0)
    if len(items) == 1:
        return tuple(map(f32, items[0]))  # type: ignore[return-value]
    mid = len(items) // 2
    return compose_f32(reduce_tree_f32(items[:mid]), reduce_tree_f32(items[mid:]))


def layers_to_chunks(alphas: list[Fraction], colors: list[Fraction], k: int) -> list[ExactPair]:
    n = len(alphas)
    chunks: list[ExactPair] = []
    for p in range(k):
        lo, hi = n * p // k, n * (p + 1) // k
        c, t = Fraction(0), Fraction(1)
        for alpha, color in zip(alphas[lo:hi], colors[lo:hi]):
            c += t * alpha * color
            t *= 1 - alpha
        chunks.append((c, t))
    return chunks


def to_float_pairs(xs: list[ExactPair]) -> list[Pair]:
    return [(float(c), float(t)) for c, t in xs]


def main() -> None:
    rng = random.Random(20261111)
    cases = 120
    k_values = [2, 3, 4, 8, 16, 32, 64]
    max_abs_color_error = {str(k): 0.0 for k in k_values}
    max_abs_transmittance_error = {str(k): 0.0 for k in k_values}
    max_f32_color_delta_vs_serial = {str(k): 0.0 for k in k_values}
    exact_composition_checks = 0

    for _ in range(cases):
        n = rng.randint(16, 256)
        # Fractions keep the oracle exact while covering transparent through
        # opaque layers and varied but bounded colors.
        alphas = [Fraction(rng.randrange(0, 100), 100) for _ in range(n)]
        colors = [Fraction(rng.randrange(0, 101), 100) for _ in range(n)]
        for k in k_values:
            chunks = layers_to_chunks(alphas, colors, k)
            serial = (Fraction(0), Fraction(1))
            for chunk in chunks:
                serial = compose_exact(serial, chunk)
            tree = reduce_tree_exact(chunks)
            assert tree == serial, (n, k, tree, serial)
            exact_composition_checks += 1

            # The direct layer-by-layer expression is an independent exact
            # oracle for the chunk boundaries and compositing convention.
            direct = Fraction(0)
            trans = Fraction(1)
            for alpha, color in zip(alphas, colors):
                direct += trans * alpha * color
                trans *= 1 - alpha
            assert serial == (direct, trans), (n, k)

            chunk_floats = to_float_pairs(chunks)
            serial32 = (0.0, 1.0)
            for chunk in chunk_floats:
                serial32 = compose_f32(serial32, chunk)
            tree32 = reduce_tree_f32(chunk_floats)
            ref_color, ref_trans = float(direct), float(trans)
            key = str(k)
            max_abs_color_error[key] = max(max_abs_color_error[key], abs(tree32[0] - ref_color))
            max_abs_transmittance_error[key] = max(max_abs_transmittance_error[key], abs(tree32[1] - ref_trans))
            max_f32_color_delta_vs_serial[key] = max(
                max_f32_color_delta_vs_serial[key], abs(tree32[0] - serial32[0])
            )

    result = {
        "experiment": "ordered_splat_composition_tree",
        "seed": 20261111,
        "cases": cases,
        "chunk_counts": k_values,
        "tile_layers": "uniform integer range [16, 256] per case",
        "alpha_color_values": "rational multiples of 1/100 in [0,1]",
        "exact_tree_equals_serial_checks": exact_composition_checks,
        "exact_chunk_composition_equals_direct_layer_scan": True,
        "max_float32_tree_color_abs_error_vs_exact": max_abs_color_error,
        "max_float32_tree_transmittance_abs_error_vs_exact": max_abs_transmittance_error,
        "max_float32_tree_color_delta_vs_serial_chunk_reduction": max_f32_color_delta_vs_serial,
        "operation_model": {
            "composition_operations": "k-1 for both serial and balanced tree",
            "serial_dependency_depth": "k-1",
            "balanced_tree_dependency_depth": "ceil(log2(k)) for the recursively split tree",
            "measured_gpu_latency": None,
        },
        "limitations": [
            "CPU synthetic reference only; no CUDA/GPU kernel is implemented or timed.",
            "Float32 results emulate rounding at composition operations but not device-specific compiler behavior.",
            "The tree reduces dependency depth; the benefit depends on a GPU implementation exposing independent merges.",
        ],
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
