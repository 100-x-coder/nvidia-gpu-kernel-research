#!/usr/bin/env python3
"""Reproducible interval-bound study for wrap-safe integer dot-product tiles.

This is a range-analysis experiment, not a GPU benchmark or proof-system run.
Only bounds explicitly declared as verifier-enforced preconditions are used.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

P = 2**31 - 2**24 + 1  # KoalaBear prime used by Sable
INT32_MAX = 2**31 - 1
SEED = 20261112


def product_interval(xlo: int, xhi: int, wlo: int, whi: int) -> tuple[int, int]:
    assert xlo <= xhi and wlo <= whi
    corners = (xlo*wlo, xlo*whi, xhi*wlo, xhi*whi)
    return min(corners), max(corners)


def safe_chunk_length(p: int, x_bounds: tuple[int, int], w_bounds: tuple[int, int],
                      output_bounds_per_term: tuple[int, int], hardware_max: int) -> int:
    """Largest repeated-term chunk with a <p residual and signed-int32 sum.

    For n terms, S lies in [n*plo, n*phi] and the committed output z lies
    in [n*zlo, n*zhi]. Thus S-z lies in
    [n*(plo-zhi), n*(phi-zlo)]. Require both endpoints strictly inside
    (-p,p), and the raw dot-product interval inside signed int32.
    """
    plo, phi = product_interval(*x_bounds, *w_bounds)
    zlo, zhi = output_bounds_per_term
    assert plo <= phi and zlo <= zhi
    best = 0
    for n in range(1, hardware_max + 1):
        residual_lo = n * (plo - zhi)
        residual_hi = n * (phi - zlo)
        sum_lo, sum_hi = n * plo, n * phi
        if not (-p < residual_lo and residual_hi < p):
            break
        if not (-2**31 <= sum_lo and sum_hi <= INT32_MAX):
            break
        best = n
    return best


def check_certificate(p: int, n: int, xvals: list[int], wvals: list[int],
                      x_bounds: tuple[int, int], w_bounds: tuple[int, int]) -> None:
    assert len(xvals) == len(wvals) == n
    assert all(x_bounds[0] <= x <= x_bounds[1] for x in xvals)
    assert all(w_bounds[0] <= w <= w_bounds[1] for w in wvals)
    total = sum(x*w for x, w in zip(xvals, wvals))
    # The full-output interval and strict residual bound exclude modular aliases.
    lo, hi = product_interval(*x_bounds, *w_bounds)
    assert n * lo <= total <= n * hi
    assert -2**31 <= total <= INT32_MAX
    out_lo, out_hi = n * lo, n * hi
    assert -p < n * lo - out_hi and n * hi - out_lo < p
    aliases = [z for z in (total - p, total, total + p) if out_lo <= z <= out_hi]
    assert aliases == [total]


def main() -> None:
    rng = random.Random(SEED)
    n = 100_000
    rows = 512
    x_bounds = (-127, 127)
    global_w = (-127, 127)
    # Per-row weight ranges are a public, verifier-checked range contract.
    # Synthetic heterogeneous channel bounds test planner behavior only.
    magnitudes = [rng.randint(8, 127) for _ in range(rows)]
    global_product = product_interval(*x_bounds, *global_w)
    global_chunk = safe_chunk_length(P, x_bounds, global_w, global_product, n)
    row_chunks = []
    max_safe = []
    for m in magnitudes:
        wb = (-m, m)
        chunk = safe_chunk_length(P, x_bounds, wb, product_interval(*x_bounds, *wb), n)
        assert chunk > 0
        max_safe.append(chunk)
        row_chunks.append((n + chunk - 1) // chunk)

    # Check signed corner cases, asymmetric intervals, and the exact strict
    # boundary n*residual == p (which must be rejected).
    assert product_interval(-2, 3, -5, 7) == (-15, 21)
    assert safe_chunk_length(P, (-1, 1), (-1, 1), (-1, 1), 10) == 10
    boundary_n = (P - 1) // (2 * 127 * 127)
    strict_chunk = safe_chunk_length(P, (-127, 127), (-127, 127),
                                     product_interval(-127, 127, -127, 127), P)
    assert strict_chunk == boundary_n
    assert 2 * 127 * 127 * (strict_chunk + 1) >= P
    # Generated concrete values stay within the declared input contracts.
    xs = [rng.randint(-127, 127) for _ in range(4096)]
    ws = [rng.randint(-127, 127) for _ in range(4096)]
    check_certificate(P, len(xs), xs, ws, x_bounds, global_w)

    baseline_chunks = rows * ((n + global_chunk - 1) // global_chunk)
    adaptive_chunks = sum(row_chunks)
    result = {
        "study": "synthetic interval-planner evaluation; not a GPU benchmark",
        "seed": SEED,
        "prime": P,
        "hardware_accumulator": "signed int32",
        "reduction_length": n,
        "output_rows": rows,
        "activation_interval": list(x_bounds),
        "global_weight_interval": list(global_w),
        "global_safe_chunk_length": global_chunk,
        "per_row_safe_chunk_length_min": min(max_safe),
        "per_row_safe_chunk_length_median": sorted(max_safe)[rows // 2],
        "per_row_safe_chunk_length_max": max(max_safe),
        "rows_safe_without_split": sum(c >= n for c in max_safe),
        "baseline_row_chunk_count": baseline_chunks,
        "interval_aware_row_chunk_count": adaptive_chunks,
        "chunk_count_reduction_fraction": 1 - adaptive_chunks / baseline_chunks,
        "strict_wrap_boundary_chunk_length": strict_chunk,
        "checks": [
            "interval corner products",
            "strict modular-wrap boundary",
            "int32 accumulator bounds",
            "4096 deterministic concrete dot-product terms; modular aliases excluded",
        ],
        "limitations": [
            "synthetic declared channel ranges; real ranges must be enforced by the proof circuit",
            "no GPU kernel, prover, end-to-end model, or latency evaluation",
        ],
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
