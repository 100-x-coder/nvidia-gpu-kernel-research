#!/usr/bin/env python3
"""Seeded CPU study of scalar versus pair Huffman coding for quantized codes.

Only the Python standard library is used. The synthetic Markov source is not a
model checkpoint and all hardware-cycle values are a transparent decoder-work
proxy, not measured performance.
"""
from __future__ import annotations
import heapq
import json
import math
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

SEED = 20261115
LANES = 32
WORD_BITS = 32
MAX_CODE_BITS = 11
SIZES = (4096, 65536, 262144)
ALPHABETS = (16, 32, 64)
PERSISTENCES = (0.0, 0.5, 0.8, 0.95)
REPEATS = 4


def markov_codes(n: int, alphabet: int, persistence: float, rng: random.Random) -> list[int]:
    out = [rng.randrange(alphabet)]
    for _ in range(n - 1):
        if rng.random() < persistence:
            out.append(out[-1])
        else:
            out.append(rng.randrange(alphabet))
    return out


def huffman_lengths(counts: Counter[int]) -> dict[int, int]:
    """Return optimal binary Huffman lengths for positive-frequency symbols."""
    if not counts:
        return {}
    heap = []
    serial = max(counts) + 1
    for symbol, freq in sorted(counts.items()):
        heapq.heappush(heap, (freq, serial, symbol))
        serial += 1
    if len(heap) == 1:
        return {heap[0][2]: 1}
    parents: dict[int, tuple[int, int]] = {}
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        node = serial
        serial += 1
        parents[node] = (a, b)
        heapq.heappush(heap, (f1 + f2, node, node))
    root = heap[0][2]
    lengths = {}
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node not in parents:
            lengths[node] = max(1, depth)
        else:
            left, right = parents[node]
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
    return lengths


def pack_cost(codes: list[int], alphabet: int, arity: int) -> dict:
    """Measure exact Huffman payload plus declared canonical-table/layout bytes."""
    lanes: list[list[int]] = [[] for _ in range(LANES)]
    per_lane = math.ceil(len(codes) / LANES)
    for lane in range(LANES):
        start = lane * per_lane
        stop = min(len(codes), start + per_lane)
        segment = codes[start:stop]
        if arity == 1:
            lanes[lane] = segment
        else:
            if len(segment) % 2:
                raise ValueError("pair-mode lane segments must have an even number of codes")
            lanes[lane] = [segment[i] * alphabet + segment[i + 1]
                           for i in range(0, len(segment) - 1, 2)]
    counts = Counter(sym for lane in lanes for sym in lane)
    lengths = huffman_lengths(counts)
    assert sum((Fraction(1, 2 ** width) for width in lengths.values()), Fraction()) <= 1
    max_len = max(lengths.values(), default=0)
    payload_bits = 0
    padded_bits = 0
    for lane in lanes:
        bits = sum(lengths[sym] for sym in lane)
        payload_bits += bits
        padded_bits += ((bits + WORD_BITS - 1) // WORD_BITS) * WORD_BITS
    # Canonical table: one length byte and the minimum-width symbol ID per used symbol.
    symbol_bytes = 1 if alphabet ** arity <= 256 else 2
    table_bytes = len(lengths) * (1 + symbol_bytes)
    # One 32-bit aligned start offset per lane; fixed header counts alphabet and arity.
    layout_bytes = LANES * 4 + 8
    codeable = max_len <= MAX_CODE_BITS
    total_bits = padded_bits + 8 * (table_bytes + layout_bytes)
    weights_per_symbol = arity
    decoder_cycles = max((len(x) for x in lanes), default=0)
    assert sum(map(len, lanes)) * arity == len(codes)
    return {
        "arity": arity,
        "max_code_bits": max_len,
        "codeable_at_11_bits": codeable,
        "payload_bits": payload_bits,
        "lane_padding_bits": padded_bits - payload_bits,
        "codebook_bytes": table_bytes,
        "layout_bytes": layout_bytes,
        "total_bits_including_metadata_and_padding": total_bits,
        "bits_per_weight": total_bits / len(codes),
        "decoder_work_cycles_proxy": decoder_cycles,
        "weights_per_decoder_symbol": weights_per_symbol,
    }


def self_check() -> None:
    lengths = huffman_lengths(Counter({0: 1, 1: 1, 2: 1, 3: 1}))
    assert set(lengths.values()) == {2}
    assert sum((Fraction(1, 2 ** width) for width in lengths.values()), Fraction()) == 1
    rng = random.Random(SEED)
    for alphabet in (16, 32, 64):
        sample = markov_codes(2048, alphabet, 0.8, rng)
        scalar = pack_cost(sample, alphabet, 1)
        pair = pack_cost(sample, alphabet, 2)
        selected = min(
            (candidate for candidate in (scalar, pair) if candidate["codeable_at_11_bits"]),
            key=lambda candidate: candidate["total_bits_including_metadata_and_padding"],
        )
        assert scalar["codeable_at_11_bits"]
        assert selected["total_bits_including_metadata_and_padding"] <= scalar["total_bits_including_metadata_and_padding"]
        assert not pair["codeable_at_11_bits"] or pair["max_code_bits"] <= MAX_CODE_BITS


def main() -> None:
    self_check()
    rng = random.Random(SEED)
    records = []
    for alphabet in ALPHABETS:
        for persistence in PERSISTENCES:
            for n in SIZES:
                repeat_records = []
                for repeat in range(REPEATS):
                    codes = markov_codes(n, alphabet, persistence, rng)
                    scalar = pack_cost(codes, alphabet, 1)
                    pair = pack_cost(codes, alphabet, 2)
                    candidates = [x for x in (scalar, pair) if x["codeable_at_11_bits"]]
                    selected = min(candidates, key=lambda x: x["total_bits_including_metadata_and_padding"])
                    repeat_records.append({"scalar": scalar, "pair": pair, "selected_arity": selected["arity"]})
                    assert selected["total_bits_including_metadata_and_padding"] <= scalar["total_bits_including_metadata_and_padding"]
                records.append({
                    "alphabet_size": alphabet,
                    "persistence_probability": persistence,
                    "weights": n,
                    "repeats": repeat_records,
                    "pair_selected_count": sum(x["selected_arity"] == 2 for x in repeat_records),
                    "pair_ineligible_count": sum(not x["pair"]["codeable_at_11_bits"] for x in repeat_records),
                    "mean_scalar_bits_per_weight": sum(x["scalar"]["bits_per_weight"] for x in repeat_records) / REPEATS,
                    "mean_pair_bits_per_weight": sum(x["pair"]["bits_per_weight"] for x in repeat_records) / REPEATS,
                    "mean_eligible_pair_bits_per_weight": (
                        sum(x["pair"]["bits_per_weight"] for x in repeat_records if x["pair"]["codeable_at_11_bits"])
                        / max(1, sum(x["pair"]["codeable_at_11_bits"] for x in repeat_records))
                    ),
                    "mean_selected_bits_per_weight": sum(min(
                        [x[k]["bits_per_weight"] for k in ("scalar", "pair") if x[k]["codeable_at_11_bits"]]
                    ) for x in repeat_records) / REPEATS,
                    "mean_selected_decoder_work_cycles_proxy": sum(
                        next(x[k]["decoder_work_cycles_proxy"] for k in ("scalar", "pair")
                             if x[k]["arity"] == y["selected_arity"])
                        for y in repeat_records for x in [y]
                    ) / REPEATS,
                })
    output = {
        "study": "Arity selection for Huffman-coded quantized codes",
        "source_paper": "arXiv:2611.17108v1",
        "seed": SEED,
        "repeats_per_condition": REPEATS,
        "decoder_lanes": LANES,
        "word_alignment_bits": WORD_BITS,
        "maximum_supported_codeword_bits": MAX_CODE_BITS,
        "alphabet_sizes": list(ALPHABETS),
        "persistence_probabilities": list(PERSISTENCES),
        "matrix_code_counts": list(SIZES),
        "synthetic_source": "stationary uniform-marginal Markov chain; retain previous code with probability p, else draw uniformly",
        "metadata_assumption": "one byte per used code length plus minimum-width symbol ID; 32 four-byte lane offsets and an eight-byte header; both arities charged equally for lane layout",
        "limitations": ["synthetic integer sequences, not quantized checkpoint weights", "cycle proxy counts decoder symbols per lane only and is not a hardware timing result", "no GPU or ASIC decoder was available", "pair codebooks exceeding the 11-bit lookup window are marked ineligible"],
        "records": records,
    }
    dest = Path(__file__).with_name("results.json")
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(f"self-check passed; wrote {dest}; conditions={len(records)}; repeats={len(records)*REPEATS}")
    for r in records:
        if r["weights"] == SIZES[-1]:
            print(f"A={r['alphabet_size']:2d} p={r['persistence_probability']:.2f} "
                  f"pair={r['mean_pair_bits_per_weight']:.3f} "
                  f"scalar={r['mean_scalar_bits_per_weight']:.3f} "
                  f"select2={r['pair_selected_count']}/4 "
                  f"pair_ineligible={r['pair_ineligible_count']}/4")


if __name__ == "__main__":
    main()
