# Arity Selection for Huffman-Coded Quantized Weights

**Lee Si Yuan**  
Independent Researcher  
[siyuanlee.cs@gmail.com](mailto:siyuanlee.cs@gmail.com)

## Abstract

LazyWeights combines entropy-shaped quantized codes, matrix-level Huffman coding, and 32 interleaved decode lanes. Its released design pairs 4-bit codes into one Huffman symbol, while wider code alphabets use single codes. This paper studies a small extension: choose scalar or pair symbols per matrix by comparing the complete modeled storage cost, subject to the existing 11-bit maximum codeword constraint. The selector includes Huffman payload bits, a canonical-table representation, 32-bit lane padding, and lane-offset metadata. A standard-library CPU experiment evaluates seeded stationary Markov streams over 16-, 32-, and 64-symbol alphabets, four correlation levels, three matrix sizes, and four repeats per condition. At 262,144 symbols, pair mode is selected in all four runs for a 16-symbol alphabet at persistence 0.5 and 0.8, reducing modeled rate from 4.005 to 3.444 and 2.766 bits per weight. Pair mode is rejected at persistence 0.95 because unconstrained Huffman trees violate the 11-bit limit; it is also rejected in every tested 64-symbol case. The result supports a guarded per-matrix choice in this synthetic model, not a measured model-compression or hardware-speed claim.

## Source and prior-access audit

The source is Arnab Sanyal, Gourav Datta, and Sandeep P. Chinchali, “LazyWeights: Decode-on-Demand Edge LLMs,” arXiv:2611.17108v1 (12 November 2026). Its title and abstract appear in the daily feed. Before this extension, the identifier was absent from the repository's `papers/` artifacts and paper-specific Git history. The full v1 text was fetched and read for this work. This audit can establish repository history and the available session context; it cannot establish personal or external access logs.

LazyWeights already reports pair coding for 4-bit codes and a length-limited canonical Huffman code with an 11-bit decode window. The extension here is the matrix-level rate selector and its explicit treatment of table, lane-padding, and 11-bit feasibility costs across different code alphabet widths. It does not claim that pair coding itself is new.

## Method

For a matrix represented by codes from an alphabet of size $A$, the experiment compares arity $a=1$ and $a=2$. In pair mode, consecutive codes within each of 32 contiguous lane segments become one symbol in the product alphabet of size $A^2$. Each sweep size is divisible by 64, so every lane contains an even number of weights and no scalar tail is dropped.

The histogram over the full synthetic matrix defines an ordinary binary Huffman tree. Its exact integer code lengths determine payload bits. A candidate is infeasible when its longest codeword exceeds 11 bits, matching the source design's decoder-window constraint. The modeled total adds (i) one byte per used codeword length, (ii) the minimum-width symbol ID for every used code, (iii) 32 four-byte lane offsets, (iv) an eight-byte header, and (v) per-lane padding to a 32-bit boundary. Both modes pay the same lane-layout overhead. The pair selector chooses the lower total among feasible candidates; scalar mode is always feasible in these experiments.

The source sequence starts with a uniform random symbol. Each subsequent symbol repeats the previous one with probability $p$; otherwise, it is drawn uniformly from the alphabet. Thus the stationary marginal is uniform and $p$ controls adjacent-code dependence. We test $A\in\{16,32,64\}$, $p\in\{0,0.5,0.8,0.95\}$, matrix sizes 4,096, 65,536, and 262,144 codes, four deterministic repeats, and seed 20261115. The Python script reports exact Huffman length sums and metadata under the stated serialization model. A decoder-work proxy counts the longest lane's number of Huffman symbols, assuming one symbol per lane per cycle; it excludes shared-memory-port stalls, code-table lookup cost, buffering, and physical timing.

## Results

At 262,144 weights, four repeats per point:

| Alphabet | Persistence $p$ | Scalar (bits/weight) | Pair (bits/weight) | Pair feasible / selected | Selector (bits/weight) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.00 | 4.005 | 4.020 | 4/4 feasible, 0/4 selected | 4.005 |
| 16 | 0.50 | 4.005 | 3.444 | 4/4 feasible, 4/4 selected | 3.444 |
| 16 | 0.80 | 4.005 | 2.766 | 4/4 feasible, 4/4 selected | 2.766 |
| 16 | 0.95 | 4.005 | 2.257* | 0/4 feasible, 0/4 selected | 4.005 |
| 32 | 0.50 | 5.006 | 4.301* | 3/4 feasible, 3/4 selected | 4.477 |
| 64 | 0.50 | 6.008 | 5.345* | 0/4 feasible, 0/4 selected | 6.008 |

`*` The displayed pair rate averages unconstrained Huffman trees, including trees that exceed the allowed 11-bit maximum. It is diagnostic only; the selector cannot choose those candidates. Selector rate averages each repeat's selected feasible mode.

For 16-symbol codes at $p=0.5$, pair coding saves 14.0% of modeled bits per weight; at $p=0.8$, it saves 30.9%. The symbol-work proxy halves from 8,192 scalar symbols per lane to 4,096 pair symbols per lane at this matrix size. This is an upper-bound work count, not a latency prediction. At $p=0.95$, the pair tree becomes infeasible despite its attractive unconstrained rate: rare transitions deepen the longest codeword beyond 11 bits. The 32-symbol case at $p=0.5$ selects pair mode in three of four runs; the 64-symbol case cannot meet the codeword limit in any repeat at this size.

Small matrices pay a larger fixed table cost. At 4,096 weights, for example, pair mode is not selected for the 16-symbol alphabet at $p=0.5$, while it is selected at $p=0.8$ and $p=0.95$. This reinforces that the choice must include metadata and feasibility, not only entropy or expected code length.

## Lean check

`papers/formal/AritySelectionClaims.lean` defines the selector guard and proves that the selected bit count never exceeds the feasible scalar candidate's bit count. The proof is intentionally narrow: it verifies the fallback rule, not Huffman optimality, the synthetic measurements, the 11-bit property of any generated tree, or decoder hardware. It compiles with Lean 4.35.0 using the core Lean environment.

## Limitations

The sequences are synthetic and do not represent GPTQ, AWQ, or GGUF checkpoint code distributions. Ordinary Huffman trees are measured and then rejected if they exceed 11 bits; this experiment does not implement a length-limited Huffman optimizer, which might recover some rejected pair cases. The codebook and lane metadata costs are an explicit accounting model, not a byte-for-byte LazyWeights bundle. No actual bitstream decoder, model-quality check, GPU, FPGA, or ASIC was run. In particular, pair-mode table lookup and the source paper's shared eight-word read port may change real throughput. The study therefore supports testing the guarded selector on real quantized matrices; it does not establish end-to-end compression or speedups.

## Reproduction

From this directory, run `python3 experiment.py`. It uses only the Python standard library and writes `results.json`. The Lean selector lemma can be checked from `papers/formal` with `~/.elan/bin/lean AritySelectionClaims.lean`.

## References

Arnab Sanyal, Gourav Datta, and Sandeep P. Chinchali. 2026. *LazyWeights: Decode-on-Demand Edge LLMs*. arXiv:2611.17108v1. https://arxiv.org/abs/2611.17108v1.

David A. Huffman. 1952. A method for the construction of minimum-redundancy codes. *Proceedings of the IRE*, 40(9):1098–1101. https://doi.org/10.1109/JRPROC.1952.273898.
