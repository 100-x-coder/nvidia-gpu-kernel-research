# Lean-Checked Claims and Reproduction Scope

The theorems in `PaperClaims.lean` were checked with Lean 4.35.0 and Mathlib v4.35.0. They prove narrow algebraic implications used by the papers; they do not prove the Python simulations, GPU behavior, or empirical results.

| Paper | Lean theorem | What it proves | Key assumptions / limits |
|---|---|---|---|
| Feedback batching | `PaperClaims.batch_memory_safe` | `n * actual ≤ budget` follows from a conservative per-item bound and a capacity already constrained by the reserved budget. | Requires `actual ≤ observed`, `n * observed ≤ reserve * budget`, nonnegative `n` and `budget`, and `reserve ≤ 1`. The CPU simulation uses exact stable class footprints; it does not establish an upper bound for arbitrary drifting inputs. |
| Noise-aware promotion | `PaperClaims.paired_shared_drift_cancels` | Common additive drift cancels in paired log-latency subtraction. | Algebra only; it does not establish that real GPU timings share drift or satisfy the simulation's distribution. |
| Noise-aware promotion | `PaperClaims.log_gate_iff_ratio` | The log-latency threshold is equivalent to the stated latency ratio for positive baseline/candidate latencies and positive gain. | Algebraic transformation only; the confidence rule's statistical calibration remains an empirical/modeling question. |
| Causal prefix microscaling | `PaperClaims.prefix_summary_independent_of_suffix` | Any deterministic summary of a prefix is unchanged when suffix values change while that prefix stays fixed. | Abstract prefix property; it does not formalize E4M3 rounding, floating-point implementation, or gradient behavior. |
| Causal prefix microscaling | `PaperClaims.prefix_maximum_nondecreasing` | The running maximum used to derive prefix scales cannot decrease when one more value is admitted. | Proves the prefix-maximum recurrence; it does not formalize the logarithm/ceiling/power-of-two conversion or quantizer. |
| Ada-CISA | `PaperClaims.noa_beta_zero_is_uniform` | The normalized exponentiated norm weights become uniform when the temperature is zero. | Applies to the stated softmax formula; the Python implementation also clips exponents and has a small-mean fallback. |
| ACDG | `SLAClaims.scalar_decay_perturbation_bound` | A diagonal gate error bounded by `e` scales each scalar state coordinate by at most `e` in absolute value. | Coordinatewise lemma only. The finite-matrix Frobenius aggregation, Python dynamic program, and floating-point behavior are not formalized. Run `lake env lean SLAClaims.lean` from `papers/formal`. |
| Ada-CISA | `PaperClaims.uniform_mean_separates_salient_key` | Splits one key's contribution from the background term in an arithmetic chunk mean; its coefficient is `1 / C`. | Algebra only; it does not prove that this term dominates retrieval scores or that a miss is caused by dilution. |
| Noise-aware promotion | `PaperClaims.monotone_map_lt_iff` | A strictly monotone map preserves and reflects strict inequalities over a linear order. | Generic order fact; used to explain why a monotone transform preserves a threshold comparison. |

Compile from this directory with:

```sh
lake exe cache get
lake build PaperClaims
```

Each paper's `code_attachment.zip` contains its runnable Python experiment, the corresponding checked-in JSON output, and this Lean proof source/documentation. The scripts use only the Python standard library; Ada-CISA's tests use the standard-library `unittest` runner.

## Muir readout quantization bound

`MuirReadoutClaims.lean` proves that coordinate-wise error radii bound a nonnegative weighted signed sum, and that the peak-over-levels quantization bound decreases as the number of levels increases. It was checked with the repository-pinned Lean 4.35.0 / Mathlib toolchain. These lemmas formalize the algebra used by the SABQ paper; they do not prove greedy allocation optimality, code execution, empirical measurements, model quality, or hardware speed.

## Robust decode-boundary regret

`RobustDecodeClaims.lean` proves that if an oracle utility dominates a fixed action pointwise over a finite scenario index, then any nonnegative weighted sum of oracle utility also dominates the weighted sum for that fixed action. Checked using `/home/leesiyuan/.elan/bin/lake env lean RobustDecodeClaims.lean` from `papers/formal` with Lean 4.35.0 / Mathlib v4.35.0. The finite-action minimax selection statement in the paper follows directly from selecting an argmin of worst-case regret; the Lean lemma does not prove scenario generation, surrogate validity, practical switching value, or accelerator behavior.
