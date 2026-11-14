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
| Range-certified dot-product tiling | `RangeCertifiedDotClaims.no_modular_alias`, `interval_residual_bound`, `interval_field_equality` | Endpoint bounds put the integer residual strictly inside `(-p,p)`; a modular equality then implies integer equality. | The proof assumes circuit-enforced sum/output intervals and a positive modulus; it does not verify that a circuit enforces those intervals, that model metadata is bound, or GPU arithmetic. From `papers/formal`, run `lake build Mathlib.Tactic.Linarith` followed by `lake env lean RangeCertifiedDotClaims.lean`. |
| Ordered parallel splat composition | `SplatCompositionClaims.compose_assoc` | The pairwise front-to-back segment composition is associative over real arithmetic. | Exact algebra only; float32 reassociation and GPU behavior are not formalized. The identity laws are also proved. Run `PATH="$HOME/.elan/bin:$PATH" lake env lean SplatCompositionClaims.lean` from `papers/formal`. |

Compile from this directory with:

```sh
lake exe cache get
lake build PaperClaims
```

Each paper's `code_attachment.zip` contains its runnable Python experiment, the corresponding checked-in JSON output, and this Lean proof source/documentation. The scripts use only the Python standard library; Ada-CISA's tests use the standard-library `unittest` runner.

The ordered splat composition paper uses `experiment.py` directly and does not package a code attachment.

## Muir readout quantization bound

`MuirReadoutClaims.lean` proves that coordinate-wise error radii bound a nonnegative weighted signed sum, and that the peak-over-levels quantization bound decreases as the number of levels increases. It was checked with the repository-pinned Lean 4.35.0 / Mathlib toolchain. These lemmas formalize the algebra used by the SABQ paper; they do not prove greedy allocation optimality, code execution, empirical measurements, model quality, or hardware speed.

## Robust decode-boundary regret

`RobustDecodeClaims.lean` proves that if an oracle utility dominates a fixed action pointwise over a finite scenario index, then any nonnegative weighted sum of oracle utility also dominates the weighted sum for that fixed action. Checked using `/home/leesiyuan/.elan/bin/lake env lean RobustDecodeClaims.lean` from `papers/formal` with Lean 4.35.0 / Mathlib v4.35.0. The finite-action minimax selection statement in the paper follows directly from selecting an argmin of worst-case regret; the Lean lemma does not prove scenario generation, surrogate validity, practical switching value, or accelerator behavior.

## Range-certified dot-product tiling

`RangeCertifiedDotClaims.lean` proves that a field equality lifts to integer equality when the full residual interval is strictly within the field modulus. Checked with the repository-pinned Lean 4.35.0 / Mathlib v4.35.0. The experiment uses synthetic operand ranges and does not establish circuit enforcement or accelerator behavior.

## Aggregation-gap certificate

`AggregationGapClaims.lean` was checked with Lean 4.35.0. It proves that the two load vectors `(10, 0)` and `(5, 5)` have equal sums but different maxima, and proves `x^2 + y^2 ≤ max(x,y) * (x+y)` for natural-number `x,y`. The paper gives a direct termwise derivation of the corresponding finite nonnegative-vector lower bound; that general statement is not formalized in Lean. No hardware counters, queueing delay, or GPU runtime are verified. Run `~/.elan/bin/lake env lean AggregationGapClaims.lean` from this directory.

## Decay-aware recurrent-state error budget

`DecayBudgetClaims.lean` checks the scalar step rule and by induction the invariant `e_t ≤ B` for `e_t ≤ α_t e_(t-1) + δ_t`, with `0 ≤ α_t ≤ 1` and `0 ≤ δ_t ≤ B(1-α_t)`. This proves only the real-number recurrence. It does not formalize the matrix Frobenius norm, SVD tail residual, floating-point effects, or SinkState's local/frozen update. Run `~/.elan/bin/lake env lean DecayBudgetClaims.lean` from this directory using Lean 4.35.0 / Mathlib 4.35.0.
