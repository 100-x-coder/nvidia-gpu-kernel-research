module

import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Order.Basic
import Mathlib.Tactic.Ring

/-!
Machine-checked lemmas for algebraic claims used by the accompanying papers.
This file deliberately proves mathematical implications only; empirical rates,
synthetic outputs, and hardware projections are checked by their Python scripts.
-/
namespace PaperClaims

/-- If the estimated per-item footprint upper-bounds the true footprint and the
chosen capacity fits within the reserved budget, total payload memory fits the
worker budget whenever the reserve is at most one. -/
theorem batch_memory_safe {α : Type} [LinearOrder α] [CommRing α] [IsOrderedRing α]
    (n budget reserve observed actual : α)
    (hn : 0 ≤ n) (hbudget : 0 ≤ budget)
    (hreserve : reserve ≤ 1) (hactual : actual ≤ observed)
    (hcap : n * observed ≤ reserve * budget) :
    n * actual ≤ budget := by
  calc
    n * actual ≤ n * observed := mul_le_mul_of_nonneg_left hactual hn
    _ ≤ reserve * budget := hcap
    _ ≤ budget := by
      calc
        reserve * budget ≤ 1 * budget := mul_le_mul_of_nonneg_right hreserve hbudget
        _ = budget := one_mul budget

/-- Shared additive drift cancels when subtracting paired log-latencies. -/
theorem paired_shared_drift_cancels {α : Type} [CommRing α]
    (muB muC drift epsB epsC : α) :
    ((muC + drift + epsC) - (muB + drift + epsB)) =
      ((muC - muB) + (epsC - epsB)) := by
  ring

/-- The paper's log-latency threshold is equivalent to the stated latency
ratio threshold for positive latencies and a positive practical gain. -/
theorem log_gate_iff_ratio (baseline candidate gain : ℝ)
    (hbaseline : 0 < baseline) (hcandidate : 0 < candidate) (hgain : 0 < gain) :
    Real.log candidate - Real.log baseline < -Real.log (1 + gain) ↔
      candidate / baseline < 1 / (1 + gain) := by
  have hright : -Real.log (1 + gain) = Real.log (1 / (1 + gain)) := by
    calc
      -Real.log (1 + gain) = Real.log 1 - Real.log (1 + gain) := by simp
      _ = Real.log (1 / (1 + gain)) := by
        rw [Real.log_div one_ne_zero (by positivity : (1 + gain : ℝ) ≠ 0)]
  rw [← Real.log_div hcandidate.ne' hbaseline.ne', hright]
  exact (Real.log_lt_log_iff (div_pos hcandidate hbaseline)
    (by positivity : 0 < (1 / (1 + gain) : ℝ)))

/-- The NOA softmax weights reduce to uniform weights at zero temperature. -/
noncomputable def noaWeight (n : Nat) (beta meanNorm : ℝ) (norm : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (beta * (norm i / meanNorm)) /
    ∑ j : Fin n, Real.exp (beta * (norm j / meanNorm))

theorem noa_beta_zero_is_uniform {n : Nat} (meanNorm : ℝ)
    (norm : Fin n → ℝ) (i : Fin n) :
    noaWeight n 0 meanNorm norm i = 1 / (n : ℝ) := by
  simp [noaWeight]

/-- A single key's contribution to a uniform arithmetic mean of a chunk has
coefficient `1 / C`; the surrounding background contribution remains separate. -/
theorem uniform_mean_separates_salient_key (chunkSize salient backgroundMean : ℝ) :
    (salient + (chunkSize - 1) * backgroundMean) / chunkSize =
      salient / chunkSize + ((chunkSize - 1) * backgroundMean) / chunkSize := by
  ring

/-- A value computed only from a prefix is unchanged when later suffix elements
change, provided the observed prefixes agree. -/
def prefixSummary {α β : Type} (summary : List α → β) (xs : List α) (i : Nat) : β :=
  summary (xs.take (i + 1))

theorem prefix_summary_independent_of_suffix
    {α β : Type} (summary : List α → β) (xs ys : List α) (i : Nat)
    (hprefix : xs.take (i + 1) = ys.take (i + 1)) :
    prefixSummary summary xs i = prefixSummary summary ys i := by
  simp [prefixSummary, hprefix]

/-- A running maximum cannot decrease when one more value is admitted. -/
def prefixMaximum {α : Type} [LinearOrder α] [Zero α] (xs : List α) : Nat → α
  | 0 => 0
  | n + 1 => max (prefixMaximum xs n) (xs[n]?.getD 0)

theorem prefix_maximum_nondecreasing {α : Type} [LinearOrder α] [Zero α]
    (xs : List α) (n : Nat) :
    prefixMaximum xs n ≤ prefixMaximum xs (n + 1) := by
  simp only [prefixMaximum]
  exact le_max_left _ _

/-- Strictly monotone transformations preserve strict inequalities. This is the
order fact used when translating a log-latency threshold to an exponential one. -/
theorem monotone_map_lt_iff {α β : Type} [LinearOrder α] [Preorder β]
    (f : α → β) (hf : StrictMono f) (a b : α) :
    a < b ↔ f a < f b := hf.lt_iff_lt.symm

end PaperClaims
