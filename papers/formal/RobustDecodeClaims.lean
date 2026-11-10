import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
Formal expected-utility ordering used by the robust decode-boundary note.
This proves only a finite weighted-sum consequence of pointwise oracle
dominance; it does not validate the paper's surrogate or hardware behavior.
-/
namespace RobustDecodeClaims

/-- A scenario-wise oracle that dominates a fixed action also dominates it
after averaging over any nonnegative finite scenario weights. -/
theorem weighted_oracle_dominates_fixed {n : Nat}
    (weight oracle fixed : Fin n → ℝ)
    (hweight : ∀ i, 0 ≤ weight i)
    (horacle : ∀ i, fixed i ≤ oracle i) :
    (∑ i : Fin n, weight i * fixed i) ≤
      (∑ i : Fin n, weight i * oracle i) := by
  apply Finset.sum_le_sum
  intro i hi
  exact mul_le_mul_of_nonneg_left (horacle i) (hweight i)

end RobustDecodeClaims
