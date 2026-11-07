import Mathlib.Basic.Real.Basic

/-!
Narrow formal lemma accompanying the adaptive contiguous decay grouping study.
This proves the scalar coordinatewise inequality used in the diagonal operator
bound. It does not formalize finite-matrix Frobenius aggregation, the dynamic
program, or floating-point behavior.
-/

namespace SLAClaims

/-- A diagonal gate error bounded by `e` scales each scalar state coordinate
by at most `e` in absolute value. -/
theorem scalar_decay_perturbation_bound (a e x : ℝ)
    (ha : |a| ≤ e) :
    |a * x| ≤ e * |x| := by
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_right ha (abs_nonneg x)

end SLAClaims
