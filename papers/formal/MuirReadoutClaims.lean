import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
Formal bounds for the accompanying CPU reference study of blockwise readout
quantization. This file proves a weighted perturbation inequality; it does not
prove the empirical result, quantizer implementation, or accelerator behavior.
-/
namespace MuirReadoutClaims

/-- If every scalar approximation error is at most its radius, then any
nonnegative weighted signed sum has absolute error at most the weighted sum of
radii. This is the scalar readout bound used tile by tile in the paper. -/
theorem weighted_readout_error_bound {n : Nat} (weight error radius : Fin n → ℝ)
    (hweight : ∀ i, 0 ≤ weight i)
    (herror : ∀ i, |error i| ≤ radius i) :
    |∑ i : Fin n, weight i * error i| ≤ ∑ i : Fin n, weight i * radius i := by
  calc
    |∑ i : Fin n, weight i * error i| ≤
        ∑ i : Fin n, |weight i * error i| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i : Fin n, weight i * |error i| := by
      simp_rw [abs_mul, abs_of_nonneg (hweight _)]
    _ ≤ ∑ i : Fin n, weight i * radius i := by
      apply Finset.sum_le_sum
      intro i hi
      exact mul_le_mul_of_nonneg_left (herror i) (hweight i)

/-- Increasing the level count cannot increase the peak-over-levels
quantization-error bound when the scale and sensitivity are nonnegative. -/
theorem more_levels_reduce_bound (sensitivity peak lo hi : ℝ)
    (hs : 0 ≤ sensitivity) (hp : 0 ≤ peak) (hlo : 0 < lo) (horder : lo ≤ hi) :
    sensitivity * peak / hi ≤ sensitivity * peak / lo := by
  have hnum : 0 ≤ sensitivity * peak := mul_nonneg hs hp
  exact div_le_div_of_nonneg_left hnum hlo horder

end MuirReadoutClaims
