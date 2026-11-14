import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-!
The scalar arithmetic behind the decay-aware error-budget invariant.
This does not formalize matrix norms, an SVD implementation, or SinkState.
-/

theorem decay_budget_step
    (alpha b B delta : ℝ)
    (halpha0 : 0 ≤ alpha) (hbB : b ≤ B)
    (hdelta : delta ≤ B - alpha * b) :
    alpha * b + delta ≤ B := by
  have hmul : alpha * b ≤ alpha * B := mul_le_mul_of_nonneg_left hbB halpha0
  linarith

theorem propagated_error_within_certificate
    (alpha e b B delta : ℝ)
    (halpha0 : 0 ≤ alpha)
    (he : e ≤ b)
    (hlocal : delta ≤ B - alpha * b) :
    alpha * e + delta ≤ B := by
  have hmul : alpha * e ≤ alpha * b := mul_le_mul_of_nonneg_left he halpha0
  linarith

theorem one_step_triangle_bound
    (alpha e delta bNext : ℝ)
    (hdelta : delta ≤ bNext - alpha * e) :
    alpha * e + delta ≤ bNext := by
  linarith

theorem recurrent_error_stays_bounded
    (decay truncation error : ℕ → ℝ) (B : ℝ)
    (hdecay0 : ∀ t, 0 ≤ decay t)
    (hbudget : ∀ t, truncation t ≤ B - decay t * B)
    (hinit : error 0 ≤ B)
    (hstep : ∀ t, error (t + 1) ≤ decay t * error t + truncation t) :
    ∀ t, error t ≤ B := by
  intro t
  induction t with
  | zero => exact hinit
  | succ t ih =>
      have hmul : decay t * error t ≤ decay t * B := by
        exact mul_le_mul_of_nonneg_left ih (hdecay0 t)
      have hnext := hstep t
      have hcap := hbudget t
      calc
        error (t + 1) ≤ decay t * error t + truncation t := hnext
        _ ≤ decay t * B + (B - decay t * B) := add_le_add hmul hcap
        _ = B := by linarith
