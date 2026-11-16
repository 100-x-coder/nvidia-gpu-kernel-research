module

import Mathlib.Basic.Real.Basic

/-!
A compositional safety lemma for residual-bounded frequency screening.
The premise `hres` is the omitted-band envelope (derived in the paper by
triangle inequality and Cauchy--Schwarz); this theorem proves that a candidate whose
coarse peak plus envelope is below threshold cannot contain a full-band
peak at or above threshold.
-/

theorem certified_frequency_screen_no_false_dismissal
    (low high threshold bound : ℝ)
    (hres : |high| ≤ bound)
    (hgate : |low| + bound < threshold) :
    |low + high| < threshold := by
  calc
    |low + high| ≤ |low| + |high| := abs_add_le low high
    _ ≤ |low| + bound := by simpa [add_comm] using add_le_add_right hres |low|
    _ < threshold := hgate
