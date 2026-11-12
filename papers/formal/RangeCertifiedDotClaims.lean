import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega

/-!
Range certificates for integer dot-product relations checked in a prime field.
These lemmas formalize the interval-to-integer implication. They do not verify
that a concrete proof circuit enforces its claimed input ranges.
-/
namespace RangeCertifiedDotClaims

/-- If the integer relation residual lies strictly inside (-p,p), its being
zero modulo p forces the intended integer equality. -/
theorem no_modular_alias {p s z : ℤ} (hp : 0 < p)
    (hbound : -p < s - z ∧ s - z < p)
    (hmod : ∃ k : ℤ, s - z = p * k) : s = z := by
  obtain ⟨k, hk⟩ := hmod
  have hklo : -1 < k := by nlinarith
  have hkhi : k < 1 := by nlinarith
  have hkzero : k = 0 := by omega
  subst k
  have hzero : s - z = 0 := by nlinarith
  omega

/-- Endpoint checks on the interval difference imply every permitted pair of
sum and output values has a wrap-free residual. -/
theorem interval_residual_bound {p sumLo sumHi outLo outHi s z : ℤ}
    (hsLo : sumLo ≤ s) (hsHi : s ≤ sumHi)
    (hzLo : outLo ≤ z) (hzHi : z ≤ outHi)
    (hLower : -p < sumLo - outHi)
    (hUpper : sumHi - outLo < p) :
    -p < s - z ∧ s - z < p := by
  constructor
  · nlinarith
  · nlinarith

/-- Combining the interval certificate with a field equality recovers the
integer equality that a modular constraint alone would not guarantee. -/
theorem interval_field_equality {p sumLo sumHi outLo outHi s z : ℤ}
    (hp : 0 < p)
    (hsLo : sumLo ≤ s) (hsHi : s ≤ sumHi)
    (hzLo : outLo ≤ z) (hzHi : z ≤ outHi)
    (hLower : -p < sumLo - outHi)
    (hUpper : sumHi - outLo < p)
    (hmod : ∃ k : ℤ, s - z = p * k) : s = z := by
  apply no_modular_alias hp (interval_residual_bound hsLo hsHi hzLo hzHi hLower hUpper)
  exact hmod

end RangeCertifiedDotClaims
