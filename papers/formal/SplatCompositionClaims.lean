import Mathlib.Basic.Real.Basic

/-!
Formal claims for the ordered splat composition extension.

The structure models an already-reduced depth segment by its premultiplied
color contribution and remaining transmittance. The proofs concern exact real
arithmetic only; they say nothing about floating-point error or GPU execution.
-/

namespace SplatCompositionClaims

structure Segment where
  color : ℝ
  transmittance : ℝ

/-- Compose two consecutive front-to-back segments. -/
def compose (a b : Segment) : Segment :=
  ⟨a.color + a.transmittance * b.color,
   a.transmittance * b.transmittance⟩

def identity : Segment := ⟨0, 1⟩

theorem compose_assoc (a b c : Segment) :
    compose (compose a b) c = compose a (compose b c) := by
  cases a with
  | mk ac ta =>
    cases b with
    | mk bc bt =>
      cases c with
      | mk cc tc =>
        simp only [compose, Segment.mk.injEq]
        constructor <;> simp [mul_add, mul_assoc, add_assoc]

theorem compose_identity_left (a : Segment) :
    compose identity a = a := by
  cases a with
  | mk ac ta => simp [compose, identity]

theorem compose_identity_right (a : Segment) :
    compose a identity = a := by
  cases a with
  | mk ac ta => simp [compose, identity]

end SplatCompositionClaims
