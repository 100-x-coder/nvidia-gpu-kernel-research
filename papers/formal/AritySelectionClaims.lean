namespace AritySelectionClaims

def selectedBits (scalarBits pairBits pairDepth : Nat) : Nat :=
  if pairDepth ≤ 11 ∧ pairBits < scalarBits then pairBits else scalarBits

theorem guarded_selection_never_exceeds_scalar
    (scalarBits pairBits pairDepth : Nat) :
    selectedBits scalarBits pairBits pairDepth ≤ scalarBits := by
  unfold selectedBits
  split
  · rename_i h
    exact Nat.le_of_lt h.2
  · exact Nat.le_refl _

end AritySelectionClaims
