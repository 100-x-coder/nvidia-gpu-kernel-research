import Mathlib

/-! Toy formal witnesses for the aggregation gap. -/
def total (x y : Nat) : Nat := x + y
def makespan (x y : Nat) : Nat := max x y

theorem equal_aggregate_different_makespan :
    total 10 0 = total 5 5 ∧ makespan 10 0 ≠ makespan 5 5 := by
  constructor <;> decide

theorem max_dominates_second_moment_certificate (x y : Nat) :
    x*x + y*y ≤ makespan x y * total x y := by
  unfold makespan total
  rcases le_total x y with h | h
  · rw [max_eq_right h]
    nlinarith [Nat.mul_le_mul_left x h, Nat.mul_le_mul_left y (Nat.zero_le x)]
  · rw [max_eq_left h]
    nlinarith [Nat.mul_le_mul_left y h, Nat.mul_le_mul_left x (Nat.zero_le y)]
