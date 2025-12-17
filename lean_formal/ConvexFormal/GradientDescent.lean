/-
Copyright (c) 2026 xumingyue. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: xumingyue
-/

import Mathlib

/-!
# Formalized Gradient Descent Basics

This file formalizes:

- convex function (`ConvexFun`)
- strong convexity with a given gradient oracle (`StronglyConvexWithGrad`)
- Lipschitz-continuous gradient (`LipschitzGradient`)
- fixed-step gradient descent recursion (`gdStep`, `gdIter`)
- linear-rate convergence from a one-step contraction inequality

The bridge from "strong convex + smooth" to one-step contraction is kept as
an explicit hypothesis (`hContractionFromTheory`), so the induction-based
rate proof is fully machine checked and reusable.
-/

namespace ConvexFormal

/-! ## Core concepts -/

def ConvexFun (f : ℝ → ℝ) : Prop :=
  ∀ x y t, 0 ≤ t → t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

def StronglyConvexWithGrad (f grad : ℝ → ℝ) (m : ℝ) : Prop :=
  0 < m ∧
  ∀ x y,
    f y ≥ f x + grad x * (y - x) + (m / 2) * (y - x) ^ 2

def LipschitzGradient (grad : ℝ → ℝ) (L : ℝ) : Prop :=
  0 ≤ L ∧ ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖

/-! ## Fixed-step gradient descent recursion -/

def gdStep (grad : ℝ → ℝ) (α : ℝ) (x : ℝ) : ℝ := x - α * grad x

def gdIter (grad : ℝ → ℝ) (α : ℝ) (x0 : ℝ) : ℕ → ℝ
  | 0 => x0
  | n + 1 => gdStep grad α (gdIter grad α x0 n)

def error (grad : ℝ → ℝ) (α x0 xStar : ℝ) (n : ℕ) : ℝ :=
  ‖gdIter grad α x0 n - xStar‖

/-!
If each GD step is contractive around `xStar` with factor `q`, then the full
trajectory converges linearly at rate `q^n`.
-/
theorem linearConvergence_fromContraction
    (grad : ℝ → ℝ) (α x0 xStar q : ℝ)
    (hq_nonneg : 0 ≤ q)
    (hstep : ∀ x, ‖gdStep grad α x - xStar‖ ≤ q * ‖x - xStar‖) :
    ∀ n : ℕ, error grad α x0 xStar n ≤ q ^ n * error grad α x0 xStar 0 := by
  intro n
  induction n with
  | zero =>
      simp [error]
  | succ n ih =>
      have h1 : error grad α x0 xStar (n + 1) ≤ q * error grad α x0 xStar n := by
        simpa [error, gdIter] using hstep (gdIter grad α x0 n)
      calc
        error grad α x0 xStar (n + 1)
            ≤ q * error grad α x0 xStar n := h1
        _ ≤ q * (q ^ n * error grad α x0 xStar 0) := by
              exact mul_le_mul_of_nonneg_left ih hq_nonneg
        _ = q ^ (n + 1) * error grad α x0 xStar 0 := by
              ring

/-!
This theorem captures the standard statement for fixed-step GD under
strong-convex + smooth assumptions.

The difficult inequality
`‖T(x)-x*‖ ≤ ((L-m)/(L+m)) ‖x-x*‖`
is supplied as hypothesis `hContractionFromTheory`; once that lemma is fully
formalized, the end-to-end linear-rate theorem is immediate.
-/
theorem gradientDescent_linearRate
    (f grad : ℝ → ℝ)
    (m L α x0 xStar : ℝ)
    (hStrong : StronglyConvexWithGrad f grad m)
    (_hLip : LipschitzGradient grad L)
    (_hFixed : grad xStar = 0)
    (hmL : m ≤ L)
    (hContractionFromTheory :
      ∀ x, ‖gdStep grad α x - xStar‖ ≤ ((L - m) / (L + m)) * ‖x - xStar‖) :
    ∀ n : ℕ,
      error grad α x0 xStar n
        ≤ ((L - m) / (L + m)) ^ n * error grad α x0 xStar 0 := by
  rcases hStrong with ⟨hm_pos, _⟩
  have h_num_nonneg : 0 ≤ L - m := sub_nonneg.mpr hmL
  have h_den_pos : 0 < L + m := by nlinarith [hm_pos, hmL]
  have hq_nonneg : 0 ≤ (L - m) / (L + m) := by
    exact div_nonneg h_num_nonneg (le_of_lt h_den_pos)
  intro n
  exact linearConvergence_fromContraction grad α x0 xStar ((L - m) / (L + m))
    hq_nonneg hContractionFromTheory n

end ConvexFormal
