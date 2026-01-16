-- Tutorial declarations for Lean type theory features
-- Each declaration exercises a specific feature of the type system

import Tutorial.Meta
set_option linter.unusedVariables false

/-- Basic definition -/
good_def tut01 : Type := Prop

-- tut01bad1: Mismatched types
bad_def tut01_bad01 : Prop := unchecked Type

-- tut02: Arrow type (function type)
good_def tut02 : Type := Prop → Prop

-- tut03: Dependent type (forall)
good_def tut03 : Prop := ∀ (p: Prop), p

-- tut04: Lambda expression
good_def tut04 : Type → Type → Type := fun x y => x

-- tut05: Lambda reduction (requires two declarations)
good_def tut05 : tut04 Prop (Prop → Prop) := ∀ p : Prop, p

-- tut06: Universe polymorphism
def tut06f.{u} : Sort u → Sort u → Sort u := fun α β => α
good_def tut06 : tut06f Prop (Prop → Prop) := ∀ p : Prop, p

set_option debug.skipKernelTC true in
open Lean Meta in
run_meta addDecl <| .defnDecl {
  name := `tut06_bad01
  levelParams := [`u, `u]
  type := .sort 1
  value := .sort 0
  hints := .opaque
  safety := .safe
}

-- tut07: Function from Prop to Type (tests imax universe levels)
good_def tut07a : (p : Prop) → Prop := fun p => Type → p

good_def tut07b : (α : Type) → Type 1 := fun α => Type → α
