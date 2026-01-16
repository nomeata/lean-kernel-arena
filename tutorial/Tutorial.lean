-- Tutorial declarations for Lean type theory features
-- Each declaration exercises a specific feature of the type system

import Tutorial.Meta
set_option linter.unusedVariables false

/-- Basic definition -/
good_def basicDef : Type := Prop

/-- Mismatched types -/
bad_def badDef : Prop := unchecked Type

/-- Arrow type (function type) -/
good_def arrowType : Type := Prop → Prop

/-- Dependent type (forall) -/
good_def dependentType : Prop := ∀ (p: Prop), p

/-- Lambda expression -/
good_def simpleLambda : Type → Type → Type := fun x y => x

/-- Lambda reduction (requires two declarations) -/
good_def betaReduction : simpleLambda Prop (Prop → Prop) := ∀ p : Prop, p

/-- The type of a declaration has to be a type, not some other expression -/
bad_def nonTypeType : simpleLambda := unchecked Prop

/-- Some level computation -/
good_def levelComp1 : Type 0 := Sort (imax 1 0)

/-- Some level computation -/
good_def levelComp2 : Type 1 := Sort (max 1 0)

/-- Some level computation -/
good_def levelComp3 : Type 2 := Sort (imax 2 1)

def levelParamF.{u} : Sort u → Sort u → Sort u := fun α β => α

/-- Level parameters -/
good_def levelParams : levelParamF Prop (Prop → Prop) := ∀ p : Prop, p

/-- Duplicate universe paramers -/
bad_decl .defnDecl {
  name := `tut06_bad01
  levelParams := [`u, `u]
  type := .sort 1
  value := .sort 0
  hints := .opaque
  safety := .safe
}

/-- Some level computation -/
good_def levelComp4.{u} : Type 0 := Sort (imax u 0)

/-- Some level computation -/
good_def levelComp5.{u} : Type u := Sort (imax u u)

/-- Type inference for forall using imax -/
good_def imax1 : (p : Prop) → Prop := fun p => Type → p

/-- Type inference for forall using imax -/
good_def imax2 : (α : Type) → Type 1 := fun α => Type → α
