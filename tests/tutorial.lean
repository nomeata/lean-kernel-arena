-- Tutorial declarations for Lean type theory features
-- Each declaration exercises a specific feature of the type system
set_option linter.unusedVariables false

-- tut01: Basic definition
def tut01 : Type := Prop

-- tut02: Arrow type (function type)
def tut02 : Type := Prop → Prop

-- tut03: Dependent type (forall)
def tut03 : Prop := ∀ (p: Prop), p

-- tut04: Lambda expression
def tut04 : Type → Type → Type := fun x y => x

-- tut05: Lambda reduction (requires two declarations)
def tut05f : Type → Type → Type := fun x y => x
theorem tut05thm : tut05f Prop (Prop → Prop) = Prop := rfl

-- tut06: Universe polymorphism
def tut06f.{u} : Sort u → Sort u → Sort u := fun α β => α
theorem tut06thm : tut06f Prop (Prop → Prop) = Prop := rfl

-- tut07: Function from Prop to Type (tests imax universe levels)
def tut07 : Prop → Type := fun _ => Unit