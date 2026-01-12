import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  -- Define f : Prop → Type using imax for the universe level
  let f_decl : DefinitionVal := {
    name := `f
    levelParams := []
    type := mkForall `p BinderInfo.default (mkSort levelZero) (mkSort levelOne) -- Prop → Type
    value := mkLambda `p BinderInfo.default (mkSort levelZero)
            (mkForall `x BinderInfo.default (mkSort levelOne) (mkBVar 1)) -- fun p => Type → p
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl f_decl)