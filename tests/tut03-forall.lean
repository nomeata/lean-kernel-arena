import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  let decl : DefinitionVal := {
    name := `f
    levelParams := []
    type := mkSort levelZero  -- Prop
    value := mkForall `p BinderInfo.default (mkSort levelZero) (mkBVar 0) -- ∀ (p: Prop), p
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl decl)