import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  let decl : DefinitionVal := {
    name := `f
    levelParams := []
    type := mkForall `x BinderInfo.default (mkSort levelOne) 
           (mkForall `y BinderInfo.default (mkSort levelOne) (mkSort levelOne)) -- Type → Type → Type
    value := mkLambda `x BinderInfo.default (mkSort levelOne)
            (mkLambda `y BinderInfo.default (mkSort levelOne) (mkBVar 1)) -- fun x y => x
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl decl)