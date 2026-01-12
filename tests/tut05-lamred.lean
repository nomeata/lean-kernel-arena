import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  -- First define f : Type → Type → Type := fun x y => x
  let f_decl : DefinitionVal := {
    name := `f
    levelParams := []
    type := mkForall `x BinderInfo.default (mkSort levelOne) 
           (mkForall `y BinderInfo.default (mkSort levelOne) (mkSort levelOne)) -- Type → Type → Type
    value := mkLambda `x BinderInfo.default (mkSort levelOne)
            (mkLambda `y BinderInfo.default (mkSort levelOne) (mkBVar 1)) -- fun x y => x
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl f_decl)
  
  -- Then define thm : f Prop (Prop → Prop) which should reduce to Prop
  let thm_decl : TheoremVal := {
    name := `thm
    levelParams := []
    type := mkApp2 (mkConst `f) (mkSort levelZero) 
              (mkForall `p BinderInfo.default (mkSort levelZero) (mkSort levelZero))
    value := mkSort levelZero }
  addDecl (.thmDecl thm_decl)