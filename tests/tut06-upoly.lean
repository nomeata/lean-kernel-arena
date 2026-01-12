import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  -- Define f : Sort u → Sort u → Sort u := fun α β => α (universe polymorphic version)
  let f_decl : DefinitionVal := {
    name := `f
    levelParams := [`u]
    type := mkForall `α BinderInfo.default (mkSort (mkLevelParam `u)) 
           (mkForall `β BinderInfo.default (mkSort (mkLevelParam `u)) (mkSort (mkLevelParam `u)))
    value := mkLambda `α BinderInfo.default (mkSort (mkLevelParam `u))
            (mkLambda `β BinderInfo.default (mkSort (mkLevelParam `u)) (mkBVar 1))
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl f_decl)
  
  -- Define thm : f Prop (Prop → Prop) which should reduce to Prop
  let thm_decl : TheoremVal := {
    name := `thm
    levelParams := []
    type := mkApp2 (mkConst `f [levelZero]) (mkSort levelZero) 
              (mkForall `p BinderInfo.default (mkSort levelZero) (mkSort levelZero))
    value := mkSort levelZero }
  addDecl (.thmDecl thm_decl)