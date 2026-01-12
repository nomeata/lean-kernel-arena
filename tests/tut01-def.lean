import Lean
open Lean Meta Elab Tactic
set_option debug.skipKernelTC true
run_meta
  let decl : DefinitionVal := {
    name := `f
    levelParams := []
    type := mkSort levelOne  -- Type
    value := mkSort levelZero -- Prop
    hints := .opaque
    safety := .safe }
  addDecl (.defnDecl decl)