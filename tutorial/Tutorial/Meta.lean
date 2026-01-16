import Lean
import Tutorial.TestCaseEnv

open Lean Elab Term Command
open Lean.Parser.Command

def addTestCaseDeclCore (descr? : Option String) (decl : Lean.Declaration) (outcome : Outcome) : CoreM Unit := do
  match outcome with
  | .good => addDecl decl
  | .bad =>
    withOptions (fun o => debug.skipKernelTC.set o true) do
      addDecl decl
  registerTestCase {
    decl := decl.getNames.head!
    outcome := outcome
    description := descr?
  }

def addTestCaseDecl (descr? : Option String) (declName : Name) (levelParams : List Name) (typeExpr : Expr) (valueExpr : Expr) (outcome : Outcome) (declKind : ConstantKind) : CoreM Unit := do
  let decl ← match declKind with
    | .defn => pure <| .defnDecl {
        name := declName
        levelParams := levelParams
        type := typeExpr
        value := valueExpr
        hints := .opaque
        safety := .safe
      }
    | .thm => pure <| .thmDecl {
        name := declName
        levelParams := []
        type := typeExpr
        value := valueExpr
      }
    | _ => throwError "Unsupported declaration kind in test case: {repr declKind}"
  addTestCaseDeclCore descr? decl outcome

open TSyntax.Compat in -- due to plainDocComments vs. docComment
def elabAndAddTestCaseDecl (descr? : Option (TSyntax ``plainDocComment)) (name : TSyntax ``declId) (type : Term) (value : Term) (outcome : Outcome) (declKind : ConstantKind) : CommandElabM Unit := liftTermElabM do
  let descrStr? ← descr?.mapM (getDocStringText ·)
  let descrStr? := descrStr?.map (·.trimAscii.copy)
  let (declName, lparams) ← match name with
    | `(declId| $n:ident) => pure (n.getId, [])
    | `(declId| $n:ident .{ $[$ls:ident],* }) => pure (n.getId, ls.toList.map (·.getId))
    | _ => throwUnsupportedSyntax
  withLevelNames lparams do
    let typeExpr ← instantiateMVars (← elabTerm type none)
    let valueExpr ← instantiateMVars (← elabTerm value (some typeExpr))
    addTestCaseDecl descrStr? declName lparams typeExpr valueExpr outcome declKind

elab descr?:(plainDocComment)? "good_def " name:declId ":" type:term ":=" value:term : command => do
  elabAndAddTestCaseDecl descr? name type value Outcome.good ConstantKind.defn

elab descr?:(plainDocComment)? "bad_def " name:declId ":" type:term ":=" value:term : command => do
  elabAndAddTestCaseDecl descr? name type value Outcome.bad ConstantKind.defn

open TSyntax.Compat in -- due to plainDocComments vs. docComment
def elabRawTestDecl (descr? : Option (TSyntax `Lean.Parser.Command.plainDocComment)) (decl : Term) (outcome : Outcome) : CommandElabM Unit := liftTermElabM do
  let descrStr? ← descr?.mapM (getDocStringText ·)
  let descrStr? := descrStr?.map (·.trimAscii.copy)
  let expectedType := Lean.mkConst ``Lean.Declaration
  let declExpr ← elabTerm decl (some expectedType)
  let decl ← Lean.Meta.MetaM.run' <| unsafe Meta.evalExpr (α := Lean.Declaration) expectedType declExpr
  addTestCaseDeclCore descrStr? decl outcome

elab descr?:(plainDocComment)? "good_decl " decl:term : command => do
  elabRawTestDecl descr? decl .good

elab descr?:(plainDocComment)? "bad_decl " decl:term : command => do
  elabRawTestDecl descr? decl .bad

section Unchecked

/-- An elaborator that just inserts the term, without regard for the acutal type needed here -/
syntax (name := unchecked) "unchecked" term : term

section
open Lean Meta Elab Term


@[term_elab «unchecked»]
def elabUnchecked : TermElab := fun stx expectedType? => do
  match stx with
  | `(unchecked $t) =>
    let some expectedType := expectedType? |
      tryPostpone
      throwError "invalid 'unchecked', expected type required"
    let e ←  elabTerm t none
    let mvar ← mkFreshExprMVar expectedType MetavarKind.syntheticOpaque
    mvar.mvarId!.assign e
    return mvar
  | _ => throwUnsupportedSyntax

end

end Unchecked
