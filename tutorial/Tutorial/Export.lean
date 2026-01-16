import Export
open Lean

/-! This module should not be necessary if lean4export's API were more complete -/

def semver := "3.0.0"

def exportMetadata : Json :=
  let leanMeta := Json.mkObj [
    ("version", versionString),
    ("githash", githash)
  ]
  let exporterMeta := Json.mkObj [
    ("name", "lean4export"),
    ("version", semver)
  ]

  Json.mkObj [
    ("meta", Json.mkObj [
      ("exporter", exporterMeta),
      ("lean", leanMeta)
    ])
  ]

def exportDeclsFromEnv (env : Lean.Environment) (constants : Array Name) : IO Unit := do
  initSearchPath (← findSysroot)
  M.run env do
    modify (fun st => { st with
      exportMData  := false
      exportUnsafe := false
    })
    IO.println exportMetadata.compress
    for c in constants do
      modify (fun st => { st with noMDataExprs := {} })
      let _ ← dumpConstant c
