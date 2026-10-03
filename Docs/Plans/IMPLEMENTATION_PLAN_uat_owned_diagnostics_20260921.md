# Owned production workflow diagnostics

Task13260.278.9; continuation of the approved engineering sweep. This bounded diagnostic run precedes the full Stage4 freeze and does not replace any43-family, four-cell, installation or upgrade gate.

## Stage 1: Commit and prepare owned sources
**Goal**: Preserve a reviewed source checkpoint and prepare distinct SQLite/PostgreSQL source copies using the existing isolation adapter.
**Success Criteria**: Exact commit/source manifests, locked dependency provenance, no shared checkout/service changes, and restricted official PostgreSQL fixture receipt.
**Tests**: Existing126 regressions, type/lint/build checks; source hashes; dependency origins; official fixture/role preflight.
**Status**: Complete

## Stage 2: Build and start diagnostic profiles
**Goal**: Production WebUI and API serving attributable source on owned ports.
**Success Criteria**: Build inside each source archive; no dev server; health/port/process ownership receipts; fresh data roots and isolated browser state. Reused Python/frontend dependencies are disclosed.
**Tests**: Source hash audit before/after build, production artifact hashes, health and canonical engine/role readback.
**Status**: Complete

## Stage 3: Exercise bounded linked journeys
**Goal**: Run four Content Review cases and source-grounding/five-card journeys against real owned SQLite and PostgreSQL backends.
**Success Criteria**: Exact case identities and first attempts retained, no silent skip/pass, canonical content/version/IDs/reload outcomes recorded.
**Tests**: Existing strengthened Playwright cases with autostartfalse, retries0, one worker, explicit provider/mode and private output. Seeded-auth diagnostics are separate from normal UI login and setup.
**Status**: In Progress

## Stage 4: Record findings and resume full sweep
**Goal**: Capture every harness/product/quality finding and finish causal repairs before qualification.
**Success Criteria**: Tracker/tasks include actual outcomes and unresolved obligations; no diagnostic result presented as full UAT or release approval.
**Tests**: Planned/result reconciliation; source/artifact ownership review; SQLite/official PostgreSQL regression for each causal product repair.
**Status**: Not Started

Diagnostic1: committed d45d0a47869f3ef1624370535d2c60674c4ca823; two source archives each25,238files, identical sourceSHA105a2f427d7c6520ed2904c75ecad759a37b87549e49cbec8a897cf645142ec9. Both production archive builds and token/budget checks pass. Artifact recorder initially rejected generated symlinks; failed attempt preserved and recording corrected without altering source/build.753 links/3dangling standalone dependency links per artifact require packaging investigation before standalone installation qualification. Current diagnostics use dependency-bearing next start. SQLite/API18921/Web18922; official restricted-role PostgreSQL/API18923/Web18924. Both health checks confirm actual engine and single_user mode; each model/page preflight200. SQLite six-case native attempt underway, retries0, seeded auth, real llama.cpp.
