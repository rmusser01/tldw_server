# Canonical role for newly seeded Character greetings

Task: TASK-13260.278.18.83.45, association recorded before edits.
ADR required: no. Existing native H1 sender roles and ADR-049 govern this interoperability repair; no authority or storage architecture change.
Scope: shared Character conversation factory, existing real DB/native prompt-delivery regression, this owned plan and associated task. Preserve accepted greeting text, behavior snapshot identity, transaction, caller guards, disabled/empty greeting and Sync exclusions. Historical named rows remain unchanged.

## Stage 1: Reproduce
**Goal**: A real seeded Character conversation must expose an assistant greeting to native history capture.
**Success Criteria**: Added assertion fails on unchanged factory before any production edit.
**Tests**: Existing real database/native prompt-delivery integration test, normal configured fixture and 600/300-second limits.
**Status**: Complete

## Stage 2: Minimal repair and affected qualification
**Goal**: Use the canonical assistant sender for newly seeded greetings.
**Success Criteria**: Existing initial-message semantics and accepted content unchanged; affected tests, syntax/lint and scoped baseline/current Bandit qualified.
**Tests**: Native prompt delivery, greeting API and behavior snapshot consumers genuinely affected by the factory change; no model or runtime replay.
**Status**: Complete

## Stage 3: Review and live integration gate
**Goal**: Independent actual-source review and later supported live integration.
**Success Criteria**: New patch reviewed; no live completion inferred from local tests. Previously consumed Send recovery stays consumed.
**Tests**: Root owns later runtime and browser acceptance; no startup or additional Send in this unit.
**Status**: In Progress

Causal RED on unchanged factory:1failed36warnings/3.19s pytest/7.044s outer/natural1. Added real native snapshot assertion sees E2E-TestBot instead of assistant. No setup/provider/model failure. Minimal repair changes only newly seeded greeting sender to canonical assistant; accepted content/name/snapshot and initial-message loop unchanged. Initial tracked-clean preflight correctly stopped before writes because the factory already contains78 added dictionary-qualification lines. Those existing WIP hunks were read and preserved byte-for-byte by inverse substitution; no stale clean claim. Running APIs retain pre-repair imported source; no restart or corrected runtime claim.

Local qualification: all three complete affected files passed naturally:147passed3223warnings/365.91s pytest/372.902s outer at unchanged600/300 limits. Both touched Python bodies compile; diff whitespace clear. Ruff retains the same inherited factory I001 and no new diagnostic. Scoped baseline/current Bandit production0/test0/errors0; final test-only comment scan0findings0errors. Independent actual-source review CLEAR across both changed bodies and connected callers/native consumers; original11 assertions preserved plus one role assertion. Existing78 dictionary WIP lines preserved exactly. Both running APIs imported pre-repair source; no restart/additional Send/model compliance/live acceptance. Stage3 remains In Progress for later supported live integration.
