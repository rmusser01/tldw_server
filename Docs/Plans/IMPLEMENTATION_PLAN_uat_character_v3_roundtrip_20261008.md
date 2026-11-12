# B01 Character V3 export/import repair — 2026-10-08

Associated task: TASK-13260.278.18.83.45 (root-owned; this new reviewable unit was recorded before edits). Parent owns task updates and runtime/UAT validation. Prompt production remains frozen.

## Diagnosis and scope

Actual B01 default Character export has `spec: chara_card_v3`, version `3.0`, nested required fields, and generation extensions. Seven optional text fields are `null`. The upload endpoint already selects JSON and the shared importer already dispatches to V3. The V3 validator rejects nullable optional text, falls through V2/V1, and finally reports missing flat V1 name. The V3 export branch alone emits nullable DB text verbatim; existing V2/PNG export uses `_as_export_str`.

Use that existing export helper for V3 text. Treat legacy `null` optional V3 text as omitted text (empty-string parser default), while preserving required fields, every non-null type/length guard, collection validation, ownership, extensions and format markers. No importer routing/default/auth changes.

ADR check: ADR required: no. Searched `Docs/ADR/README.md`; this restores the existing supported V3 export/import contract without changing API shape, owner-bound storage, security or module boundaries. ADR-020 continues to govern per-user storage; ADR-006 governs scoped Bandit output. No new architecture rule.

## Stage 1: Reproduce the V3 boundary failure
**Goal**: Causal regression over existing V3 parser and real export/import endpoint paths.
**Success Criteria**: Existing source rejects a valid card with legacy optional-null text and produces invalid nullable V3 export; fixtures do not mask validation/auth/lifecycle.
**Tests**: Nullable optional text normalization; required/wrong-type rejection; cookie/CSRF-backed actual default V3 export/reimport and already-produced legacy card import; existing V1/V2/PNG controls.
**Status**: Complete

## Stage 2: Repair the existing V3 boundaries
**Goal**: Minimal compatibility correction in existing V3 parser and exporter.
**Success Criteria**: New V3 exports have typed text defaults; existing nullable optional-text exports import with unchanged required validation and extension payloads; V1/V2/PNG behavior survives.
**Tests**: Focused new regressions and existing V3 parser/property/import controls; normal configured plugins, supported Python 3.12 environment, 600-second outer/300-second case caps, maximum three distinct attempts per issue.
**Status**: Complete

## Stage 3: Qualify and hand off the source
**Goal**: Scoped compile/Ruff/baseline-current static Bandit and independent review/live validation handoff.
**Success Criteria**: No new security/lint findings; exact files/results/limitations provided to root, source frozen on root signal.
**Tests**: Compile touched Python; Ruff touched scope with baseline comparison; existing primary Python 3.11/Bandit only as an AST scanner (application tests stay Python 3.12); whitespace and connected-source review.
**Status**: In Progress

## Boundaries

Do not modify Prompt production, unrelated Character/Regenerate scope, parent task/plan, live API/frontend/profile processes, browser/protected requests, secrets/auth stores, model assets, environment/install/network, proof/logs or Git history/index/publication. No fake application imports or custom SQL/database provisioning. Test-owned ephemeral databases use existing canonical fixtures and official DB abstractions.

## Causal red evidence

- Supported Python 3.12.13 / pytest 9.1.1, configured plugins and timeout300, outer600: 10 failed,25 passed,469 warnings,26.47s pytest/33.718s outer,natural exit1; no skips or fixture failures.
- Eight optional-null-text parser regressions failed. Canonical minted-cookie/CSRF-backed real router and official per-user DB legacy import reproduced HTTP400 `Import validation failed: Missing required field in V1 card: name`. Real default V3 export failed structural validation. V2/PNG export/import controls passed.
- Baseline static scanner verified existing primary Python3.11.13/Bandit1.9.3:19 inherited B101 test assertions in existing V3 tests,0 production findings,0 scan errors. Scanner is AST-only; application qualification remains Python3.12.

## Scoped green and static evidence

- New regressions, existing V3 property/parser controls, existing V1/implicit-V2/file-import controls:45 passed,114 unrelated cases deselected,456 warnings,no skips,38.59s pytest/44.783s outer,natural0. Same Python3.12 and configured plugins/caps.
- Actual default V3 and existing V2/PNG exports reimport with names changed and nested generation extensions preserved. Legacy-null V3 cookie import rejects missing CSRF; valid CSRF imports through canonical per-user DB. Official other-owner DB does not contain that imported name.
- Compile4 Python files passed. Ruff baseline/current:3 inherited import-order findings (2 endpoint,1 existing V3 test);0 production/new parser findings. New integration test C420 fixture comprehension corrected to equivalent `dict.fromkeys`; focused integration repeat:4 passed,no skips,457 warnings,24.56s pytest/30.624s outer,natural0. Final compile4/Ruff confirmed0 new findings.
- Baseline/current Bandit:19 inherited LOW/HIGH B101 assertions in existing V3 tests only;0 production findings,0 new findings,0 scan errors. Whitespace diff clean.
- Source ruling was sent to root before production edits. Validation compatibility is narrowly limited to null optional text; required fields and null/wrong-type collections remain invalid. New V3 export strings reuse existing V2/PNG defaults. No network/spec retrieval permitted; this is explicit compatibility for the server's already-produced files, not a claim that null text is valid interchange syntax.
- Independent review and actual B01 live retry are root-owned and pending; source qualification does not establish live loaded-runtime success, full-app/UAT matrix, other auth/DB profiles, or UI results. No runtime/Git publication work performed.

## Freeze and review handoff

Root completed connected-source independent review with no substantiated Critical/Important/Minor findings and requested production freeze. Production source is frozen after equivalent test-only C420 cleanup and its affected-module qualification. Stage3 remains In Progress solely for root-owned actual downloaded-input/live B01 validation and final task evidence. Existing invalid explicit-V3 fallback diagnostics are unchanged and outside this nullable-text correction. No fixture failures occurred in red/green; no guards/plugins were weakened or skipped.
