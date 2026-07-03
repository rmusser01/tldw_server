## Stage 1: Backend release and WebSearch review fixes
**Goal**: Address valid release-helper, WebSearch logging, Sync path-validation, media navigation CodeQL comment placement, and test-marker feedback.
**Success Criteria**: Release helper raises when the post-release README anchor is absent; WebSearch diagnostic logs are redacted and lazy; Sync temp path inputs are validated at the helper boundary; CodeQL comments sit adjacent to alerting regex calls; touched backend tests have pytest markers.
**Tests**: Targeted pytest for release docs, WebSearch sanitizer/logging, Sync blob store boundary checks, and media navigation where practical.
**Status**: Complete

## Stage 2: MCP docs corpus review fixes
**Goal**: Resolve docs package correctness/configuration comments with local, minimal changes.
**Success Criteria**: Text extraction has no duplicate branch; NormalizedURL default construction stores a string canonical URL; locked_down policy rejects ignored allow settings; IP-literal URL prefixes fail fast; HTML inline whitespace is preserved; non-UTF-8 local imports raise DocsError; SQLite connections use timeout/WAL and PRAGMA identifier allowlists; keyword orphan pruning is moved out of per-document writes; public host adapter methods have docstrings; default docs module respects robots.
**Tests**: Targeted MCP docs pytest modules for acquisition, importers, schema store, standalone mount, and host adapter.
**Status**: Complete

## Stage 3: Frontend correctness and test hygiene
**Goal**: Fix frontend review findings that affect runtime correctness or test reliability.
**Success Criteria**: Audio recorder clears recorder ref on synchronous start failure; greeting selection cannot double-post while persistence is in flight; character stream watchdog aborts only its owned controller; persona connect timeout is cleared on unmount; route character intent signatures are scoped to resolved intent fields; brittle/leaky tests are hardened; standalone UX gate stages Next static assets.
**Tests**: Targeted Vitest suites for useAudioRecorder, ChatGreetingPicker, character chat watchdog, usePersonaLiveControl, Playground route intent, TTS drawer test, WebLayout contract, and plasmo storage watch.
**Status**: Complete

## Stage 4: CodeQL review annotations and PR metadata
**Goal**: Address CodeQL PR review comments without changing test semantics or leaking secrets.
**Success Criteria**: Test-only URL substring and credential-storage alerts are either fixed with stronger assertions/helpers or annotated directly at test-only sinks; the human-authored PR change-summary gate is documented as a requester action; remaining broad "consider" comments are triaged as intentionally out of scope when they are not tied to a current failure.
**Tests**: Relevant CI/unit tests plus static search for moved CodeQL annotations.
**Status**: Complete

**Notes**: The concrete CodeQL/test issues were addressed. Broad low-priority "consider extracting shared helper" suggestions were intentionally not folded into this release-merge cleanup because they would refactor unrelated UI areas without fixing a current behavioral failure. The PR's human-authored `Change summary` gate remains a requester action per `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`.

## Stage 5: Verification and finalization
**Goal**: Verify touched scope, update Backlog.md, commit, and push to the PR branch.
**Success Criteria**: Targeted Python and frontend tests pass or any blocker is documented; Bandit runs on touched Python scope; Backlog task includes touched files, verification, and final summary; changes are committed and pushed to update PR #2571.
**Tests**: `python -m pytest ...`, `bunx vitest run ...`, Bandit for touched Python paths, plus `git status`/diff review.
**Status**: Complete

**Verification Results**:
- Backend focused pytest: 25 passed, 5 warnings.
- MCP docs focused pytest: 7 passed, 3 warnings.
- MCP docs schema-store pytest: 17 passed, 3 warnings.
- Frontend focused Vitest suite: 7 files passed, 90 tests passed.
- Persona live follow-up Vitest: 1 file passed, 12 tests passed.
- `git diff --check`: clean.
- Bandit touched Python scope: completed with existing low-severity baseline findings in untouched `WebSearch_APIs.py` lines only (`B311`, `B101`); no new findings in changed lines.
