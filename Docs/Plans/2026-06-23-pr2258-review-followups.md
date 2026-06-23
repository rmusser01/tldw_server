# PR 2258 Review Follow-ups Implementation Plan

Backlog: TASK-2397

## Stage 1: Audit Review Feedback
**Goal**: Verify which unresolved PR #2258 comments remain valid on the current `dev`-based follow-up branch.
**Success Criteria**: Inline and outside-diff comments are categorized as fixed, still-valid, or intentionally skipped with rationale.
**Tests**: GitHub review-thread/comment extraction and local code inspection.
**Status**: Complete

Notes:
- All unresolved inline Gemini/CodeRabbit comments and outside-diff CodeRabbit comments were reviewed against the current branch.
- Qodo's DB URL exposure concern remained valid and is addressed in Stage 2 with GitHub log masks before env export.
- Qodo's PR full-suite and OS Postgres auto-start concerns are obsolete in the current branch: macOS/Windows full-suite shards run on PRs when backend files changed and set `TLDW_TEST_NO_DOCKER=1`.

## Stage 2: CI, Config, And Documentation Fixes
**Goal**: Address workflow/config/documentation findings that do not require application runtime changes.
**Success Criteria**: Redis service images are pinned, public frontend envs do not fall back to server secrets, shard coverage invocation is explicit, setup-ffmpeg no-ops when disabled, duplicate task headings are resolved, and PostgreSQL URL exports are masked before writing to `GITHUB_ENV`.
**Tests**: YAML parsing, targeted workflow contract tests where available, and `git diff --check`.
**Status**: Complete

## Stage 3: Runtime Correctness And Lifecycle Fixes
**Goal**: Address still-valid application code findings around FTS updates, sync state isolation, message ordering, SQLite rollback, Chroma retry behavior, evaluation service cleanup, analytics bootstrap validation, persona WebSocket ordering, and APKG connection cleanup.
**Success Criteria**: Runtime paths are corrected without broad refactors and with focused regression coverage where practical.
**Tests**: Targeted pytest modules for touched runtime areas plus compile checks.
**Status**: Complete

## Stage 4: Test Isolation And Assertion Hardening
**Goal**: Address test-only findings that can mask regressions or leak state across modules.
**Success Criteria**: Fixtures restore env/state, tests assert the intended boundaries, and skip conditions only cover explicit environment failures.
**Tests**: Targeted pytest modules for touched tests.
**Status**: Complete

## Stage 5: Final Verification And PR Update
**Goal**: Verify the complete follow-up change set and update PR #2431.
**Success Criteria**: Relevant tests pass without `--maxfail`, Bandit is run on touched Python code, task notes summarize results, and the branch is pushed to PR #2431.
**Tests**: Targeted pytest, compileall on touched Python files, Bandit on touched Python scope, YAML parse/contract checks, `git diff --check`, and fresh PR check polling after push.
**Status**: In Progress

Verification:
- `python -m py_compile` on touched Python files passed.
- `python -m pytest -q tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_e2e_required_redis_contract.py`: 36 passed.
- Focused backend regression suite covering API deps, collections, DB/cache/media, evaluations webhook, sync, media import resilience, personalization, and RAG: 81 passed, 1 skipped.
- APKG/media guard/embeddings verification: 64 passed, 14 skipped. The real HuggingFace embedding tests are now explicit opt-in via `RUN_REAL_HF_EMBEDDING_TESTS=true`.
- `bunx vitest run __tests__/frontend-quickstart-networking.test.ts`: 11 passed.
- `git diff --check` passed.
- Bandit on touched production Python and touched tests exited 0.
