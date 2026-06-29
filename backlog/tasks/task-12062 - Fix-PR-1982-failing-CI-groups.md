---
id: TASK-12062
title: Fix PR 1982 failing CI groups
status: Done
assignee:
  - Codex
created_date: ''
updated_date: '2026-06-29 15:11'
labels:
  - ci
  - pr-1982
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current failing GitHub Actions checks on PR #1982 with minimal targeted fixes. Initial groups: README release docs contract, setup audio install-plan payload shape, audio voices auth dependency contract, ChromaDB dimension mismatch recreation, embeddings missing-model policy, privilege registry snapshot, Windows public profile env parsing, TTS env isolation and missing credentials 503, sandbox concurrency cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current identified PR #1982 CI failures are addressed with targeted code, docs, and tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented minimal fixes for the current PR #1982 failing CI groups:
- Restored README release status wording expected by the release docs contract.
- Sent setup install plans to background execution without the false default trusted_custom_model_acknowledged flag when no custom embedding models are requested.
- Switched audio voice routes back to TokenScopeGuard usage and regenerated the privilege route registry snapshot.
- Recreated ChromaDB target collections on verified embedding dimension mismatch so model dimension changes recover instead of failing the shard.
- Kept the embeddings missing-model policy error code stable as model_denied.
- Removed non-TTS ANTHROPIC_API_KEY leakage from TTS config env overrides.
- Preserved BYOK resolver monkeypatch compatibility while resolving missing TTS credentials to the expected 503 path.
- Made sandbox store backend/path resolution prefer process env over stale settings attrs and added regression coverage.
- Quoted parsed Docker env-file values and sourced the generated exports so Git Bash preserves DATABASE_URL/JOBS_DB_URL overrides containing equals signs.

Verification:
- Focused regression pytest set: 30 passed, 17 warnings.
- CI-style sandbox shard with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1: 16 passed, 5 warnings.
- bash -n Dockerfiles/entrypoints/tldw-app-first-run.sh: passed.
- Bandit on touched Python app paths wrote /tmp/bandit_pr1982_ci.json with 0 findings and 9 existing nosec skips.
- git diff --check: passed.

Known skips/blockers:
- The macOS CI sandbox failure is Python 3.12-specific in Actions; local verification used the project Python 3.11 venv because no usable local Python 3.12 pytest environment is installed.
- Two untracked watchlist template files are present in the worktree but are unrelated to this task and intentionally not staged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared and verified targeted fixes for the current PR #1982 CI failures covering release docs, setup install-plan queuing, audio auth/BYOK behavior, ChromaDB embedding dimension recovery, embeddings policy codes, TTS env isolation, sandbox store env precedence, Docker env-file parsing, and the privilege route snapshot.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
