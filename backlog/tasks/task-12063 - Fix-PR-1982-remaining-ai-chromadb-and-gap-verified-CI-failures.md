---
id: TASK-12063
title: Fix PR 1982 remaining ai-chromadb and gap-verified CI failures
status: Done
assignee:
  - Codex
created_date: ''
updated_date: '2026-06-29 19:42'
labels:
  - ci
  - pr-1982
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and address the remaining failed PR #1982 CI checks on commit d480942a408081248b24a12ee724ddac6f8e0714, specifically the ai-chromadb and gap-verified-3 full-suite shards plus related aggregate failures. Keep changes minimal, verify locally where feasible, run Bandit for touched Python code, and push fixes to the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remaining ai-chromadb and gap-verified-3 PR #1982 shard failures are reconciled with current code and tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigated failed PR #1982 check runs for commit d480942a408081248b24a12ee724ddac6f8e0714. Downloaded direct logs for ai-chromadb on Ubuntu 3.12/3.13, macOS 3.12, and Windows 3.12 plus gap-verified-3 on Ubuntu 3.12/3.13.

Root causes:
- ai-chromadb failed because two Chroma dimension contract tests still expected mismatch writes against existing/unknown-size collections to raise ValueError without deleting or upserting, while the prior patch recreated every mismatched collection.
- gap-verified-3 failed because missing embedding model validation should remain model_required rather than model_denied; model_denied is reserved for policy allowlist denial and maps to a different HTTP status.

Fixes:
- ChromaDB store_in_chroma now recreates a mismatched collection only when collection.count() is confirmed to be integer 0. Populated or unknown-size collections fail closed with ValueError and do not delete existing data.
- resolve_provider_model now reports missing required models as model_required again.
- Updated the newer provider-resolution test expectation to match the canonical missing-model validation contract.

Verification:
- Focused red set before fix: 3 failed, 4 passed.
- Focused red set after fix: 7 passed, 5 warnings.
- CI-style ai-chromadb path: 89 passed, 3 skipped, 15 xfailed, 2 xpassed, 7 warnings.
- CI-style Embeddings_isolated path: 117 passed, 7 warnings.
- CI-style gap-verified-3 full path set: 462 passed, 13 warnings.
- Bandit on touched Python app files: 0 findings.
- git diff --check: passed.

Known skips/blockers:
- Local verification is on the project Python 3.11 venv; the failing CI matrix used Python 3.12/3.13. The failed contracts reproduce locally on 3.11 and now pass locally.
- Existing unrelated untracked watchlist template files remain unstaged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the remaining PR #1982 ai-chromadb and gap-verified-3 failures by restoring missing-model validation to model_required and making Chroma dimension mismatch recovery non-destructive except for confirmed-empty collections. Verified with the focused red set and both CI-style shard path sets locally.
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
