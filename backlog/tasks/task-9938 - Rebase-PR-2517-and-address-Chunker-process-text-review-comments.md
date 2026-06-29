---
id: TASK-9938
title: Rebase PR 2517 and address Chunker process_text review comments
status: In Progress
labels:
- pr-review
- chunking
- process-text
priority: High
ordinal: 12059
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2517 on latest dev, inspect active review comments, implement verified Chunker process_text fixes, run focused verification, push the updated branch, and resolve addressed review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR #2517 branch is rebased onto latest origin/dev without unresolved conflicts.
- [ ] #2 All active review comments are inventoried, technically verified, and either fixed or answered with rationale.
- [ ] #3 Focused tests, compile checks, shard coverage or relevant CI guard checks, and Bandit on touched production code are run as applicable.
- [ ] #4 Updated branch is pushed and review threads are replied to/resolved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/chunker-process-text-refactor` onto latest `origin/dev` (`c6d2191c5b`) without conflicts.
- Inventoried active PR #2517 review threads:
  - Gemini: add explicit `None` handling in `process_text/dispatch.py` metadata conversion.
  - Gemini: use normalized `method_lower` for adaptive and multi-level method membership checks in `process_text/options.py`.
  - Qodo: add missing docstrings to new Chunker/process_text modules and `_process_text_telemetry_hooks`.
  - Qodo: add approved pytest markers to new tests.
  - Qodo: loosen equivalence tests that asserted internal call paths instead of behavior.
- Implemented the review fixes and added regression coverage for normalized method aliases and `None` metadata.
- Verification run before commit:
  - `python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q` -> `81 passed`.
  - `python -m ruff check ...` on touched Chunking modules/tests -> passed.
  - `git diff --check` -> passed.
  - `python -m compileall` on touched production modules -> passed.
  - `python -m bandit -r ... -f json -o /tmp/bandit_pr2517.json` -> 0 findings.
  - `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` -> passed (`new_uncovered=0`).
  - Full `tldw_Server_API/tests/Chunking` run under sandbox passed 402 tests with 4 skips and failed 6 tokenizer tests due blocked `huggingface.co` DNS; rerunning those 6 failures with network access passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Backlog task and implementation plan are updated with verification evidence.
- [ ] #2 Code changes are committed with a clear message.
- [ ] #3 No unrelated dirty files are staged or modified.
- [ ] #4 PR branch is pushed to GitHub and remaining check status is reported.
<!-- DOD:END -->
