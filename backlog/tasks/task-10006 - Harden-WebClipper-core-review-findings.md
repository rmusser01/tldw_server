---
id: TASK-10006
title: Harden WebClipper core review findings
status: Done
assignee: []
created_date: '2026-06-23'
updated_date: '2026-06-23'
labels:
  - web-clipper
  - security
  - backend
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current-code WebClipper core review findings: prevent client `clip_id` collisions from claiming existing notes, harden clipper-created attachment safety, make core note lookups backend-safe, add practical request payload bounds, and reduce the most direct API/core coupling where it can be done without broad refactoring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing non-clip notes cannot be overwritten through WebClipper `clip_id` collisions.
- [x] #2 Clipper attachments cannot persist active HTML/SVG content that is later served inline.
- [x] #3 WebClipper note and workspace-note lookups use backend-safe deleted predicates.
- [x] #4 Clipper request payloads have practical body, metadata, keyword, and attachment bounds.
- [x] #5 Focused WebClipper tests and Bandit on touched Python scope are run, or blockers are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan file: `IMPLEMENTATION_PLAN_webclipper_review_hardening_10006.md`

Initial scope:
1. Add failing regression tests for the review findings.
2. Update WebClipper schema and service helpers for note-claim, attachment, payload, and backend-safety behavior.
3. Run focused tests, Bandit, and diff hygiene; record results here.

Backlog MCP was unavailable in this session and `backlog task create` hung without output twice, so this task file was created manually after user approval.

Implemented:
- Moved WebClipper schema definitions into `tldw_Server_API.app.core.WebClipper.schemas` and kept the API schema module as a compatibility re-export.
- Added schema bounds for clip IDs, titles, URLs, body fields, keywords, attachment counts/content, capture metadata JSON, and enrichment payloads.
- Rejected client saves that try to claim an existing note without an existing WebClipper sidecar.
- Removed active HTML/SVG attachment support and blocked `.htm`, `.html`, and `.svg` clipper attachment filenames.
- Replaced SQLite-only WebClipper note/workspace-note deleted predicates with backend-native deleted values.

Verification:
- RED: focused new WebClipper regression selection failed with 7 failures before implementation.
- GREEN: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_endpoint_error_mapping.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py -q` -> 56 passed, 125 warnings.
- Black: `python -m black --check` on touched Python files -> 4 files unchanged.
- Bandit: `/tmp/bandit_webclipper_10006.json` -> 0 results, 0 errors, 0 skipped.
- Diff hygiene: `git diff --check` on touched scope -> passed.
- Clean worktree PR verification on `codex/webclipper-review-hardening-10006`: focused WebClipper suite -> 56 passed, 124 warnings; Black -> 4 files unchanged; Bandit `/tmp/bandit_webclipper_10006_worktree.json` -> 0 results, 0 errors, 0 skipped; `git diff --check` and `git diff --cached --check` passed.
- PR review follow-up:
  - Rebased the branch onto latest `origin/dev`.
  - Moved clipper sidecar lookup into the `save_clip()` transaction before collision decisions.
  - Enforced the capture metadata JSON bound after service-level metadata merge.
  - Removed manual SQL preparation from `_fetch_note_row()` so backend-aware transaction wrappers prepare once.
  - Replaced the private-helper test with public `save_clip()` regression coverage for stale pre-transaction sidecar state and merged metadata bounds.
  - Focused service unit run after the follow-up: `python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py -q` -> 25 passed, 57 warnings.
  - Final post-format focused WebClipper suite: `python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_endpoint_error_mapping.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py -q` -> 57 passed, 123 warnings.
  - Final Black check on touched Python files -> 4 files unchanged.
  - Final Bandit report `/tmp/bandit_webclipper_2496_final.json` -> 0 results, 0 errors, 0 skipped.
  - Final diff hygiene: `git diff --check` and `git diff --check origin/dev...HEAD` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened WebClipper save/enrichment contracts against the reviewed collision, attachment, backend portability, payload-boundary, and core/API coupling findings. Focused WebClipper service/API/DB verification passed, Bandit reported zero findings on touched source, and diff hygiene passed.
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
