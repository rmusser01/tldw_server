---
id: TASK-12795
title: Harden StudyPacks review findings
status: Done
assignee: []
created_date: '2026-06-24 00:00'
updated_date: '2026-06-25 02:07'
labels:
  - study-packs
  - review-hardening
dependencies: []
references:
  - tldw_Server_API/app/core/StudyPacks
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and remediate validated StudyPacks review findings around provenance integrity, evidence bounds, regeneration locator preservation, JSON response strictness, deck-name allocation behavior, and import coupling discovered during validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings have focused red-to-green regression coverage.
- [x] #2 StudyPacks source evidence and citation provenance cannot be silently fabricated from caller/model text.
- [x] #3 StudyPacks prompt/persistence evidence size is bounded.
- [x] #4 Regeneration preserves source locator context.
- [x] #5 JSON output parsing, deck-name allocation, and heavy import coupling are hardened where validated.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Manual Backlog task file fallback approved by the user after the Backlog MCP workflow was unavailable and repeated non-interactive `backlog task create` attempts hung without output.

Implemented validated fixes:
- Source resolver only accepts `excerpt_text` when it matches resolved note/media/message evidence and bounds evidence with `STUDY_PACK_MAX_EVIDENCE_CHARS_PER_SOURCE`, marking truncated locators.
- Generation validation now requires model `citation_text` to match bundle evidence, rejects embedded/trailing JSON, checks deck-name candidates with exact lookup, and no longer imports the Workflows adapter helper.
- Regeneration source extraction preserves labels, locators, and persisted evidence as an excerpt hint when available.

PR #2503 rebase/review follow-up:
- Rebased `codex/studypacks-review-hardening` onto latest `origin/dev` (`e664332b682e9be4e1f89d05a262de155cebfa6e`).
- Addressed Qodo review comments by documenting new helpers, removing broad exception swallowing from `extract_openai_content`, adding pytest unit markers, bounding regeneration excerpt hints with truncation metadata, returning a controlled 400 when total regenerate job payloads still exceed the Jobs cap, and caching deck-name fallback scans.

Modified files:
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
- `tldw_Server_API/app/core/StudyPacks/generation_service.py`
- `tldw_Server_API/app/core/StudyPacks/jobs.py`
- `tldw_Server_API/tests/StudyPacks/test_source_resolver.py`
- `tldw_Server_API/tests/StudyPacks/test_generation_service.py`
- `tldw_Server_API/tests/StudyPacks/test_study_pack_jobs.py`
- `tldw_Server_API/tests/StudyPacks/test_generation_service_imports.py`
- `tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py`
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and fixed all confirmed StudyPacks review issues. Verification completed with:
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/StudyPacks -q` -> 74 passed, 168 warnings.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/StudyPacks tldw_Server_API/app/api/v1/endpoints/flashcards.py -f json -o /tmp/bandit_study_packs_12009_review_rebase.json` -> no findings.
- `git diff --check` on touched StudyPacks files -> clean.

No known blockers or skipped validations remain for this task.
<!-- SECTION:FINAL_SUMMARY:END -->
