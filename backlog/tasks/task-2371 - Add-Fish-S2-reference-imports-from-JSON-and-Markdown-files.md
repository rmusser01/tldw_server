---
id: TASK-2371
title: Add Fish S2 reference imports from JSON and Markdown files
status: Done
labels:
  - tts
  - fish-s2
  - api
modified_files:
  - tldw_Server_API/app/core/TTS/fish_s2_reference_imports.py
  - tldw_Server_API/app/api/v1/endpoints/audio/audio_voices.py
  - tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_reference_imports.py
  - tldw_Server_API/tests/TTS_NEW/integration/test_fish_s2_reference_endpoints.py
  - Docs/STT-TTS/TTS-SETUP-GUIDE.md
  - Docs/superpowers/plans/2026-06-27-fish-s2-reference-imports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing Fish Audio S2 TTS PR with an import endpoint for managed Fish S2 references from JSON and Markdown files that match the reference metadata format. Reuse the existing `create_fish_s2_reference` service path so local voice metadata and remote Fish references remain consistent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add JSON and Markdown import support for Fish S2 managed references.
- [x] Reuse the existing `create_fish_s2_reference` service path for imported items.
- [x] Document accepted JSON and Markdown file shapes.
- [x] Focused Fish S2 tests and Bandit pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a core parser for `.json`, `.md`, and `.markdown` Fish S2 reference import files. Added `POST /api/v1/audio/providers/fish_s2/references/import`, with coverage for existing-voice JSON, bulk JSON, embedded base64 audio JSON, Markdown frontmatter, and invalid imports. The endpoint returns per-item results/errors and preserves the existing voice upload scope/rate-limit path.

Reopened for code-review fixes: restore native Fish streaming format constraints, harden import partial-error behavior, and add import size/item/audio limits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Fish S2 reference imports from JSON and Markdown files on the existing Fish S2 commercial API branch/PR. Verification: pytest Fish S2 focused suite passed with 57 selected tests and 32 deselected; Bandit on touched production Python files reported zero findings; `git diff --check` passed.

Code-review fixes added native Fish streaming format enforcement, indexed partial import errors/results, and file/item/decoded-audio limits. Verification: pytest Fish S2 focused suite passed with 66 selected tests and 32 deselected; Bandit on touched production Python files reported zero findings; `git diff --check` passed.
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
