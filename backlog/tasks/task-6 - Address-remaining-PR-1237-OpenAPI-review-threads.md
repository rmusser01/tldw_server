---
id: TASK-6
title: Address remaining PR 1237 OpenAPI review threads
status: Done
assignee:
  - Codex
created_date: '2026-05-03 18:37'
updated_date: '2026-05-03 18:43'
labels:
  - pr-review
  - openapi
  - phase4
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1237'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567209'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567210'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567211'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567212'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567216'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567219'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178567220'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the additional CodeRabbit review threads on PR #1237 by verifying each OpenAPI contract/tagging finding against runtime code and applying narrow contract fixes where valid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAPI contracts for audio speech, chat document generation, quickstart fallback, HAL raw formats, and VN asset content match reachable runtime response media types.
- [x] #2 OpenAPI tag normalization ignores malformed non-sequence tag values instead of expanding strings into single-character tags.
- [x] #3 Public control-plane routes retain the health tag in the generated OpenAPI schema.
- [x] #4 Focused contract tests, diff check, and Bandit verification are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect each reviewed endpoint/helper against current runtime code and existing contract tests.
2. Add or update focused OpenAPI contract tests that encode the valid reviewer findings before changing production code.
3. Apply minimal response-contract/tagging fixes in the reviewed files.
4. Run focused pytest coverage for the changed contract tests plus git diff --check and Bandit on touched backend files.
5. Commit, push to codex/phase4-openapi-contract-testing, and resolve the addressed review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented validated CodeRabbit follow-ups: generic PCM OpenAPI media type for audio speech and reading TTS, explicit chat document JSON/SSE response content, explicit quickstart HTML fallback response, HAL Atom/RSS media types, VN asset image media types, hardened operation tag extraction for malformed scalar tags, and restored health tags on public control-plane routes. Red test run before production edits failed 7 expected assertions; final focused run passed 12 tests with 19 warnings. git diff --check passed. Bandit on touched backend/test files passed with B101 skipped for pytest asserts and zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the remaining PR #1237 OpenAPI review threads with narrow contract and tag fixes, plus focused regression coverage for the media-type and tagging gaps.
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
