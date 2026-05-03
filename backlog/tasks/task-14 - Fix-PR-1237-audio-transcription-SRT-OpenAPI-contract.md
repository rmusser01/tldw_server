---
id: TASK-14
title: Fix PR 1237 audio transcription SRT OpenAPI contract
status: Done
assignee: []
created_date: '2026-05-03 20:36'
updated_date: '2026-05-03 20:39'
labels:
  - openapi
  - audio
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1237'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the PR #1237 review finding where the audio transcription 200-response description mentions SRT but the OpenAPI 200-response content map does not expose an SRT-specific media type. The endpoint already supports response_format=srt, so align the generated contract and focused tests with runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generated OpenAPI for /api/v1/audio/transcriptions and /api/v1/audio/translations documents an SRT media type when SRT is described as supported.
- [x] #2 Focused contract tests/constants cover the SRT content type for both OpenAI-compatible audio routes.
- [x] #3 Runtime SRT responses use the documented SRT media type or the description is adjusted to match actual runtime support.
- [x] #4 Focused pytest verification and touched-scope Bandit checks are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify runtime SRT support and current OpenAPI response content. 2. Add failing contract/runtime assertions for SRT media type. 3. Update endpoint OpenAPI content and SRT response media type. 4. Run focused pytest, OpenAPI verifier, diff check, and Bandit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification before implementation: focused pytest failed because generated OpenAPI lacked application/x-subrip and response_format=srt returned text/plain; charset=utf-8.

Implementation: added application/x-subrip to the shared audio transcript response content map and changed response_format=srt responses to return application/x-subrip. Updated OpenAPI contract constants and SRT runtime test coverage.

Verification: focused red run failed before implementation for missing application/x-subrip and text/plain SRT runtime response. Green checks: 73 related pytest tests passed, apps/packages/ui and apps/extension verify:openapi passed with existing reviewed exceptions, git diff --check passed, Bandit on audio_transcriptions.py reported zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the OpenAI-compatible audio transcription/translation response contract to include the SRT-specific application/x-subrip media type, matching documented response_format=srt support. The SRT runtime response now returns application/x-subrip instead of generic text/plain, and focused tests cover both generated OpenAPI content and actual SRT response headers. Verification covered the focused red-green regression, the broader OpenAPI/audio timed-segment pytest files, both frontend OpenAPI drift verifiers, git diff whitespace checks, and Bandit on the touched backend file.
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
