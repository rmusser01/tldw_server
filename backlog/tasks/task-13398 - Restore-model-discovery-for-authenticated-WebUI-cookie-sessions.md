---
id: TASK-13398
title: Restore model discovery for authenticated WebUI cookie sessions
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Buddy UAT reaches a healthy cookie-authenticated WebUI connection but the shared model service requires a token or API key, leaving model selection empty. Restore authenticated model discovery while retaining existing exact-origin authentication limits.

## Implementation Plan

1. Reproduce with the existing model-service test.
2. Reuse the canonical active-cookie-session guard and separate cookie cache scope.
3. Verify targeted catalog/auth regressions and real fresh/upgraded WebUI model selection.

ADR required: no
ADR path: N/A
Reason: routine correction honoring existing authentication authority; no new storage, provider or security boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exact-origin quickstart single-user cookie session discovers selectable server models without an API key.
- [x] #2 Cross-origin, advanced-deployment, multi-user without token and missing-auth-source configurations stay denied.
- [x] #3 Cookie model caches use a distinct scope and existing key/token caching regressions pass.
- [x] #4 Fresh and upgraded real WebUI sessions select a model and receive the expected reply.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the shared catalog admission fix by reusing isActiveCookieSessionConfig and separating cookie cache scope. Red: the real guard rejected the valid same-origin cookie session (35 passed, 1 failed). Green: 43 targeted model-service/browser-network cases passed after the fix, including cross-origin, advanced, multi-user and missing-auth negative cases. Scoped ESLint reports zero errors with a pre-existing inputMods warning; git diff --check passes. Real fresh and v65-to-v73 upgraded WebUI sessions discovered DeepSeek models and returned expected replies; workspace defaults, ownership and mode persistence were exercised through the UI. Evidence remains private at /private/tmp/buddy-all-uat-20260930/uat-report.md; no raw logs published. ADR required: no; existing cookie guard and authentication authority reused. Bandit is not applicable to the two touched TypeScript files. Task remains In Progress pending review/publication and canonical acceptance-criteria population (Python CLI has no add-AC option).
September 30 final UAT self-review: the shared catalog repair leaves the canonical exact-origin cookie guard and multi-user token admission intact. Exact two-file review patch retained privately at /private/tmp/buddy-all-uat-20260930/cookie-model-catalog.patch (SHA256 667f83f1d596076c6848c97933b00522e58caaee4ff8836bb65ec0b07b08881f). Current merged Chatbook synthetic realtime request succeeded after selecting its private profile before imports; no recovery guard was weakened. Native Terminal Computer Use is explicitly unavailable. No new provider request, raw log publication or Done status is claimed.
Canonical AC repair: the thin Python CLI cannot add criteria; recreated this still-unpublished owned task through its existing Python repository create API, retaining all notes and DoD items. Later section edits continue through the Python CLI. The original inline plan is now a distinct Markdown heading in the CLI-owned description. Status remains In Progress pending review.
Approved publication follow-up: isolated branch codex/buddy-cookie-model-uat-followup starts on actual dev f3f1b4fdbe3fe461b371ece30887c5fff8476d9d. Affected catalog/auth sources and both existing UAT task records were unchanged on the incoming base. Final targeted run passes 43 cases in 1.90s; scoped ESLint zero errors with one existing warning, diff checks pass. Sanitized source-hashed receipt: Docs/Reviews/2026-09-30-buddy-cookie-model-uat.md. Canonical AC now contains four checked outcomes; task stays In Progress for hosted review/checks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored selectable model discovery for exact-origin quickstart cookie sessions by reusing existing authentication admission and separating cookie cache scope. Targeted and real-browser evidence passes; review/hosted gates remain pending. Broader human/native and historical-trigger UAT remains open in its existing tasks.
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
