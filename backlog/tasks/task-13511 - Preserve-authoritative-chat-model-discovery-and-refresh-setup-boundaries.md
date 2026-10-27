---
id: TASK-13511
title: Preserve authoritative chat model discovery and refresh setup boundaries
status: Done
labels:
- chat
- ux
- bug
- frontend
dependencies:
- TASK-13454
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved NN/G chat remediation on latest dev1fc353. Owned browser advertises No chat models configured while live metadata returns enabled custom_openai_api/gpt-4o-mini. Shared discovery turns failures into empty catalogs and client caches survive configuration/account changes. Fix actual catalog readiness and lifecycle across shared WebUI/extension callers without weakening model validation or replaying admitted turns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery distinguishes authoritative empty from failed or stale reads while retaining supported offline cache behavior.
- [x] #2 Setup and account changes retire old results and refresh the visible shared chat catalog.
- [x] #3 Dispatch does not treat a Current placeholder or failed discovery as verified model readiness.
- [x] #4 Failure-first shared-boundary regression tests, typecheck, extension compile and independent review pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Separate completion400 transport hypothesis remains unproven; no CORS relaxation or silent replay is authorized by this task.
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_authoritative_catalog_13511.md. Shared service now distinguishes strict failures from authoritative empty/advisory fallback, scopes caches to authorization, fences retired network and early config lookups, and preserves same-owner raw in-flight discovery. Shared QueryClient catalog follows setup/account invalidations and masks retained data during failed/background reads. Picker/readiness/dispatch/setup consumers reuse it; explicit Retry added to main/compact, Models refresh, onboarding, Quick Chat helper/popout and character generator. Root regressions fail before corrections, including mixed error policies, cookie join and 3 early config retirement cases. Both independent source reviews clear bounded fixes. Final regression/types/lint/locale checks are being recorded before finalization. No Python source changed: Bandit N/A. Owned IAB roleplay sent one synthetic turn, reopened with three rows; read-only raw server messages preserve exact greeting placeholders/outer whitespace. Blank display-name fallback is a Stage3 presentation follow-up. Earlier standard completion400 remains unproven; no replay or CORS change.
Final18suites296pass; shared types and extension compile exit0; no added touched lint; official locale mirrors limited to2 English keys and other bytes restored; diff check clean. Both source reviewers cleared root fences and Retry paths. Final evidence and remaining parent limitations are in the master design. The completed temporary stage plan is retired after verification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed authoritative discovery/error semantics and cache/query ownership at their shared roots; reused across WebUI/extension readiness, selection, dispatch and setup. Completed independent review and 296 related checks. Owned roleplay saved and reopened; no claim that unproven standard transport or remaining all-ten/native/D7 acceptance is complete.
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
