---
id: TASK-304.11
title: Design Research Studio capability health contract PR follow-up
status: Done
assignee: []
created_date: '2026-05-13 01:21'
updated_date: '2026-05-13 01:26'
labels:
  - design
  - research-studio
  - webui
  - backend
  - verification
dependencies: []
documentation:
  - Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the PR-scoped follow-up for Research Studio capability-aware health and authenticated live-backend verification. The agreed direction is backend-owned capability status for source browsing, chat, text artifact generation, slides, audio summary, export/download, and sync/share; frontend action-boundary gating must consume that contract rather than infer from raw subsystem health. The PR must also require local/manual CDP verification against a valid authenticated backend and at least one real text artifact generation using locally saved existing LLM credentials, with evidence recorded in Backlog and the PR body.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec defines a backend-owned Research Studio capability health contract and stable status/mode semantics.
- [x] #2 Design spec defines frontend action-boundary gating for read-only browsing, chat, text artifacts, slides, audio summary, export/download, and sync/share.
- [x] #3 Design spec requires local/manual authenticated CDP verification using a valid backend API key.
- [x] #4 Design spec requires real end-to-end text artifact generation using existing locally saved LLM credentials, and explicitly keeps this verification out of required CI.
- [x] #5 Design spec defines evidence recording expectations for Backlog and PR body, including provider, source, artifact type, result, and caveats.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md. The spec defines a backend-owned Research Studio capability endpoint, stable status/mode semantics, frontend action-boundary gating, authenticated CDP verification, and local/manual real summary artifact generation using existing saved LLM credentials.

Verification: git diff --check passed for the docs/task-only change. Bandit not run because no backend Python or executable code changed in this design slice.

Spec review note: subagent review was not dispatched because this Codex session policy only permits spawning subagents when explicitly requested by the user. Performed an inline self-review against the approved design sections and current route/health evidence instead.

2026-05-13 design review reopened the task to patch issues found before implementation planning: endpoint auth/rate-limit ownership, stale capability payload handling, source-selection versus capability-gating precedence, and local manual generation evidence boundaries.

2026-05-13 design review patches applied: endpoint must be authenticated/rate-limited, implemented through lightweight local collectors rather than HTTP calls; contract now includes status/mode semantics table and ttl_seconds; frontend refreshes stale payloads and gives no-source guidance precedence; export/download distinguishes browser-local from backend-dependent exports; real generation evidence avoids recording full generated text and treats missing saved credentials as a PR verification blocker.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented and review-hardened the approved backend-owned Research Studio capability contract design, including endpoint auth/sanitization, TTL refresh behavior, source-gate precedence, action-boundary gating, and local/manual real LLM summary generation using existing saved credentials.
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
