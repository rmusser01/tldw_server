---
id: TASK-326
title: Design VN script authoring graph API
status: Done
assignee: []
created_date: '2026-05-14 01:32'
updated_date: '2026-05-14 01:45'
labels:
  - vn
  - scripts
  - design
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1610'
  - 'https://github.com/rmusser01/tldw_server/pull/1641'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a backend-first design spec for a computed VN script authoring graph API under the existing VN Scripts surface. The spec must capture the approved choices: static graph before dry-run execution; support stored drafts, supplied drafts, and published versions; return both outline and detailed graph layers; include graph diagnostics plus existing validation diagnostics; compute on demand with no persistence; use strict statically knowable edge semantics; expose stable IDs, bracket JSON paths, compact op summaries, content hashes, and a script_authoring_graph capability flag. The design should include the critique amendments about validator reachability reuse, conservative terminal semantics, supplied-draft limits, content-hash definition, non-mutating validation, pinned published-version context, malformed-draft tolerance, and route naming.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs with the approved V1 graph API contract and critique amendments included.
- [x] #2 Spec defines routes, response schemas, graph/outline data model, diagnostics, error behavior, security boundaries, limits, hashing, validation context, and non-goals.
- [x] #3 Spec includes testing and rollout guidance for a backend-only first sprint.
- [x] #4 Spec receives a review pass and any identified issues are addressed before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the design spec in the isolated worktree on branch `codex/vn-script-authoring-graph-design`.
2. Review the spec against current VN script validator/service/API behavior on `origin/dev`.
3. Patch any design gaps found in review.
4. Run documentation hygiene checks and record verification.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created `Docs/superpowers/specs/2026-05-14-vn-script-authoring-graph-design.md`.
- Scope is backend-only design for computed authoring graph/outline APIs; no runtime code or WebUI implementation in this task.
- Review pass found and patched spec gaps for encoded IDs, explicit truncation, deterministic ordering, non-mutating live validation, stale preview revision behavior, and published-version validation context.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Reviewed the spec against current `origin/dev` VN script validator/service/API behavior.
- `git diff --check` -> passed.
- Bandit skipped because this task only changes Markdown design/task files.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote the backend-first VN script authoring graph API design spec. The spec defines read-only graph routes for stored drafts, supplied draft previews, and published versions; a combined outline/detail response; graph diagnostics plus validation diagnostics; conservative static edge semantics; content hashing; graph semantics versioning; encoded stable IDs; bracket JSON paths; explicit limits/truncation behavior; non-mutating validation; and backend-only rollout/testing guidance.
<!-- SECTION:FINAL_SUMMARY:END -->
