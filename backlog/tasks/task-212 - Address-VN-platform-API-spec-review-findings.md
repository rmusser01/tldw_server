---
id: TASK-212
title: Address VN platform API spec review findings
status: Done
assignee: []
created_date: '2026-05-10 02:46'
updated_date: '2026-05-10 02:48'
labels:
  - vn
  - api
  - design
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1486'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the full VN platform API design spec after review. Scope is documentation/task metadata only. Address the seven identified design issues before implementation planning: published script profile snapshotting, runtime turn/action transaction recovery, cleanup blockers for pinned manifests and saves, multipart idempotency, public-vs-debug script state exposure, explicit character safety metadata policy behavior, and a single explicit metadata database boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec requires published script versions and sessions to snapshot effective policy/generation profile configuration so replay does not drift after admin profile edits.
- [x] #2 Spec defines durable runtime turn/action request records, in-flight recovery, timeout/abandoned-lock handling, and replay semantics for non-Job runtime commands.
- [x] #3 Spec defines generated-file cleanup blockers/refcounts for published manifest snapshots, active sessions, checkpoints, save slots, and TTS outputs.
- [x] #4 Spec defines idempotency for multipart upload/import-preview endpoints including byte hashing and item upload duplicate behavior.
- [x] #5 Spec keeps public script state spoiler-safe and moves raw label/cursor/interpreter internals to owner-gated debug diagnostics.
- [x] #6 Spec defines unknown/absent/conflicting/imported character safety metadata behavior by content rating and policy profile.
- [x] #7 Spec explicitly chooses the VN metadata database boundary and records docs-only verification plus Bandit skip rationale.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add cross-cutting profile snapshot rules for published script versions and sessions. 2. Add durable runtime action request semantics for synchronous VN runtime commands. 3. Tighten generated-file cleanup blockers and multipart idempotency. 4. Make script state spoiler-safe, define safety metadata policy behavior, and pin the metadata database boundary. 5. Run docs-only verification, record Bandit skip rationale, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md to address all seven review findings. Verification: git diff --check exited 0; the spec file exists; rg confirmed the expected profile snapshot, runtime action request, cleanup blocker, multipart idempotency, spoiler-safe state, character safety metadata, metadata boundary, and Bandit skip sections. Bandit is not applicable because this touches markdown/task metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all seven VN platform API spec review findings in Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md. Added immutable policy/generation profile snapshots at publish/session creation, durable runtime action request recovery rules, cleanup blockers for generated-file references, multipart idempotency hashing, spoiler-safe public script state, explicit character safety metadata policy defaults, and a single per-user ChaChaNotes metadata boundary. Verification: git diff --check passed; spec path sanity passed; targeted rg checks found each required section. Bandit skipped because the change is docs/task metadata only.
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
