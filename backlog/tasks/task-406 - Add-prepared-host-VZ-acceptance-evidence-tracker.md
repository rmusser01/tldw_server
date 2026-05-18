---
id: TASK-406
title: Add prepared-host VZ acceptance evidence tracker
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 15:00'
labels:
  - sandbox
  - vz_linux
  - host-gated
  - docs
dependencies: []
references:
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next sandbox roadmap slice for prepared-host VZ Linux acceptance evidence and gap tracking. The slice should make real-host acceptance runs reproducible and reviewable by documenting the evidence packet, expected skip taxonomy, residual manual boundaries, and next gaps without changing runtime behavior or enabling PR-triggered real VM execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prepared-host acceptance evidence packet is documented with exact fields for host, bundle/helper versions, commands, results, artifacts, expected skips, and residual gaps.
- [x] #2 The tracker keeps real VZ execution host-gated/manual and does not enable PR-triggered or scheduled destructive behavior.
- [x] #3 Operator docs or roadmap docs link to the evidence tracker so contributors know where to record manual/nightly prepared-host results.
- [x] #4 Verification covers docs references, host-gated workflow policy tests, diff hygiene, and a Bandit skip rationale for docs-only changes unless production code changes are added.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Added Docs/Sandbox/vz-linux-prepared-host-evidence.md as the prepared-host evidence packet tracker and gap log.
- Linked the tracker from the host-gated acceptance policy, macOS operator notes, and sandbox roadmap.
- Added focused infrastructure tests that guard the tracker fields and host-gated/manual execution boundary.
- Kept the slice docs/test-only; no workflow triggers or runtime behavior changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a prepared-host VZ Linux evidence tracker with exact packet fields, expected skip taxonomy, acceptance checklist, residual gaps, and recording guidance. Cross-linked it from the acceptance policy, operator notes, and roadmap, and added focused tests for the evidence contract and manual/host-gated boundary.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
- git diff --check

Bandit skipped: docs/test-only change; no production Python/runtime code changed.
Known skip: no new real VZ VM smoke was run in this slice; the tracker explicitly records that no dated prepared-host evidence packet has been added yet.
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
