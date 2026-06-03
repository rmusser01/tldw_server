---
id: TASK-515
title: Backfill Sandbox/vz runtime ADR from reviewed inventory
status: Done
labels:
- docs
- process
- adr
modified_files:
- Docs/ADR/010-sandbox-vz-runtime-ownership.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- Docs/Design/2026-05-02-apple-containerization-evaluation.md
- tldw_Server_API/app/core/Sandbox/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the Sandbox/vz runtime decision from ADR inventory row INV-016 into a canonical ADR, linking the source design document and updating the inventory and ADR index.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review INV-016 and the Apple containerization evaluation source before writing the ADR.
- [x] #2 Create one canonical ADR for the Sandbox/vz runtime posture using the next ADR number.
- [x] #3 Update the ADR index, inventory mapping, and source design document with ADR links.
- [x] #4 Record documentation-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/ADR/inventory/2026-06-03-decision-inventory.md row INV-016 as the reviewed backfill source. Draft ADR-010 from Docs/Design/2026-05-02-apple-containerization-evaluation.md, preserving one decision per ADR: do not require Apple container, keep repo-owned helper/guest-agent path, move only narrow image-store metadata toward OCI compatibility, and keep networking as a separate policy milestone. Verify ADR metadata/index/source links/inventory mapping with rg and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reviewed `INV-016`, `Docs/Design/2026-05-02-apple-containerization-evaluation.md`, and current Sandbox module docs before writing the ADR. Current docs remain consistent with the source decision: `vz_linux` is repo-owned, OCI metadata is scaffolding only, and `deny_all` has no attached guest network device.

Created `Docs/ADR/010-sandbox-vz-runtime-ownership.md` as one backfilled ADR covering the Sandbox/vz runtime ownership and Apple `container`/`containerization` posture. Updated `Docs/ADR/README.md`, `Docs/ADR/inventory/2026-06-03-decision-inventory.md`, the Apple containerization source design, and `tldw_Server_API/app/core/Sandbox/README.md`.

Verification:
- ADR metadata and section check:

```bash
rg -n '\*\*Status:\*\* Accepted|\*\*Backfilled from:\*\* `Docs/Design/2026-05-02-apple-containerization-evaluation.md`|\*\*Related task:\*\* TASK-515|## Decision|## Alternatives considered|## Consequences|## Follow-up' Docs/ADR/010-sandbox-vz-runtime-ownership.md
```

- ADR/index/source/inventory link check:

```bash
rg -n "ADR-010|010-sandbox-vz-runtime-ownership|TASK-515|INV-016|Covered by" Docs/ADR/README.md Docs/ADR/inventory/2026-06-03-decision-inventory.md Docs/Design/2026-05-02-apple-containerization-evaluation.md tldw_Server_API/app/core/Sandbox/README.md
```

- Diff hygiene:

```bash
git diff --check
```

- Bandit skipped: documentation-only task; no Python/code paths touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled Sandbox/vz runtime ownership into ADR-010 from `INV-016` and `Docs/Design/2026-05-02-apple-containerization-evaluation.md`. Updated the ADR index, inventory row/slice mapping, source `Covered by` link, and Sandbox module README why-link. Verification recorded above; Bandit skipped because this touched documentation only. Known blockers: none.
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
