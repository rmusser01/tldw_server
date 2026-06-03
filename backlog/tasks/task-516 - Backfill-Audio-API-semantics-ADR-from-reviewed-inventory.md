---
id: TASK-516
title: Backfill Audio API semantics ADR from reviewed inventory
status: Done
labels:
- docs
- process
- adr
modified_files:
- Docs/ADR/011-audio-api-semantics.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- Docs/Design/STT_TTS_Audio_API_Design.md
- tldw_Server_API/app/core/TTS/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the implemented Audio API semantics from ADR inventory row INV-021 into a canonical ADR, linking the source design document and updating the inventory and ADR index while leaving unresolved preset storage ownership in INV-022 inventory-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review INV-021, INV-022, and the STT/TTS Audio API design before writing the ADR.
- [x] #2 Create one canonical ADR for implemented Audio API auth, TTS routing, streaming error, and download-link semantics using the next ADR number.
- [x] #3 Update the ADR index, inventory mapping, and source design document with ADR links.
- [x] #4 Record documentation-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/ADR/inventory/2026-06-03-decision-inventory.md row INV-021 and Docs/Design/STT_TTS_Audio_API_Design.md as the backfill source. Draft ADR-011 as one Audio API semantics decision covering centralized auth, model-first TTS routing with fallback/priority behavior, structured streaming TTS errors by default, and non-streaming-only return_download_link behavior. Keep INV-022 separate because preset storage ownership is explicitly unresolved. Verify ADR metadata/index/source links/inventory mapping with rg and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reviewed `INV-021`, `INV-022`, `Docs/Design/STT_TTS_Audio_API_Design.md`, and TTS module docs before writing the ADR. `INV-021` covers implemented Audio API semantics. `INV-022` remains explicitly unresolved and is not included in ADR-011.

Created `Docs/ADR/011-audio-api-semantics.md` as one backfilled ADR covering centralized Audio API auth, model-first TTS routing with configured fallback/retry behavior, structured streaming errors by default, and non-streaming-only `return_download_link` semantics. Updated `Docs/ADR/README.md`, `Docs/ADR/inventory/2026-06-03-decision-inventory.md`, the STT/TTS Audio API source design, and `tldw_Server_API/app/core/TTS/README.md`.

Verification:
- ADR metadata and section check:

```bash
rg -n '\*\*Status:\*\* Accepted|\*\*Backfilled from:\*\* `Docs/Design/STT_TTS_Audio_API_Design.md`|\*\*Related task:\*\* TASK-516|## Decision|## Alternatives considered|## Consequences|## Follow-up' Docs/ADR/011-audio-api-semantics.md
```

- ADR/index/source/inventory link check:

```bash
rg -n "ADR-011|011-audio-api-semantics|TASK-516|INV-021|INV-022|Covered by" Docs/ADR/README.md Docs/ADR/inventory/2026-06-03-decision-inventory.md Docs/Design/STT_TTS_Audio_API_Design.md tldw_Server_API/app/core/TTS/README.md
```

- Diff hygiene:

```bash
git diff --check
```

- Bandit skipped: documentation-only task; no Python/code paths touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled implemented Audio API semantics into ADR-011 from `INV-021` and `Docs/Design/STT_TTS_Audio_API_Design.md`. Updated the ADR index, inventory row/slice mapping, source `Covered by` link, and TTS module README why-link. `INV-022` remains unresolved and inventory-only. Verification recorded above; Bandit skipped because this touched documentation only. Known blockers: none.
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
