---
id: TASK-490.13.12
title: 'Sync v2 M3: Close broader domain expansion stage'
status: Done
labels:
- sync
- sync-v2
- m3
- domains
- verification
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and close Stage 5 broader domain expansion after source cache, media metadata, and derived-content reassessment have landed in separate commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 5 roadmap status and Step 4 reflect the verified state of the domain expansion work.
- [x] #2 Full Sync v2/domain-related test coverage is run against the current branch state or any skip is documented with rationale.
- [x] #3 Ruff, Bandit, and diff checks are run on the relevant touched scope or documented as not applicable.
- [x] #4 Backlog task records the final verification evidence and next stage handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed Stage 5 after verifying the separate domain-family commits:

- `503919b07 feat(sync): promote source cache domain`
- `867e4a60b feat(sync): promote media metadata domains`
- `b8b8588f6 docs(sync): classify derived content domains`

Roadmap status now marks Stage 5 complete and Step 4 checked.

Ruff note: a broad check over the whole Sync materializer/test directories surfaced existing baseline warnings outside the Stage 5 touched files (`BLE001` blind exception catches in older notes/chat materializers and import ordering in unrelated files). The closure gate therefore used the Stage 5 touched-file scope instead of refactoring unrelated code in this verification task.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and closed the Sync v2 M3 broader domain expansion stage. Source cache, media metadata, and derived-content classification each landed as separate commits, and the roadmap now marks Stage 5 complete.

Verification:

- `python -m pytest tldw_Server_API/tests/Sync -q` -> 358 passed.
- `python -m ruff check <Stage 5 touched Sync v2 files>` -> all checks passed.
- `python -m bandit -r <Stage 5 touched production files> -f json -o /tmp/bandit_sync_v2_m3_domain_expansion_stage.json` -> 0 results.
- `git diff --check` -> clean.

Next handoff: Stage 6 stricter encryption and key rotation.

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
