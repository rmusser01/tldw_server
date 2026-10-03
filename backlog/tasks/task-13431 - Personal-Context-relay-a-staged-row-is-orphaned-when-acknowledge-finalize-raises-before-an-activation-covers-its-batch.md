---
id: TASK-13431
title: 'Personal Context relay: a staged row is orphaned when acknowledge/finalize
  raises before an activation covers its batch'
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-13410 (#3078, landed via #3088). PersonalContextRelay._relay_owned now always finishes a row it has staged (deadline checks only stop rows before their Sync write). But if acknowledge_row or finalize_authority raises after stage_authority wrote the hidden home-authority envelope, the row is left staged with the Sync envelope at apply_status=pending. If a later activation install marks that batch covered_by_activation before a relay retry, no relay revisits it, and only purge_terminal batches have orphan cleanup. A pending ingress then blocks require_materialization_predecessors_applied for the dataset, the same symptom TASK-13410 fixed for the slow-host path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A relay or activation path finishes or compensates a staged-but-unfinished row even after its batch is covered by an activation, or activation refuses to cover a batch with a staged-unfinished row
- [ ] #2 A regression test injects an acknowledge_row/finalize_authority failure after staging and shows later projections in the dataset still apply
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
