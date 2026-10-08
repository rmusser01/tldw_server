---
id: TASK-13534
title: Correct Knowledge workstream duplication and unresolved verification failures
status: In Progress
labels:
- knowledge
- research
- corrective
- audit
dependencies:
- TASK-13530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit merged PR3196, PR3205, PR3211 and PR3213 against pre-existing mechanisms and current dev. Remove demonstrated duplication and unsupported capture transport restrictions through existing stack contracts; reproduce and repair every previously qualified owning failure without treating inherited origin as completion. Preserve accepted source evidence, credentials and ownership protections, and use the repository landing procedure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every production addition in the four-PR workstream has an evidence-based reuse or duplication assessment with existing counterpart and caller references.
- [ ] #2 Public capture uses existing governed backend selection and transports; any required boundary repair is shared and demonstrated, with no HTTPX-only or curl prohibition justified by missing local dependencies.
- [x] #3 Four quota-eviction and one split-storage failures are reproduced and repaired under their owning project harnesses with unchanged assertions.
- [ ] #4 The historical CSV failure and affected default-harness, lint and format qualifications are diagnosed and resolved or remain explicitly blocked with evidence rather than silently closed.
- [ ] #5 Affected backend and client tests, security checks, types and builds pass; task13514 and related tracking accurately describe corrected and unfinished work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User explicitly requested corrective work and undoing duplicated functionality on2026-10-08. Base is current dev97ea9cd5fa7e3a61ee4e7c56f3d9643b7311d511. Audit first; no wholesale revert or blind removal of security/data guards. Four-PR diffs preserved in/private/tmp/knowledge-correction-20261008. Highest task13533 across dev and52openPRs checked before creating13534. ADR assessment pending existing-decision/caller audit; no new architecture approved by this tracking record.
Reopened under TASK13534 correction at user request. Prior qualified storage/default/lint/format failures are not closure evidence. Reproduce and diagnose owning cases; preserve original failed receipts and avoid unproved inheritance or timing claims.
Corrective spec: Docs/Design/2026-10-08-knowledge-mechanism-correction.md; plan IMPLEMENTATION_PLAN_knowledge_mechanism_correction_20261008.md. Independent full production inventories cover45 backend paths and all client paths across four actual merges. Verified scope: shared capture routing/preflight/stream reuse, unused writer removal, strict discovery, Quick Notes conflict, component-unmount retry durability, and checkpoint-loss pin restore. Fresh frozen dependencies reproduce default4 quota + native1 split failure. CSV cause remains unknown; observed lazy readiness/coverage gaps are separate. Existing ADR026/031/034/042/059/065/066 govern; no new durable rule. Accepted ADR065 substantive additions at7a06601ab8 are a verified historical immutability deviation, not proof of missing behavior approval.
Stage1 correction: storage red default8/4 and native11/1; corrected12/12 each with all assertions; full neighbors67/67 each. Actual ArtifactModalContent delayed1500ms reproduces original missingCSV assertion; modal-ready act+dynamicImportSettled passes, CSV/JSON type/payload/filename/cleanup validated. Affected27 owning files685/685 pass; exhaustive skill filename suite56/56. Canonical WebUI typecheck, extension compile and strict boundaries pass. Exact189 lint errors11→0, warnings1600→1530; per-finding f225562fc2 context ledger retained in private Task1 evidence under TASK13530.2. All25 correction files already fail whole-file Prettier at baseline; bounded ranges applied, wider debt retained explicitly. Supplemental touched UI diagnostics251/251, no new messages; broader unsupported UI type probe remains failed receipt and is not an owning gate. Historical broadCSV cause remains unknown. No new ADR: restores existing harness/validation contracts governed by026/031/034/042/059/065/066. Report: .superpowers/sdd/IMPLEMENTATION_PLAN_knowledge_mechanism_correction_20261008/task-1-report.md; no later functional fixes or build/browser replay in this stage.
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
