---
id: TASK-13609
title: Bind retention candidates to frozen reviewed references
status: Done
labels:
- chat
- assets
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the approved H2 private typed candidate admission seam. Reuse original retention request/source reservation and active lease; derive candidate hash and representation from the exact retained reference. Preserve caller work on failure. This reserves metadata only and does not qualify quota, upload production service, claims or publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact original request, source fence and live generation are required before candidate admission.
- [x] #2 Hash and representation derive from the requested retained reference; unavailable or foreign references cannot reserve bytes.
- [x] #3 Replay and failures preserve original candidate identity and unrelated caller work; connected store and upload regressions pass with skips explicit.
- [x] #4 Shared scope serialization refuses malformed unchecked fields before normalization across typed callers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; ADR049/050 and proposed071 govern owner/source/lifecycle; no persistence/public API changes. Approved H2 Task2.3. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_retention_candidates_13609.md.
Initial causal15failed15officialPGskip6.66s; green15/15 8.58s. Initialconnected157pass158skip16warnings131.93s. Independent reviews identified finalclock gap and malformedscope normalization; actualcausal4failed4skip30deselect2.74s. Fixed fresh finalclock and sharedscope serializer original-fieldvalidation; focused19/19 10.36s. Strengthened unchangedstoredlease controlledclock positive/expired +scope5pass5skip30deselect3.94s. Finalconnected9suites running. Metadata-only no publicproducer/quota/claim/bytespublication.
Final9connected363pass236skip16warnings121.48s. Fullcanonical2129/3301 bd7997199762d42481de4c4b32536f3c82f279d695c86ff44afa427b36771212 fingerprint/path checks pass unchanged. Three-fileRuff0/productionHEAD0; actual2prodBandit0/errors0; scopedformat/compile/diff0; bothfinalsourceCLEAR. Initialnormal6filehooks0 withhouseRuff/Blacknativefilters skipped; no bypass. Onlyowncompletedplan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_candidates_13609_completed.md. Bounded record Docs/Reviews/CHAT_RETENTION_CANDIDATES_13609_2026_10_12.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added typed frozen-reference retention candidate admission using originalrequest/source/lease and existingcandidate store. Derived hash/representation and immutablemetadata replay; localrollback preservescallerwork. Reproduced/fixed finalclock gap and sharedscope normalizationroot. Final363/236skip, canonicalcontractunchanged, staticsecurityclear, bothreviewsCLEAR. Metadataonly; broadH2/UX remainsopen.
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
