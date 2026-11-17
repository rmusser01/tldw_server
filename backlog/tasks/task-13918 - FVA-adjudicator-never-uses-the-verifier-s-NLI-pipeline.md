---
id: TASK-13918
title: FVA adjudicator never uses the verifier's NLI pipeline
status: To Do
labels:
- claims
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
FVAPipeline.__init__ (Claims_Extraction/fva_pipeline.py near lines 164-166) reads the verifier's _nli when the pipeline is built, but HybridClaimVerifier loads it lazily on first use (claims_engine.py near lines 970-1008), so ClaimAdjudicator always gets nli_pipeline=None and assesses every document with the LLM: one stance call per supporting and per counter-evidence document. Listed in TASK-13881's description, outside its acceptance criteria.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With local NLI enabled, the adjudicator uses the verifier's loaded NLI pipeline, with a test
- [ ] #2 Without local NLI, the adjudicator keeps the LLM path
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
