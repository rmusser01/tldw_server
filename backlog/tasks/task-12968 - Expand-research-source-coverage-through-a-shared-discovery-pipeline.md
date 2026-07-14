---
id: TASK-12968
title: Expand research source coverage through a shared discovery pipeline
status: To Do
labels:
- research
- discovery
- deep-research
- source-catalog
priority: High
references:
- TASK-12964
- TASK-2336
- https://www.sourclip.com/resources/research-sources
documentation:
- Docs/Design/2026-07-13-research-source-coverage-shared-discovery-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Program to expand credentialless research-source coverage far beyond the current eight-source catalog and make the same versioned discovery plan, execution, normalization, attribution, status, and budget machinery available to standalone Search and new Deep Research runs. Use the Sourclip research-sources page only as a dated coverage seed, never as a runtime dependency or mirrored catalog. Deliver direct, validated aggregator, and metadata/link-only coverage for every feasible credentialless inventory row; defer page retrieval until the shared egress boundary is certified, and keep credentialed/browser work separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A frozen, machine-readable seed manifest and orthogonal disposition ledger reconcile every captured example row to a canonical source or evidenced terminal disposition.
- [ ] #2 Every credentialless row deemed feasible for this program is shipped on its declared user surfaces and has current fixture plus live certification evidence; fixture-only, catalog-only, planned, and manual rows do not count as supported.
- [ ] #3 Standalone Search and newly created Deep Research runs share the canonical planner, gateway-backed executor, normalization, attribution, route outcomes, and physical-request budget contract while retaining separate persistence lifecycles.
- [ ] #4 Legacy standalone selection semantics, durable evidence source IDs, and already-active Deep Research sessions retain characterized behavior.
- [ ] #5 The catalog and user flow demonstrably support at least twelve routable targets, including bioRxiv, medRxiv, ClinicalTrials.gov, and PubMed Central, with truthful direct/indirect readiness and partial-result status.
- [ ] #6 All blocking child slices and dependencies are complete, cross-surface UAT passes, and intentionally blocked rows are reported with evidence and owner approval before program closure.
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
