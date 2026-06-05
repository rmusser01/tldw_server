---
id: TASK-2262
title: Backfill Embeddings API and media pipeline ADR
status: To Do
dependencies:
- TASK-2261
labels:
- docs
- process
- adr
- embeddings
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded Embeddings ADR from TASK-2261 evidence. Scope the accepted decision to OpenAI-compatible embeddings request/response semantics, provider resolution and allowlist safeguards, optional adapter-registry routing with legacy provider-config fallback, cache/batching/circuit-breaker reliability controls, and media embedding pipeline ownership where core Jobs owns the durable root status record while Redis Streams owns stage delivery. Keep billing/quota behavior, local provider URL policy, pgvector/Chroma backend evolution, and legacy Jobs worker details as explicit caveats unless separately confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create the next accepted ADR under `Docs/ADR/` using the standard ADR template and TASK-2261 evidence.
- [ ] #2 Keep accepted claims scoped to OpenAI-compatible API semantics, provider resolution/allowlist safeguards, optional adapter routing, cache/batching/circuit-breaker reliability controls, and Jobs-root/Redis-stage media pipeline ownership.
- [ ] #3 Update `Docs/ADR/README.md`, the INV-032 inventory row/default disposition, and the Embeddings README backlink after ADR creation.
- [ ] #4 Record verification and Bandit applicability in this task.
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
