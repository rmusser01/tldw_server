---
id: TASK-490.13.11
title: 'Sync v2 M3: Reassess derived content domains'
status: Done
labels:
- sync
- sync-v2
- m3
- domains
- design
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide whether transcripts, summaries, embeddings, and evaluation artifacts should become Sync v2 source-of-truth domains or remain rebuildable/cache domains, and record M3/M4 implications in the roadmap docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Derived-content classes are classified as source-of-truth, promoted metadata reference, rebuildable cache, or deferred with rationale.
- [x] #2 M3 domain capabilities and restore semantics are updated to avoid accidentally syncing derived blobs or user-private generated artifacts before ownership/conflict semantics are clear.
- [x] #3 Later implementation follow-ups are identified for any promoted/deferred domains.
- [x] #4 Roadmap plan Stage 5 Step 3 is checked off with verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Documented the M3 derived-content decision in the design and API docs.

- M3 capabilities intentionally do not advertise transcript, summary, embedding, or evaluation artifact domains.
- Transcripts are deferred source-of-truth candidates because generated text can later become user-corrected personal knowledge and needs stable segment identity before sync.
- Summaries are split: generated summaries are rebuildable cache, while user-pinned or edited summaries are future source-of-truth candidates.
- Embeddings are classified as rebuildable cache only and should be rebuilt from synced content/model configuration.
- Evaluation artifacts are split and deferred: projects/configs/datasets/human labels may become source-of-truth domains, while generated run outputs need artifact metadata, retention, redaction, and blob policy first.
- Existing anchors (`source_cache.entry`, `media.*`, `attachment.ref`, and M2 blobs) may reference derived artifacts, but metadata-only envelopes must not carry transcript bodies, summary text, vectors, generated metrics, or raw artifacts.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reassessed derived content for Sync v2 M3 and recorded the decision: no derived-content domains are promoted in M3. The docs now classify transcripts and user-edited summaries as deferred source-of-truth candidates, embeddings as rebuildable cache, and evaluation artifacts as split/deferred until ownership, conflict, retention, and redaction semantics are clear.

Verification:

- `git diff --check` -> clean.
- `rg -n "T[B]D|T[O]DO|FIX[M]E|transcript.*M3 capabilities.*include|embedding.*source-of-truth|metadata-only.*transcript bodies" Docs/Design/Sync_V2_M3_Polished_Multi_Device.md Docs/API/Sync_V2_M3.md Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md` -> no matches.
- Bandit skipped: documentation-only change.

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
