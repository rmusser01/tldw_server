---
id: TASK-13518
title: Improve ingestion throughput (single-pass MediaWiki, join-based builders, batched
  embeddings)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 6. Plan: Docs/Plans/2026-10-06-perf-batch-6-ingestion-throughput-implementation-plan.md. MediaWiki single-pass count (Media_Wiki.py:1001); hoist ChromaDBManager + buffer page embeddings (:642); += to join in PDF/EPUB/XML/scraped-article builders; HTML single-parse reuse (Upload_Sink.py:1095, Plaintext_Files.py:240); safe_read_file sampled decode (Utils.py:534); contextual chunking bounded concurrency (ChromaDB_Library.py:1015); async embeddings array-input batching (async_embeddings.py:233); sync embeddings sub-batching + pooled session (chat_calls.py:303); sitemap bounded-concurrency scraping + cluster extractor single DOM (Article_Extractor_Lib.py:711, cluster.py:427).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
