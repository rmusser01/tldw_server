# ADR-049: Cross-source RAG results are fused by rank, not by raw score

**Status:** Accepted
**Date:** 2026-09-22
**Backfilled from:** not backfilled
**Decision owner:** repository owner (decided 2026-09-22 during core-module review remediation)
**Related task:** TASK-13315
**Related spec/plan:** `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md` (F16)

## Decision

When `MultiDatabaseRetriever.retrieve` draws documents from **more than one** source, it
orders them by **reciprocal rank fusion** over each document's rank within its own
source, then rescales the fused values onto `(0, 1]`.

When it draws from **one** source, results are returned exactly as before, sorted by the
retriever's own score. This is a cross-source fix only.

## Context

`MultiDatabaseRetriever.retrieve` flattened every source's results into one list and
sorted it by `score`. Those scores are produced on **eleven incompatible scales**:

| Scale | Sources |
| --- | --- |
| min-max normalised | media, chunk FTS, vector |
| constant `1.0` | both notes paths |
| constant `0.5` | chat history, character cards, SQL |
| constant `0.6` / `0.4` | claims |

Sorting them together compared numbers that do not mean the same thing, and the
constants won.

**The failure.** `sources=["media_db","notes"]`, `top_k=10`, an include list of 20 note
ids. `_retrieve_allowed_notes_via_sql` returns notes ordered by `last_modified` with **no
text match required at all** — its own docstring says so — and stamps every one
`score=1.0`. Media is min-max normalised, so exactly one media document reaches `1.0`.
The global sort placed all 20 notes at or above every media document, and
`documents[:max_results]` returned a top-10 of **notes only** — zero media documents,
including the highest-BM25 matches. Generation then answered from documents that were
never scored for relevance.

**The second instance.** min-max maps every source's best hit to exactly `1.0` and
`list.sort` is stable, so which source took the top slot was decided by dict insertion
order rather than by match quality.

Rank-based fusion needs no calibration between scales: it reads only each document's
position within its own source, and each source's internal ordering is self-consistent.

## Why not raw RRF scores

`retrieve_with_fusion` and `_reciprocal_rank_fusion` already existed and were unused by
the main path. Routing through them verbatim was rejected twice over:

1. `retrieve_with_fusion` calls `retr.retrieve(query)` with no config and no per-source
   restrictions, so it would have silently dropped `allowed_media_ids`,
   `allowed_note_ids`, `index_namespace` and the whole `RetrievalConfig`. The fusion had
   to move into `retrieve`, not the other way round.
2. Raw RRF scores are ~`0.016` at rank 1. Callers depend on the `[0, 1]` range:
   `unified_pipeline` re-sorts by `score` and caps to `top_k` in three places
   (`4458`, `4516`, `4766`), applies a bounded boost `min(1.0, score * 1.1 + 0.02)`
   (`5069`), and `Research/providers/local.py:117` returns the value in an API response.

The alternative of **keeping each document's original score and fusing only the order**
was also rejected: `unified_pipeline` re-sorts by score immediately afterwards, which
would have undone the fusion completely. Rescaling onto `(0, 1]` is what keeps the fused
order stable under those re-sorts while leaving the range intact.

`min_score` is unaffected: every retriever applies it internally, before results reach
the fusion step.

## Consequences

- Multi-source result ordering changes for every caller. That is the point: the previous
  ordering was decided by whichever source used the largest constant.
- A document returned by two sources now appears **once**, with its ranks summed, where
  the global sort listed it twice. Standard RRF, and it matches the existing
  `_reciprocal_rank_fusion`, but a multi-source result set can be shorter than before
  for the same inputs.
- Single-source retrieval is byte-for-byte unchanged, which is most of the traffic and
  keeps the blast radius on the case that was actually broken.
- Scores in multi-source results are now rank-derived, so they express relative position
  rather than per-source similarity. They were never comparable across sources anyway —
  that was the bug.
- `k = 60` is the standard RRF damping constant and is not currently configurable. If a
  caller ever needs to weight sources, `_weighted_fusion` already exists.

## Follow-up

`_retrieve_allowed_notes_via_sql` returning `last_modified`-ordered rows with no text
match, stamped `score=1.0`, is its own defect — fusion stops it dominating the results
but those notes are still not relevance-ranked. Tracked as TASK-13346.
