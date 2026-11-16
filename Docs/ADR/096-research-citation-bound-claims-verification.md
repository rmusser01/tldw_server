# ADR-096: Research citation-bound Claims verification

**Status:** Proposed
**Date:** 2026-11-16
**Backfilled from:** not backfilled
**Decision owner:** Political PDF research workflow requester / PR #3480 reviewer
**Related task:** TASK-15505.16, TASK-15505.17, TASK-15505.18
**Related spec/plan:** `Docs/Design/POLITICAL_PDF_STRICT_CLAIMS_2026_11_16.md`

## Decision

Research chapter summaries, overviews and final QA answers reuse Claims_Extraction to assess semantic support against each claim's cited owned passages, preserving versioned findings separately from generated text and reviewer decisions.

## Context

Exact source anchors and valid citation IDs do not prove that a source supports an assertion. Separate generation paths bypass optional RAG Claims verification. Standard verification can search other passages or accept evidence-overlap fallback. Political-document summaries need faithful quantities, attribution, proposal wording and qualifications, with visible uncertainty.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Enable the generic RAG Claims flag | Does not cover separately generated answers and is not tied to the actual cited passages. |
| Build another judge and finding store | Duplicates the existing Claims and artifact/message ownership boundaries. |
| Delete or silently rewrite drafts on a failed check | Prevents transparent review and obscures what the model originally generated. |

## Consequences

Generation completion and support verification are independent states. Strict checks require semantic support and explicit coverage; missing citations, source changes, provider failures and bounds remain unchecked or need review. QA evidence is resolved by the server under workspace ownership, never accepted from arbitrary client quote text. Findings travel with the exact output and evidence identity through existing exports; restored findings represent historical assessments, not fresh validation. Checks add provider cost and latency, and model-based judgments still require regression evaluation and human calibration.

Existing Jobs, provider gateway, endpoint governance and evaluation-module decisions remain in force. No external fact-checking is implied.
