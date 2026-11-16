# Embedding input coverage — TASK-15505.7

## Stage1: Freeze failures
**Goal:** Reproduce silent truncation and offset coverage failures.
**Success Criteria:** Failing behavioral tests for Jobs/direct media and raw HF calls.
**Tests:** Tokenizer-bound chunk partition, dense URLs/Unicode, canonical offsets, strict input cardinality.
**Status:** Complete

## Stage2: Refine before embedding
**Goal:** Reuse actual HF tokenizer/configuration and split canonical spans before batching.
**Success Criteria:** Jobs/direct/fallback store complete refined chunks; raw input fails before truncation/cache.
**Tests:** Focused unit/property/caller regressions.
**Status:** Complete

## Stage3: Verify integration
**Goal:** Independent review and real local model index qualification.
**Success Criteria:** Connected tests/lint/Bandit pass; offline real PDFtoken audit has no oversized chunks; root-controlled indexing/replay preserves failures.
**Tests:** Embeddings Jobs/direct suites and actual cached MiniLM tokenizer audit.
**Status:** In Progress

Verification: initial raw-inference RED, Jobs/direct/fallback RED, legacy-checkpoint RED2, and preallocation/HTTP-contract RED2 observed. Final68connectedPASS, scopedBandit0, exactcanonicalPDFaudit431→2261chunks/max512/zerooversized. Parent owns controlled real reindex and frozen retrieval replay.
