# Reading List Snapshot Attestation Implementation Plan

**Task:** TASK-13150

**Goal:** Make each Reading List page's total, rows, and hydrated tags come from one database snapshot, then advertise that exact shipped guarantee through docs-info without changing the endpoint schema.

**Cross-repository consumer:** tldw_chatbook TASK-18919, `Docs/superpowers/plans/2026-08-31-library-collections-capture-reader.md`.

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a bounded transaction-correctness fix and capability attestation for an existing service contract; it adds no endpoint, schema, storage owner, dependency, or runtime boundary.

## Stage 1: Coherent Reading List page

**Status:** Complete

**Success criteria:** A controlled concurrent writer can produce only a wholly pre-write or wholly post-write page, and tag hydration uses that same transaction.

**Tests:** Add a deterministic `threading.Event` regression to `tldw_Server_API/tests/Collections/test_reading_service.py`; run the focused list/snapshot tests before and after the minimum transaction change.

## Stage 2: Exact docs-info attestation

**Status:** Complete

**Success criteria:** Both `capabilities` and `supported_features` expose literal `hasReadingSnapshotPagesV1: true`, with the existing Reading List endpoint and response unchanged.

**Tests:** Add and witness a failing focused test in `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`, then run both focused server files.

## Stage 3: Review and verification

**Status:** In Progress

**Success criteria:** Focused tests, touched-scope Bandit, formatting/diff checks, self-review, Backlog evidence, and an independent review pass are clean.

**Tests:** Run the focused Collections and config tests, `python -m bandit` over the two touched production files, and `git diff --check origin/dev...HEAD`.
