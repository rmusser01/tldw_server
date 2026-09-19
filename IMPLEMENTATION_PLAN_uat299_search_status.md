# UAT299 Character and Chat retrieval repair

Backlog: TASK13260.236. Authorized repair stage before another full UAT matrix.
Related regression repairs: TASK13260.255 (UAT317 duplicate Note conflicts) and TASK13260.256 (UAT318 historical migration/source fixtures).

## Stage 1: Reproduce search and status failures
**Goal**: Reproduce punctuation-bearing Character/Chat search with real SQLite and partial-success status with a real retriever.
**Success Criteria**: Positive, unrelated, quoted, deleted and paginated controls distinguish parse failure from no match; partial failure is not reported as searched/empty.
**Tests**: Existing DB and RAG fixtures, exact original comma/bracket cases, request isolation and UI trust summary.
**Status**: Complete

## Stage 2: Repair using existing contracts
**Goal**: Use the existing SQLite query normalizer and request-local source diagnostics; reflect actual failures in the existing QA summary.
**Success Criteria**: Preserve quoted queries, ownership, successful evidence and source-specific failure information without shared mutable request state.
**Tests**: Causal tests turn green, actual PostgreSQL compatibility, focused suites, lint, Bandit and independent review.
**Status**: In Progress

Native 63d4581613 exposed an untested transport boundary: streaming prefetch discards aggregate source_status. Add causal streamed-event coverage (ordinary and progress paths, empty/partial success, safe fields) and forward the existing diagnostic contract. New TASK13260.257/UAT319 covers document-type filtering of otherwise matching Character/Chat results and non-file PostgreSQL readiness; TASK13260.258/UAT320 covers the Balanced post-load stall. Preserve immutable native failures before replacement runs.

## Stage 3: Native acceptance and closure
**Goal**: Verify real QA source retrieval and truthful status in isolated native installs.
**Success Criteria**: Owned Character/Chat evidence is found, unrelated data excluded, partial failures disclosed; sources, cleanup and limitations recorded.
**Tests**: Native SQLite and official PostgreSQL controls, immutable source audit, owned runtime cleanup.
**Status**: In Progress

## Repair evidence checkpoint

Original SQLite punctuation, false source status, grouped-query fallback semantics, mixed-success summary, and both optional-retrieval deadline paths have retained causal failures. The final focused search/status suite passes28 with actual PostgreSQL required; UI passes56. Six Note conflict/API/fixture checks pass after both duplicate constraints are recognized and obsolete fixtures corrected. Caller cancellation propagates and awaits owned retrieval work. Independent review reports no remaining actionable findings. Bandit reports0 findings/errors; broader regression and immutable native acceptance are pending. Local evidence remains ignored in .tmp/uat299-repair.

Final source acceptance:509backend/0skips and96frontend checks pass; actual PostgreSQL required. Bandit0production/0tests (B101 excluded);9Ruff and93TypeScript baseline diagnostics unchanged. UAT317/318 verified; UAT299 native acceptance remains in progress.
