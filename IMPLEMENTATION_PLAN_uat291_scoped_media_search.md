# UAT291 — scoped PostgreSQL Media search

Tasks: TASK13260.228 and TASK13260.238 (adjacent SQLite/PostgreSQL title sorting). Preserve the frozen matrix failure and repair the query and failure-reporting boundary before another full UAT.

## Stage 1: causal regression
**Goal**: Reproduce scoped PostgreSQL parameter misbinding and hidden retrieval failures.
**Success Criteria**: Restricted official PostgreSQL fails the new filter/ranking cases before the fix; SQLite relevance controls pass and independently broken title sorting is recorded as UAT301. Standard stream failure tests distinguish errors from successful empty results.
**Tests**: Integer/UUID IDs, date/type/keyword/author filters, relevance and title sorting, pagination, owner isolation; failed and successful retrieval/stream controls.
**Status**: Complete — causal binding/title run10 failed/2 passed; hybrid partial-success controls2 failed/5 passed before repair.

## Stage 2: repair and focused validation
**Goal**: Align WHERE parameters with predicates and retain retrieval failures through the standard stream boundary.
**Success Criteria**: Causal tests pass; legitimate empty searches and partial successful sources still work; no sensitive error details reach clients.
**Tests**: SQLite/PostgreSQL restricted retrieval, FTS fallback, multi-source retrieval and streaming suites; touched-scope lint and Bandit.
**Status**: Complete —220 focused tests passed/0 skipped on real SQLite/PostgreSQL; Ruff/Bandit clear; independent follow-up review clear.

## Stage 3: targeted acceptance and tracking
**Goal**: Verify the specific-source PostgreSQL QA path and record its outcome.
**Success Criteria**: Native selected-source QA retrieves allowed content, tenant/confidential controls remain intact, failed retrieval is visible, evidence is retained outside the PR.
**Tests**: Targeted real PostgreSQL browser acceptance and source integrity; update task/tracker and commit verified changes.
**Status**: In Progress — fresh PostgreSQL native acceptance remains pending.
