# Monitoring Backend Review Design

Date: 2026-04-07
Topic: Architecture-heavy review of the backend Monitoring module and APIs in `tldw_server`
Status: Approved for review execution

## Goal

Produce an evidence-based backend review of the Monitoring area that identifies:

- concrete correctness bugs and edge-case failures
- alert identity, state-model, and source-of-truth problems
- permission, authorization, and boundary mismatches across monitoring APIs
- persistence, path-resolution, startup, and backend-selection hazards
- concurrency, singleton, and async/threading risks
- maintainability problems that are likely to create future defects
- missing, misleading, or overly narrow tests

The review should prioritize practical production risks first while still surfacing the structural weaknesses that make this area hard to change safely.

## Scope

This review is centered on the backend Monitoring surface and its direct persistence and API edges:

- `tldw_Server_API/app/core/Monitoring/__init__.py`
- `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
- `tldw_Server_API/app/core/Monitoring/notification_service.py`
- `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`
- monitoring-specific schema definitions in `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- `tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py`
- `tldw_Server_API/app/services/admin_monitoring_alerts_service.py`

Within that scope, the review should treat the backend as three related but distinct subdomains:

- topic monitoring watchlists, scans, alerts, and notification delivery
- admin monitoring control-plane state such as assignment, snooze, escalation, and event history
- the seam between persisted topic alerts and admin overlay truth

Direct validation targets are included where they materially exercise the module contract:

- focused Monitoring tests under `tldw_Server_API/tests/Monitoring/`
- focused admin monitoring tests under `tldw_Server_API/tests/Admin/`
- relevant AuthNZ monitoring tests where they validate backend selection, permissions, or monitoring summary semantics

## Non-Goals

This review does not cover:

- the admin monitoring React UI except where it exposes a backend contract smell
- Guardian/self-monitoring rule management and APIs, which are a separate subsystem despite living under `app/core/Monitoring/`
- a repo-wide review of every monitoring or metrics producer outside this module boundary
- implementation of fixes during the review itself
- unrelated observability or metrics redesign not directly tied to the Monitoring backend
- speculative cleanup work that is not connected to an identified backend risk

## Approaches Considered

### 1. Risk-first layered backend audit

Inspect APIs, overlay state, core services, persistence, and tests in dependency order, with findings ranked by operational risk and architectural brittleness.

Strengths:

- best fit for finding both immediate bugs and deeper structural weaknesses
- keeps the review tied to concrete backend seams
- aligns with the requested architecture-heavy lens

Weaknesses:

- slower than an endpoint-only pass because state transitions must be traced across layers

### 2. Endpoint-first contract audit

Start from `/api/v1/monitoring` and `/api/v1/admin/monitoring`, then inspect only the service code needed to explain observed API risks.

Strengths:

- efficient for auth, validation, and contract-shape issues
- good for user-visible control-plane defects

Weaknesses:

- weaker at surfacing service-layer brittleness, dedupe behavior, and persistence hazards

### 3. Core-service-first audit

Start inside `app/core/Monitoring` and the DB layer, then validate whether the API layer preserves or distorts those contracts.

Strengths:

- strong for concurrency, lifecycle, and persistence analysis
- useful for exposing implicit invariants and hidden coupling

Weaknesses:

- can underweight route-level permission and error-shaping issues if used alone

## Recommended Approach

Use the risk-first layered backend audit.

Execution order:

1. inspect API boundaries, route responsibilities, and permission assumptions
2. inspect alert identity and overlay-state flow across runtime rows, persisted alerts, and admin state/event history
3. inspect core service behavior for watchlist loading, rule compilation, dedupe, notification dispatch, and singleton lifecycle
4. inspect persistence and backend differences for SQLite/Postgres, schema bootstrapping, and path handling
5. validate conclusions against the most relevant tests and recent churn

This keeps the review grounded in production behavior while still surfacing the architectural weaknesses that make the area fragile.

## Review Method

### Pass 1: API and boundary audit

Inspect:

- `monitoring.py` and `admin/admin_monitoring.py`
- permission dependencies, actor resolution, error handling, and audit emission
- whether route semantics match the state they mutate
- the monitoring-specific request and response schemas used by those routes

Primary questions:

- do route names and behaviors match operator expectations?
- are read, acknowledge, dismiss, snooze, and escalate actions mapped to a coherent backend state model?
- are auth and permission checks consistent across the monitoring surfaces?

### Pass 2: Alert identity and source-of-truth audit

Inspect:

- how alert identities are built and consumed
- how runtime alert rows merge with authoritative overlay state
- event history semantics and audit consistency
- the helper/service code that translates persisted topic alerts into admin-facing identities and merged rows

Primary questions:

- is the system operating with one alert identity model or several partially overlapping ones?
- can state drift between persisted topic alerts and overlay state?
- can actions succeed against synthetic or nonexistent alert identities in ways that confuse operators or tests?

### Pass 3: Core service and operational audit

Inspect:

- watchlist loading from file and DB
- rule compilation safety and regex heuristics
- dedupe behavior, streaming alert logic, and snippet construction
- notification file sink, webhook/email dispatch, digest handling, and singleton lifecycle

Primary questions:

- can service-local caches or process-local state create surprising behavior across reloads or multi-process deployments?
- are async/thread boundaries safe and proportionate?
- do path resolution and configuration rules fail safely and predictably?

### Pass 4: Persistence and backend-compatibility audit

Inspect:

- `TopicMonitoring_DB.py`
- `admin_monitoring_repo.py`
- schema bootstrapping and backend-specific behavior

Primary questions:

- are SQLite and Postgres behaviors materially aligned where they need to be?
- are startup/schema assumptions explicit and reliable?
- do persistence APIs preserve atomicity and avoid silent data loss or replacement hazards?

### Pass 5: Test adequacy and churn audit

Inspect:

- Monitoring tests
- Admin monitoring tests
- nearby AuthNZ monitoring tests where they validate backend selection or permission invariants
- recent monitoring-related git history for recently patched risk zones

Primary questions:

- which high-risk invariants are untested or only indirectly tested?
- do current tests encode desirable contracts or merely the current implementation?
- where does recent churn suggest repeated instability or unclear ownership boundaries?

## Review Criteria

Each issue should be evaluated against one or more of these categories:

- correctness and edge-case safety
- auth, permission, and boundary correctness
- state-model integrity and source-of-truth clarity
- concurrency and lifecycle safety
- persistence and backend compatibility
- operational safety and configuration predictability
- maintainability and change-risk
- test adequacy and documentation drift

## Evidence Standard

The review should avoid speculative claims. A finding should be backed by at least one of:

- a concrete code path that can produce incorrect, unsafe, or misleading behavior
- a mismatch between route semantics and the state actually mutated
- a source-of-truth or identity-model inconsistency
- a lifecycle or backend assumption that is not adequately guarded
- a meaningful test gap around an important invariant or failure path

If local evidence is incomplete, the item should be labeled as an open question or lower-confidence risk rather than overstated as a confirmed bug.

## Deliverable Format

The final review output should be organized as:

1. findings first, ordered by severity
2. structural improvements that would materially reduce future defects
3. open questions or assumptions
4. test and documentation gaps
5. prioritized next steps

Each finding should include:

- severity (`High`, `Medium`, or `Low`)
- confidence (`High`, `Medium`, or `Low`)
- type (`correctness`, `auth`, `state model`, `concurrency`, `operational`, `maintainability`, `test gap`, or `docs drift`)
- impact
- concrete reasoning
- file reference(s)

## Success Criteria

This review is successful if:

- every important finding is tied to a concrete backend code path, contract mismatch, or test gap
- immediate production risks are clearly separated from longer-horizon cleanup work
- the review identifies the dominant structural weaknesses rather than listing isolated symptoms
- the result can directly feed either a fix plan or a set of scoped follow-up tasks

## Expected Output Character

The review should be direct and specific rather than exhaustive for its own sake.

It should prefer:

- fewer, higher-signal findings over long low-value inventories
- explicit statements about where the backend truth lives and where it does not
- practical follow-up recommendations that match the project’s existing architecture instead of proposing unrelated rewrites
