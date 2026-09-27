# Calendar Durable Sync Admission

Date: 2026-09-27
Status: Approved in chat by the requester ("approved")
Tracking: TASK-13356; PR #3019; review findings 4113440983, 4113440985, 4113440989.

## Scope and Authority

This supplements the Calendar module design with the approved backend admission
change. It does not authorize an immediate merge, waive provider smoke or human
Change-summary requirements, or expand read-only CalDAV scope.

## Persistence and Dispatch

CalendarDatabase owns a durable current reservation per binding. A reservation
contains an immutable random admission identity, actor/tenant, canonical window,
trigger reason, stable Jobs idempotency key, and optional dispatched Job identity.
The binding uniqueness constraint is authoritative across processes and repository
aliases. New admission and its queued audit commit in one Calendar transaction
before any runnable Jobs insert. Failure of either write rolls both back.

Jobs dispatch occurs outside the Calendar write transaction. Concurrent callers
dispatch the same admitted identity using existing receipt-backed Jobs admission
and replay. Transactional receipts survive terminal/archive moves; missing Job
authority fails closed. Retention spans the representable UTC date range, so a
delayed dispatcher cannot recreate retired work. Reuse existing Jobs commands and
backends without adding shared Jobs code or new locks.
Recording the dispatched identity is conditional on the admission identity; a
late caller cannot overwrite a successor admission. The audit initially identifies
the admission and is correlated with the Job after dispatch. Dispatch failure
retains the committed admission/audit for recovery rather than admitting new work.

Scheduler scans recover pending admissions even for manual-only bindings, without
changing their polling cadence or treating completion as a fresh trigger.
Completed recovery repairs correlation and retires only the exact intent. Persist
recovery-attempt ordering so bounded failures cannot starve later work.
A crash after Job creation but before recording
its identity is repaired by exact scoped idempotency lookup. Queued/processing
work, including delayed retries, remains authoritative. Only confirmed terminal
Jobs (including archived Jobs) release a reservation. Missing previously recorded
Jobs fail closed instead of silently admitting duplicates. No time-based lease
expiry alone releases runnable work.

## Legacy and Lifecycle

When no reservation exists, traverse the status-independent owner-scoped Calendar
Jobs population using existing creation-keyset pagination and filter active rows
within each page, including older work beyond the first 100. Separate active-state
scans cannot safely follow a processing Job moving back to queued retry.
Adopt an existing Job without dispatching another or adding a duplicate queued
audit. New work never depends on this legacy scan once durably correlated. Scope
and binding identity are checked before adopting or dispatching work. Separate
bindings do not hold a common application lock during Jobs I/O; SQLite retains
its normal short serialized write transactions for atomic admission and audit.

## Core and HTTP Boundary

CalendarService exposes a typed trigger use case. It owns actor/tenant ownership,
binding defaults, aware ISO timestamp validation and canonical UTC normalization,
and calls durable admission. The async endpoint maps validated schema fields,
invokes that service in the existing cancellation-drained off-loop DB phase,
maps domain errors, and serializes the typed response. Scheduled admission shares
the same validation and durability contract.

## Verification

TDD covers audit rollback, dispatch failure/recovery, post-dispatch crash,
simultaneous independent processes, old active Jobs, terminal/retry/archive
reconciliation, repeated identical windows, tenant boundaries, independent
binding dispatch, endpoint delegation, and native/AnyIO cancellation draining.
Preserve prior Calendar rate, temporal, recurrence, credential, permission and
agenda contracts. Verify against current dev, obtain independent scoped review,
run touched-source Bandit and normal hooks, publish with verified head protection,
reply individually to all three findings, then request one full exact-head review.
