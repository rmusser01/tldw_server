# VN Pending Command Recovery

Task: TASK-13385. Design approved by the requester on 2026-09-27.
Design approval does not authorize merging or waive the human-written Change
summary, current-head review or required CI merge gates.

Persist only unresolved workbench Start/Retry commands in tab-scoped sessionStorage
before POST: version, verified server API base/account ID, pack/slot ID, original
idempotency key and optional Retry source batch ID. No credentials, prompts or
recipes are stored. Use the existing authenticated profile and generation APIs.
The canonical profile endpoint is preferred; 404/410 alone permits the existing
authenticated /auth/me compatibility path for legacy single-user deployments.
Authentication failures never fall back to cached or synthetic browser identities.

Reload reads the journal only after confirming the current principal. Recovery is
an explicit action, never an automatic generation POST. It replays the saved body,
even if the current status advertises another source batch. One unresolved command
per pack blocks new Start/Retry/Cancel commands for that pack. Other packs remain
usable. A successful response clears the matching journal entry; ambiguous errors
retain it. Known pre-admission client errors may clear it, except idempotency
in-progress conflicts, which are not proof of rejection.
The existing VN recipe/source rejection envelopes are explicit pre-admission
failures, including HTTP 409. A closed code allowlist is recognized only on
Start/Retry POSTs at that status; client-owned messages exclude raw nested data.
Unknown conflicts and server failures remain ambiguous.

Logout/account/server boundaries invalidate current async operations immediately;
revalidation permits retained commands only for the same verified authority.
Successful focus/pageshow revalidation restarts interrupted detail reads without
posting generation; command pre-send verification does not restart those reads.
Unmount leaves the journal intact but prevents late callbacks from changing it.
Storage read/write/removal errors are visible and fail closed. An unreadable
journal is never used; explicit discard requires a warning that it neither cancels
server work nor proves it was rejected. No expiry silently loses ambiguous keys.

Scope excludes backend/Jobs changes, exactly-once output commits, model-byte drift,
live GPU qualification and cross-tab coordination. Browser session storage can be
copied when a tab is duplicated; existing server idempotency remains authoritative.

Verification: real storage validation tests; workbench remount/lost-response tests;
changed-source replay, principal/server isolation and stale callback tests; storage
failure tests; full VN frontend suite, typecheck, scoped lint and browser smoke.
