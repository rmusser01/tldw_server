# VN Generation Readiness And Targeted Recovery

Backlog: TASK-13249. GitHub: #2021. Approved direction: generation readiness,
actionable failures and targeted recovery, continuing the September 13 review.

The existing WebUI starts generation without the idempotency key required by the
API, offers no slot retry, and does not refresh active generation progress.

Add an authenticated, owner-scoped GET generation-preflight endpoint. Reuse the
image adapter registry/catalog to resolve slot override, pack default and server
default backends and report configuration diagnostics. Report local worker flags
as configuration only. Never claim worker liveness, load models, perform image
generation, or reject separately deployed workers based on API-process settings.
The preflight is advisory; existing generation API semantics stay intact.

The monitor shows preflight warnings, named failed slots and per-slot Retry.
Start and Retry send stable per-operation idempotency keys, reused after ambiguous
transport failures and cleared only after a successful response. Prevent repeated
submissions and guard async results against pack changes. Refresh status and slots
while a batch is active, with bounded non-overlapping polling and visible recovery
from status-load failures.

Tests cover owner scoping, backend precedence/configuration, absent local workers,
required idempotency, retry replay, active progress and stale pack responses.
Browser QA uses the existing mocked VN route to verify desktop/mobile interactions.
Recipe snapshots and worker crash/duplicate-delivery persistence hardening remain
under #2021; this slice makes no completion claim for those behaviors.
