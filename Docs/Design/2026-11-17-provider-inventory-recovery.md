# Bounded recovery for provider inventory timeouts

Task: TASK-15505.21. Related workflow: TASK-15505, PR #3480.

The political PDF walkthrough encountered intermittent authenticated OpenAI model-list
timeouts. Validation stopped at its five-second deadline before any generation request.
The same scoped credential and endpoint subsequently returned a valid inventory without
configuration changes. The twenty-second negative cache then propagated the first
failure to chapter generation and concurrent citation judges.

## Decision

Recover inside the existing shared inventory refresh, before publishing a failure.
Model-list GETs use the checked HTTP client's existing retry policy, with at most two
attempts, five-second HTTP inactivity timeouts and a ten-second absolute budget shared
by pagination, DNS, TLS, response reads and retry waits. A caller's smaller deadline
still wins. A one-attempt policy retains one attempt and the original five-second
discovery bound. Supplemental model/alias lookups retain their existing one-attempt
behavior.

Only transport failures and the existing transient inventory HTTP statuses (429, 500,
502, 503, 504) qualify. Authentication, unsupported endpoints/responses, invalid model
IDs, parsing failures, egress refusals, DNS failures and response-size failures do not
gain retries. The HTTP client owns classification, decorrelated backoff and Retry-After.
No generation call is retried by this change.

The existing credential/endpoint/scope cache and in-flight future remain the sole
coordination mechanism. Concurrent callers share the same recovery; exhausted failures
retain the existing twenty-second negative cache. Expired inventories and partial
catalogs never authorize generation. Forced-refresh and strict endpoint rules remain
unchanged. A strict caller joining an existing refresh starts no additional transport
attempt and retains its own waiting deadline, as when reading shared cached results.

The async gateway keeps ownership of its bounded synchronous discovery worker through
cancellation. It drains that bounded work, then re-raises cancellation without starting
generation. Its maximum discovery drain can therefore rise from five to ten seconds;
no new background retry or cancellation mechanism is introduced. HTTP inactivity
timeouts are not absolute per-attempt deadlines: a slow trickle is bounded by the
aggregate deadline.

## Alternatives

- Merely extending one request's timeout would tolerate slow responses but would not
  recover a failed connection or transient server response.
- Retrying the gateway after the negative-cache window would add at least twenty
  seconds and require another coordination mechanism. Forced-refresh retries could
  supersede concurrent waiters and amplify requests.
- Retrying paid generation, accepting stale inventories or changing endpoint scope
  would weaken existing correctness boundaries and does not address this failure.

## Verification

Offline tests exercise the real discovery, checked HTTP retry/deadline implementation
and HTTPx clients with MockTransport replacing only external I/O. They cover timeout
recovery, shared concurrent refreshes, exhausted failures, authentication and malformed
responses, strict attempt limits, short deadlines, cancellation, client cleanup and
generation dispatch counts. Existing inventory, gateway, native Messages, Slides and
HTTP deadline suites protect neighboring contracts. Ruff and Bandit cover touched
Python code. Live model quality remains a separate walkthrough check.

Verification on 2026-11-17: the initial timeout-to-success regression failed before
the change and passed afterward. The integrated inventory, gateway, adapter, Chat,
Messages, Slides and HTTP response/deadline selection passed 448 tests, including
24 new offline recovery cases. Ruff passed; Bandit reported zero production findings.
The CI shard coverage guard reported zero newly uncovered files. Existing local
socket/DNS tests required their normal host permissions; generation stayed mocked.

ADR assessment: no new ADR required. The shared adapter/inventory boundary remains
governed by [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md), and the
existing outbound HTTP retry policy follows [ADR-051](../ADR/051-retry-backoff-schedules.md).
No new transport, storage, authentication or worker architecture is introduced.
Existing egress-error classification work remains in TASK-13950.89.
