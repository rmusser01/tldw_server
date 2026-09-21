# Quiz Generation Observability

Task: TASK-12102.3.7. Approved scope: backend-only, cross-profile generation
counters and latency using the existing metrics registry. No new dashboard,
frontend behavior, persistence schema, or telemetry destination.

## Boundary

`generate_quiz_from_sources` owns one observation per invocation. This covers
all six profiles, the WebUI and extension's shared API, and the legacy
`generate_quiz_from_media` delegate without double counting.

A request-local observation starts a monotonic clock before normalization.
Once profile and source types are normalized, it records the request before
source resolution. An early normalization rejection is counted on exit, with
unknown dimensions where normalization did not complete. Success is recorded
only after persistence and response construction finish. Failures and
cancellation also produce terminal outcomes and latency observations.

## Classification And Privacy

Internal phase markers distinguish validation from provider and persistence
failures without changing or wrapping exceptions. OSCE's existing typed
generation errors distinguish provider failures from invalid/unverified output;
OSCE persistence failures remain runtime failures even though the existing
service wraps them in `OsceVerificationError`. Within OSCE generation, a chained
verification error identifies a verifier execution failure, while an unchained
verification rejection remains a validation error. Source resolution uses its
existing `ValueError` rejection contract for invalid, missing, or empty sources;
database/operational exceptions remain runtime failures.

Metric dimensions are allowlisted profile, source type, and outcome. Repeated
sources of one type use that type; multiple recognized types use `mixed`;
unsupported or unavailable types use `unknown`. Identifiers, content, prompts,
model names, provider names, and exception messages are never metric labels.

## Failure Isolation

The observation never suppresses the generation exception or changes its
identity. Registry lookup, registration, counter writes, histogram writes,
clock reads, and fallback logging are best-effort. Each metric write is isolated
so a failed counter does not prevent latency recording. A constant debug message
reports telemetry failures without exposing the underlying exception.

The three metric families are standard registry definitions, including explicit
latency buckets, so registry resets preserve their contracts. Operator-facing
names and examples are documented in [Quizzes API](../API/Quizzes.md#generation-metrics).

## Verification

Tests exercise the public generation service with real SQLite persistence and a
real isolated metrics registry. Coverage includes every profile, mixed sources,
early rejection, malformed output, provider and verification failures,
persistence errors, cancellation, concurrent requests, legacy delegation,
monotonic timing, registry reset/export, and unavailable observability components.
