# ADR-028: Security Restricted Legacy Pickle Compatibility

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md`
**Decision owner:** TASK-2312 adoption audit and TASK-2314 restricted-pickle backfill scope
**Related task:** TASK-2314 (`backlog/tasks/task-2314 - Backfill-Security-restricted-pickle-compatibility-ADR.md`)
**Related spec/plan:** `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md`

## Decision

Legacy pickle deserialization is allowed only through the Security module's restricted pickle helper for explicitly gated legacy compatibility paths, not as a general-purpose serialization format.

## Context

TASK-2312 found that the remaining secrets/serialization portion of INV-029 was too broad for one accepted ADR. The audit did not support a universal safe-serialization claim, but it did find a focused `safe_pickle` helper with bounded, default-disabled legacy consumers.

`tldw_Server_API/app/core/Security/safe_pickle.py` provides:

- `RestrictedUnpickler`, which blocks arbitrary global class and function resolution.
- `safe_pickle_loads()`, which accepts only bytes-like payloads and routes deserialization through `RestrictedUnpickler`.
- An allowlist limited to basic built-in value containers and `collections.OrderedDict`.
- `ValueError` wrapping for unsafe pickle payloads rejected by the restricted unpickler.

Known consumer patterns are intentionally narrow:

- Web Scraping `ContentDeduplicator` migrates legacy `content_hashes.pkl` only when `WEBSCRAPER_ALLOW_LEGACY_PICKLE_HASHES=true`; migrated data is normalized and saved back to JSON.
- Scheduler `PayloadService` loads legacy pickle payloads only when `allow_legacy_pickle_payloads` / `SCHEDULER_ALLOW_LEGACY_PICKLE_PAYLOADS` enables compatibility mode.
- Scheduler only writes pickle payloads for non-JSON-serializable payloads when that same compatibility mode is enabled.

This ADR is intentionally bounded. It accepts the restricted helper and the known gated legacy compatibility pattern. It does not claim all pickle deserialization in the repository routes through `Security.safe_pickle`, does not consolidate the Embeddings cache-local unpickler, and does not cover `SecretManager` adoption.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Continue using raw `pickle.load()` / `pickle.loads()` in legacy paths | Raw pickle can resolve arbitrary globals. Existing migration and payload paths already have a restricted helper and tests for disallowed globals. |
| Remove all legacy pickle compatibility immediately | Web Scraping and Scheduler still have explicit compatibility paths for older stored data. Removing them would strand legacy users without a migration path. |
| Enable legacy pickle compatibility by default | Pickle is a high-risk format. The current callers require explicit environment or configuration gates before loading legacy pickle payloads. |
| Treat `Security.safe_pickle` as the universal pickle boundary for the repository | The Embeddings cache currently has its own local restrictive unpickler. Calling this universal would overclaim current implementation. |
| Combine restricted pickle, AES-GCM envelopes, and `SecretManager` adoption into one Security ADR | TASK-2312 found different adoption levels and boundaries. Combining them would imply repository-wide guarantees the code does not currently provide. |

## Consequences

New persistence and payload formats should prefer JSON or another explicit safe format. Pickle compatibility is for bounded legacy migration or compatibility paths only.

Callers that must support legacy pickle data must:

- route loads through `tldw_Server_API.app.core.Security.safe_pickle.safe_pickle_loads()`;
- keep compatibility default-disabled or explicitly gated;
- document the compatibility flag or configuration boundary;
- normalize and migrate legacy data to a safer format where practical;
- test default-disabled behavior and rejection of disallowed globals.

Existing cache-local restrictive unpicklers remain outside this ADR unless a future implementation consolidates them into `Security.safe_pickle`. This ADR also does not replace `SecretManager`, require every secret read to flow through `SecretManager`, or create a broad safe-serialization policy for every module.

## Follow-up

- Use this ADR as the covering record for the restricted legacy pickle compatibility portion of INV-029.
- Keep `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md` as the evidence and caveat record for this split.
- Consider separate follow-up work for `SecretManager` adoption if the owner wants that decision promoted from inventory-only status.
