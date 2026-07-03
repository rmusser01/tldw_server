# ADR-027: Security AES-GCM JSON Envelope Helpers

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md`
**Decision owner:** TASK-2312 adoption audit and TASK-2313 crypto-envelope backfill scope
**Related task:** TASK-2313 (`backlog/tasks/task-2313 - Backfill-Security-crypto-envelope-ADR.md`)
**Related spec/plan:** `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md`

## Decision

Configured encrypted persistence paths use the Security module's AES-GCM JSON envelope helpers as the shared primitive for storing and rotating sensitive structured metadata, while keeping encryption opt-in or required according to each caller's existing boundary.

## Context

TASK-2312 found that the remaining secrets/serialization portion of INV-029 was too broad for one accepted ADR. The audit did not support a universal `SecretManager` adoption claim or a universal safe-serialization claim, but it did find a focused shared crypto primitive with multiple active consumers.

`tldw_Server_API/app/core/Security/crypto.py` provides:

- `encrypt_json_blob()` and `decrypt_json_blob()` using `WORKFLOWS_ARTIFACT_ENC_KEY`.
- `encrypt_json_blob_with_key()` and `decrypt_json_blob_with_key()` for caller-supplied keys.
- AES-GCM envelopes marked with `_enc: aesgcm:v1`, plus base64-encoded `nonce`, `ct`, and `tag` fields.
- Primary-key decrypt behavior with optional `JOBS_CRYPTO_SECONDARY_KEY` fallback for rotation windows.
- Failure-safe return behavior where unsupported crypto, missing keys, invalid envelopes, or decrypt failures return `None` instead of exposing plaintext or partial data.

Known consumer patterns include Jobs payload/result encryption and key rotation, External Sources OAuth state/token envelope handling, AuthNZ user provider secrets, admin webhook secrets, and Workflow metadata decrypt/encrypt paths.

This ADR is intentionally bounded. It accepts the shared AES-GCM JSON envelope primitive and the existing encrypted-persistence consumer pattern. It does not claim all sensitive JSON in the repository is encrypted, that all secret lookup flows use `SecretManager`, or that restricted pickle compatibility is part of this decision.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Let each persistence feature define its own encrypted JSON envelope format | Divergent envelope formats make key rotation, decrypt fallback, tests, and future migrations harder to reason about. |
| Require AES-GCM encryption for every JSON blob immediately | Current consumers have different boundaries: Jobs encryption is domain/config gated, connector token storage can fall back when crypto is unavailable, while BYOK/admin webhook helpers require configured keys. A universal requirement would overclaim current behavior. |
| Fold `SecretManager`, AES-GCM helpers, and restricted pickle into one Security ADR | TASK-2312 found different adoption levels and boundaries. Combining them would imply repository-wide guarantees the code does not currently provide. |
| Store sensitive structured metadata as plaintext plus redaction only | Redaction helps logs and UI surfaces, but it does not protect persisted structured metadata. Existing encrypted-persistence paths already use envelopes where configured or required. |
| Replace AES-GCM helpers with caller-specific key-management services | Some callers have their own key source, but the shared explicit-key helpers already let them use a common envelope format without centralizing every key policy. |

## Consequences

New encrypted persistence paths for structured JSON should prefer `tldw_Server_API.app.core.Security.crypto` helpers and the `_enc: aesgcm:v1` envelope shape instead of inventing a local envelope format.

Callers remain responsible for their key boundary:

- Jobs encryption stays gated by `JOBS_ENCRYPT` / `JOBS_ENCRYPT_<DOMAIN>` plus configured crypto keys.
- Generic Security crypto helpers use `WORKFLOWS_ARTIFACT_ENC_KEY` and optional `JOBS_CRYPTO_SECONDARY_KEY`.
- BYOK and admin webhook secret helpers use explicit-key variants and can require configured keys at their own boundary.
- Existing connector paths may preserve plaintext fallback behavior when crypto is unavailable or not configured.

Key rotation and migration work should use the explicit-key helpers when old and new keys must be supplied directly. Tests should cover envelope creation, decrypt behavior, invalid-envelope handling, key fallback, and caller-specific fallback or required-key behavior.

This ADR does not replace `SecretManager`, require every secret read to flow through `SecretManager`, or decide restricted pickle policy. Those remain separate inventory-only or future implementation-backed slices.

## Follow-up

- Use this ADR as the covering record for the AES-GCM JSON envelope portion of INV-029.
- Keep `Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md` as the evidence and caveat record for this split.
- Consider separate follow-up work for `SecretManager` adoption and restricted legacy pickle compatibility if the owner wants those decisions promoted from inventory-only status.
