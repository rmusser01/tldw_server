# Security Secrets and Serialization Adoption Audit - 2026-06-07

**Related task:** TASK-2312
**Inventory row:** INV-029
**Source candidate:** `tldw_Server_API/app/core/Security/README.md`
**Disposition:** Keep inventory-only for now. The helpers are real and tested, but current adoption does not support one accepted ADR that claims repository-wide secret management or serialization policy.

## Decision Candidate Under Review

INV-029 originally grouped Security secret management and safe serialization with egress and request-edge controls:

> Security controls are centralized for egress policy, security headers, request IDs, setup CSP/access guard, URL validation, and secret management; production should keep security middleware enabled.

ADR-019 now covers request-edge middleware and ADR-026 covers outbound egress/SSRF. This audit reviews the remaining secrets and serialization portion: `SecretManager`, AES-GCM JSON helpers in `crypto.py`, and restricted pickle compatibility in `safe_pickle.py`.

## Evidence Summary

| Area | Evidence | Result |
| --- | --- | --- |
| `SecretManager` helper availability | `tldw_Server_API/app/core/Security/secret_manager.py` defines `SecretManager`, configured secret metadata, source precedence from environment to config to default, optional cache metadata, startup validation, health checks, and convenience functions such as `get_api_key()`, `get_auth_secret()`, `get_webhook_secret()`, and `validate_production_secrets()`. Tests in `tldw_Server_API/tests/Security/test_secret_manager.py` cover override immutability and sanitized health/error output. | Confirmed as a helper surface for configured secrets. Not confirmed as a repository-wide secret retrieval policy. |
| `SecretManager` caller adoption | Source search found app-level references to the Security `SecretManager` only inside `secret_manager.py` itself. Separate `TriggerSecretManager` usage exists for ACP triggers, but it is a distinct ACP-specific encryption helper. Many current modules read secrets or API keys directly from environment/config, including AuthNZ, Chat, LLM/TTS providers, Image Generation, External Sources connectors, workflows/webhooks, Third Party integrations, and configuration loading. | Do not write an ADR claiming secret lookup is centralized or universally adopted. A future SecretManager ADR needs an implementation/adoption slice first. |
| AES-GCM JSON helper availability | `tldw_Server_API/app/core/Security/crypto.py` provides `encrypt_json_blob()`, `decrypt_json_blob()`, explicit-key variants, `WORKFLOWS_ARTIFACT_ENC_KEY`, `JOBS_CRYPTO_SECONDARY_KEY`, and AES-GCM envelopes marked `_enc: aesgcm:v1`. Tests in `tldw_Server_API/tests/Security/test_crypto.py` cover invalid-envelope failure behavior. | Confirmed as the shared Security crypto primitive. |
| AES-GCM JSON helper consumers | Known consumers include Jobs payload/result encryption and key rotation in `tldw_Server_API/app/core/Jobs/manager.py`, External Sources OAuth state/token envelope handling in `tldw_Server_API/app/core/External_Sources/connectors_service.py`, AuthNZ user provider secrets in `tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py`, admin webhook secrets in `tldw_Server_API/app/core/AuthNZ/admin_webhook_secrets.py`, and Workflow metadata decrypt/encrypt paths in `tldw_Server_API/app/core/Workflows/engine.py` and `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`. Related tests cover connector token encryption, OAuth state metadata encryption, Jobs encryption, and key rotation. | Stronger candidate for a future bounded ADR, but still should not be combined with a universal SecretManager claim. |
| Restricted pickle helper availability | `tldw_Server_API/app/core/Security/safe_pickle.py` defines `RestrictedUnpickler` and `safe_pickle_loads()`, allowing only basic built-in containers and `collections.OrderedDict`. | Confirmed as the Security-owned restricted legacy pickle helper. |
| Restricted pickle consumers | `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py` uses `safe_pickle_loads()` only for legacy content-hash migration and only when `WEBSCRAPER_ALLOW_LEGACY_PICKLE_HASHES` is enabled. `tldw_Server_API/app/core/Scheduler/services/payload_service.py` uses `safe_pickle_loads()` only for legacy scheduler payloads and only when `allow_legacy_pickle_payloads` / `SCHEDULER_ALLOW_LEGACY_PICKLE_PAYLOADS` enables compatibility mode. Tests confirm default-disabled behavior and rejection of disallowed globals. | Confirmed for bounded legacy compatibility paths. Not universal serialization policy. |
| Serialization divergence | `tldw_Server_API/app/core/Embeddings/multi_tier_cache.py` defines its own local restrictive unpickler rather than using `Security.safe_pickle`. This is not necessarily wrong for cache-local data, but it means the Security helper is not the universal pickle boundary. | Do not backfill a broad safe-serialization ADR without either narrowing it to known compatibility paths or consolidating local implementations first. |

## Disposition

Do not create an accepted ADR for the remaining secrets/serialization portion of INV-029 in its current shape.

The current evidence supports these narrower statements:

- Security provides a `SecretManager` helper with source precedence, validation, cache metadata, health checks, and sanitized test coverage.
- Security provides AES-GCM JSON envelope helpers that several Jobs, AuthNZ, External Sources, and Workflows paths use for optional or configured encrypted persistence.
- Security provides a restricted pickle loader used by bounded legacy compatibility paths in Web Scraping and Scheduler.

The current evidence does not support these broader ADR claims:

- All repository secrets are retrieved through `SecretManager`.
- All sensitive stored JSON is encrypted through `Security.crypto`.
- All pickle deserialization routes through `Security.safe_pickle`.

## Recommended Next Action

Keep INV-029 partially backfilled. ADR-019 covers request-edge middleware, ADR-026 covers outbound egress/SSRF, and this audit records why secrets/serialization remains inventory-only.

If the owner wants more ADR work here, split it into implementation-backed slices:

1. SecretManager adoption slice: migrate or explicitly exempt direct secret reads before considering any "centralized secret lookup" ADR.
2. Crypto envelope ADR slice: backfill only the shared AES-GCM JSON envelope primitive and known encrypted persistence consumers.
3. Restricted legacy pickle ADR slice: backfill only the default-disabled legacy compatibility rule, or first consolidate the Embeddings cache local unpickler if one central helper is desired.
