# UAT261 boundary harness report

## What the harness proves

The existing `complete-v2` route assembled and dispatched two fresh, otherwise identical fixture requests through its real module-level `perform_chat_api_call` boundary. At that boundary, the harness computed the existing `prompt-v1` envelope fingerprint from the actual `messages_payload` and a SHA-256 digest from the documented, non-secret dispatch controls. The two calls produced equal message and settings fingerprints.

A synthetic streaming fake proved the redacted projection can record `finish_reason`, numeric usage, and a hashed system fingerprint. Its second terminal frame omitted usage, which remained `null`. A synthetic reasoning-only frame did not become final-answer output; the second call's missing final content is represented as `present: false`, length zero, and null hash.

## What it cannot prove

This is not provider evidence. It made no network call, did not invoke a provider adapter, and used synthetic terminal metadata. It cannot reconstruct the historical UAT261 request because that request did not retain an outbound message or settings fingerprint. It also cannot attribute a differing real result to the application or provider, and it does not add durable production observability.

## Boundaries retained

The harness patches only `character_chat_sessions.perform_chat_api_call`. It uses the existing Character_Chat fixture credential-binding seam to exercise the authentic route call contract, but writes no credential, prompt, provider body, reasoning, header, user identifier, or configuration value to evidence.

See `RUN.md` for the exact run, first-three-attempt reassessment, and verification. See `boundary-harness-evidence.json` for the safe projection.

`fixture_runner.py` preserves the temporary pytest shim required for independent replay; it is not currently copied into the maintained test tree.
