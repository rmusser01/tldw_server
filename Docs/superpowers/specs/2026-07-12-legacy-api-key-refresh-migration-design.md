# Legacy Single-User API Key Refresh Migration

**Backlog task:** TASK-12950

**Status:** Independently reviewed; pending requester review

**Date:** 2026-07-12

## Problem

Before device-persistence metadata was introduced, the WebUI and browser extension saved a manually entered single-user credential as a minimal `tldwConfig` record containing `authMode`, `serverUrl`, and `apiKey`. Those released writers did not include `authSource`.

Latest `dev` only recognizes a metadata-free key as migratable when `authSource` is already `manual`. A real legacy record therefore falls through to the incomplete-credential scrub path during `TldwApiClient.initialize()`. Refreshing `/media` removes the key and returns both clients to the API-key-required state.

Fresh credentials saved by latest `dev` already contain complete metadata and are unaffected.

## Scope

Correct the shared `TldwApiClient` migration so a valid credential written by pre-metadata releases remains usable after refresh in both:

- The Next.js WebUI.
- The packaged browser extension.

The change is limited to startup migration and regression coverage. It does not add UI, storage formats, authentication modes, or automatic recovery for invalid credentials.

## Classification Rules

A stored record is a legacy manual-device credential eligible for migration only when all of these conditions hold:

1. `authMode` is `single-user`.
2. `apiKey` is a non-empty, non-placeholder string.
3. `serverUrl` normalizes to a valid server origin.
4. `credentialSource`, `apiKeyPersistence`, and `apiKeyServerOrigin` properties are genuinely absent, identifying the pre-metadata shape. Present-but-empty, invalid, or otherwise falsy values are malformed metadata, not legacy absence.
5. No active same-origin cookie session has been restored.
6. No environment API key is available for the active connection.
7. No runtime single-user API key override is available.
8. `authSource` is either absent, as written by older releases, or `manual`. A contradictory source is not migrated.

Eligible records are rewritten atomically as the same manual credential with:

- `authSource: "manual"`
- `credentialSource: "manual"`
- `apiKeyPersistence: "device"`
- `apiKeyServerOrigin` set to the normalized `serverUrl` origin

Complete current-format device credentials remain unchanged. Records with invalid origins, placeholder keys, partial or contradictory credential metadata, or replacement cookie/environment/runtime authentication continue through the existing fail-closed handling and do not hydrate the legacy key.

## Data Flow

Both clients use the same initialization path:

1. Read device-local `tldwConfig`, including the existing sync-to-local compatibility move.
2. Resolve cookie, environment, and runtime replacement authentication.
3. Classify a metadata-free single-user record using the rules above.
4. Rewrite an eligible record with complete manual/device/origin metadata before resolving the effective configuration.
5. Resolve the migrated key only when its recorded origin matches the active server origin.
6. Build authenticated media requests from the effective configuration.

The migration is idempotent: after the first successful rewrite, the record is a complete current-format device credential and no longer enters the legacy branch.

## Security and Failure Behavior

The migration does not move or newly persist a secret; the key is already in device-local persistent storage. It adds ownership, scope, and origin metadata that constrains when the existing secret may be used.

Cookie, environment, and runtime credentials retain precedence. A legacy key is never reclassified when one of those sources is active. Origin normalization and later origin matching prevent a migrated key from being sent to a different server.

Storage write failures retain the existing behavior: initialization may use only configuration that the current credential resolver considers valid, and no secret values are logged. This correction does not introduce a new retry or warning mechanism.

## Test Strategy

Development follows test-driven implementation:

1. Replace the unit expectation that a bare, valid legacy manual record is scrubbed with a failing regression test that expects complete manual/device/origin metadata and a usable key after initialization.
2. Retain or add fail-closed unit cases for contradictory `authSource`, partial metadata, invalid origin, placeholder key, active cookie session, environment key, and runtime override.
3. Add a WebUI browser regression that seeds the pre-metadata record, opens `/media`, performs a hard refresh, and verifies the authenticated media state and request header remain intact.
4. Add the equivalent packaged-extension regression using extension device storage and `options.html#/media`.
5. Run the existing authentication bootstrap, credential lifecycle, WebUI persistence, and packaged-extension persistence suites.

Because the implementation is TypeScript-only, Python Bandit is not applicable; the task record will document that skip. Existing frontend lint/type checks remain required for touched code.

## Non-Goals

- Prompting the user to confirm migration.
- Persisting session-only, cookie-derived, environment-derived, or runtime-derived keys as device credentials.
- Guessing ownership when credential metadata is partial or contradictory.
- Changing server authentication or media endpoints.
- Broad refactoring of `TldwApiClient.initialize()`.
