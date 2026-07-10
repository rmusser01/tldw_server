# Remote Single-User API Key Device Persistence Design

**Backlog:** TASK-12106
**Related:** TASK-12108 (same-origin HttpOnly sessions)
**Status:** Revised for specification review
**Date:** 2026-07-10

## Problem

Single-user authentication for a manually configured remote server uses an `X-API-KEY` request header. Browser password managers are not a dependable persistence mechanism for that credential. The WebUI in advanced mode and the browser extension therefore need an explicit, predictable choice between device persistence and session-only use.

Same-origin runtime-enabled WebUI deployments are deliberately excluded from browser-readable API-key storage. They use the HttpOnly session design in TASK-12108.

## Goals

- Manual single-user setup in onboarding and Settings shows `Remember on this device`, enabled by default for a new setup.
- Device persistence survives hard reloads and full browser restarts in the same browser profile or extension installation.
- Session-only persistence survives hard reloads and extension service-worker restarts, but not a full browser restart.
- A manual key is bound to the origin of the configured server and is never sent to another origin.
- WebUI and extension surfaces share the same policy and user-facing language.
- Browser password-manager prompting is not part of the contract.

## Non-Goals

- Persisting or exposing the runtime-configured key used by a same-origin WebUI.
- Replacing the same-origin HttpOnly session described in TASK-12108.
- Encrypting a browser-stored key without an independent user secret. Client-side encryption with a colocated key would not improve the XSS trust boundary.
- Syncing API keys between devices or browser profiles.
- Changing multi-user JWT or hosted-cookie behavior.
- Supporting multiple saved server profiles in this change.

## Chosen Approach

Use explicit app-owned credential persistence only for manually configured remote servers, with two scopes:

- `device`: store a manually entered key in device-local storage.
- `session`: store a manually entered key in browser-session storage.

The in-memory request configuration is hydrated from a manual key only when its recorded server origin matches the active remote server origin. Runtime/session-cookie auth has precedence and may not be copied into either manual credential store.

## Storage Model

Extend the non-secret connection metadata with:

```ts
type ApiKeyPersistence = "device" | "session"

type SingleUserCredentialMetadata = {
  source: "manual"
  persistence: ApiKeyPersistence
  serverOrigin: string
}
```

For a manual device credential:

- Persist the key in local storage explicitly scoped to the current device.
- Persist `persistence: "device"` and the normalized `serverOrigin` with connection metadata.
- In the extension, use `browser.storage.local`, never `browser.storage.sync`.

For a manual session credential:

- Persist the key and matching origin in session storage.
- Persist only the non-secret preference `persistence: "session"` in local connection metadata.
- In the extension, use `browser.storage.session` so the key survives extension service-worker restarts without surviving a browser restart.
- In the WebUI shim, session storage maps to `window.sessionStorage`, not `window.localStorage` with a prefix.

The existing `tldwConfig` object remains the connection metadata source of truth. Runtime hydration may expose an `apiKey` on the in-memory config consumed by request code, but storage writers include the key only for a manual `device` credential.

## Origin Binding

Normalize the configured server URL with `new URL(value).origin`. Every manual key is stored with that origin.

- Credential lookup returns no key when the active origin differs from the stored origin.
- A new-origin connection probe uses only the key currently submitted in the form. It bypasses the shared credential resolver so an old stored key cannot be inherited by the request.
- The active server configuration does not change until the explicit new-origin probe succeeds.
- After success, commit the transition in order: clear both old manual key stores, write the new credential to its selected scope, then publish the new server metadata and in-memory config.
- The key field is cleared when a populated form changes to a different valid origin.
- Cosmetic URL changes that preserve the origin do not clear the key.
- Invalid URLs never cause a stored key to be attached to a request.

## User Experience

Show the same inline control in onboarding and Settings for manual single-user setup:

- Label: `Remember on this device`
- Default: enabled for a new manual setup.
- Supporting text while enabled: `Stores this API key in this browser until you disconnect or clear browser data. Turn this off on a shared device.`
- Supporting text while disabled: `Keep signed in until this browser closes.`

Use the existing shared checkbox/switch vocabulary, visible focus treatment, and semantic labels. Do not add a confirmation modal.

When the same-origin cookie session from TASK-12108 is active, hide the manual API-key and persistence controls and show `Connected securely through this WebUI.` No browser-readable key is available to display or migrate.

Existing manual credentials retain their current scope during migration. A session-only credential is never silently promoted to device storage merely because the control defaults on for new entries.

## Save and Hydration Flow

### Manual Save

1. Validate the server URL and derive its origin.
2. Compare that origin with the active credential origin. If it differs, require a freshly entered key and keep the old server configuration active during the probe.
3. Validate the submitted key against the candidate server using an explicit request that cannot inject stored auth.
4. After successful validation, clear credential data associated with the old origin.
5. Write the manual credential to the selected storage scope.
6. Write non-secret connection metadata and the persistence preference.
7. Hydrate the in-memory request client from the selected scope.
8. Emit the existing config-updated event so other WebUI tabs or extension contexts refresh.

If a device-local write fails, do not claim the key was remembered. Fall back in this exact order: device-local storage, browser-session storage, then in-memory only. Show a warning that accurately names the achieved scope. Never fall back from session storage to persistent storage.

### Startup

1. Load non-secret connection metadata.
2. Resolve same-origin cookie-session/runtime auth first.
3. Only when that is absent and the transport is a manually configured remote server, load the manual key from the declared scope.
4. Verify the stored key origin matches the active server origin.
5. Hydrate request headers in memory.
6. Treat missing, invalid, or mismatched credentials as unconfigured auth.

### Clearing

Logout/disconnect, reset-all-settings, auth-mode changes, and server-origin changes clear:

- Device-local manual key material.
- Session manual key material.
- Manual credential metadata and runtime overrides derived from manual storage.

Network errors, timeouts, and server `5xx` responses do not clear credentials. A `401` or `403` may mark authentication invalid, but automatic deletion is out of scope; the user can replace or clear the credential explicitly.

## Migration

- Existing valid manual keys already stored persistently remain `device` credentials and gain origin metadata on the next successful load or save only in a remote/manual context where no same-origin cookie session succeeded.
- A legacy `tldwRuntimeSessionSingleUserApiKey` bridge is migrated as a manual `session` credential only when all of the following hold: the stored config is single-user, it has a valid remote server URL, no current same-origin cookie-session/runtime config is available, and runtime ownership metadata does not fingerprint-match the bridge key.
- After a same-origin cookie-session bootstrap and authenticated probe succeed, clear `tldwConfig.apiKey`, all legacy bridge/runtime slots, and every incomplete or ambiguous single-user secret record before publishing readiness. Missing, malformed, partially written, or contradictory ownership metadata is evidence to scrub, never evidence to preserve or reclassify.
- Only a complete new-format record with explicit `source: "manual"`, persistence scope, normalized remote origin, and matching non-secret manual connection metadata may survive that scrub. Preserve non-secret metadata even when its ambiguous secret is removed.
- In a remote/manual context, if legacy ownership cannot be distinguished confidently or the server origin cannot be derived, do not migrate or persist the bridge key.
- Placeholder keys are ignored.
- Configurations with an unparseable server URL do not hydrate a manual key until the URL is corrected.

Migration is idempotent and never moves a secret from session storage into local storage without a subsequent explicit manual save.

## Error Handling

- Storage unavailable or quota exceeded: connect session-only and show a non-blocking persistence warning.
- Session storage unsupported: fall back to in-memory use for the current page/context and disclose that reload persistence is unavailable.
- Same-origin session bootstrap unavailable: show the normal connection error for that deployment; do not expose the runtime key or silently convert to manual storage.
- Origin mismatch: suppress the key and show the normal API-key-required state.
- Corrupt credential metadata: ignore and clear the corrupt manual credential record without logging secret material.

Logs and diagnostics may report credential source, persistence scope, and origin-match state, but never the key value.

## Test Strategy

### Unit and Component Tests

- New manual forms default `Remember on this device` to enabled.
- Existing session metadata renders the choice disabled.
- Same-origin cookie-session auth hides manual key and persistence controls.
- Device save writes only device-local secret storage.
- Session save writes only session secret storage.
- Cookie/runtime auth is never copied to manual storage.
- Origin mismatch returns no credential and clears the form value before connection.
- Logout/reset/auth-mode changes clear both scopes.
- Persistent-write failure falls back to session-only with a warning.
- Legacy migration fails closed for both missing and malformed runtime-ownership metadata.
- Upgraded same-origin profiles with an API key in `tldwConfig` and missing, malformed, partially written, or contradictory ownership metadata scrub the key after the cookie-authenticated probe succeeds.

### Remote WebUI Browser Tests

- Manual device save, hard reload, authenticated route remains available.
- Manual device save, close browser, reopen the same persistent profile, authenticated route remains available.
- Manual session save, hard reload, authenticated route remains available.
- Manual session save, close browser, reopen the same persistent profile, manual key is absent.

The close/reopen test launches two browser processes against the same Playwright `userDataDir`; opening another page in the same context is insufficient.

### Extension Browser Tests

- Enter and save a manual device key through the real options/onboarding UI.
- Close the persistent browser context.
- Relaunch with the same `userDataDir` and the same unpacked extension path.
- Verify the extension ID/installation is unchanged, connection config is present, and authenticated UI is available without entering the key again.
- Cover session-only behavior separately by confirming the session key is absent after relaunch.

Tests exercise the UI save path rather than seeding `tldwConfig` directly.

## Security and Privacy

Device persistence intentionally stores a manually supplied remote-server API key as browser-readable secret material. The checked-by-default control is visible and explains the shared-device risk. This exception is required because a remote API origin and an extension origin cannot share the same host-only session cookie.

Controls limiting the risk:

- Same-origin deployments use TASK-12108 instead of browser-readable key storage.
- Manual keys are bound to one server origin.
- Extension keys are device-local, not synced.
- Session-only mode is available inline.
- Explicit disconnect/reset clears both stores.
- Secret values are excluded from logs, errors, diagnostics, and test artifacts.

## Rollout and Compatibility

- Land TASK-12108 and this task as linked, independently reviewable commits.
- Keep legacy configuration keys readable during migration.
- Ship remote WebUI and extension support together so shared onboarding copy does not promise behavior one surface lacks.
- Document the device-persistence security trade-off in setup/help text.
- Do not remove legacy readers until upgraded and fresh-profile regression coverage passes.
