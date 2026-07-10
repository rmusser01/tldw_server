# Single-User API Key Device Persistence Design

**Backlog:** TASK-12106  
**Status:** Approved for specification review  
**Date:** 2026-07-10

## Problem

Single-user authentication uses an `X-API-KEY` request header rather than a browser login cookie. Browser password managers therefore are not a dependable persistence mechanism. Same-origin WebUI deployments can already receive runtime-provisioned auth, but manually configured WebUI and extension users need an explicit, predictable choice between device persistence and session-only use.

The current code has multiple credential readers and writers. Some branches persist a manual key in `tldwConfig`; others scrub it and retain a session-only runtime copy. Without one policy, hard reloads, browser restarts, background-worker restarts, and auth-mode changes can disagree about whether the user is authenticated.

## Goals

- Same-origin single-user WebUI deployments use runtime-provisioned auth without asking the user for an API key.
- Manual single-user setup in onboarding and Settings shows `Remember on this device`, enabled by default for a new setup.
- Device persistence survives hard reloads and full browser restarts in the same browser profile or extension installation.
- Session-only persistence survives hard reloads and extension service-worker restarts, but not a full browser restart.
- A manual key is bound to the origin of the configured server and is never sent to another origin.
- WebUI and extension surfaces share the same policy and user-facing language.
- Browser password-manager prompting is not part of the contract.

## Non-Goals

- Adding a backend login endpoint or HttpOnly cookie session for single-user mode.
- Encrypting a browser-stored key without an independent user secret. Client-side encryption with a colocated key would not improve the XSS trust boundary.
- Syncing API keys between devices or browser profiles.
- Changing multi-user JWT or hosted-cookie behavior.
- Supporting multiple saved server profiles in this change.

## Chosen Approach

Use explicit app-owned credential persistence with two scopes:

- `device`: store a manually entered key in device-local storage.
- `session`: store a manually entered key in browser-session storage.

Runtime-provisioned keys remain runtime-owned and are never copied into either manual credential store. The in-memory request configuration is hydrated from exactly one source according to precedence:

1. Same-origin runtime-provisioned single-user key.
2. Manual key whose recorded server origin matches the active server origin.
3. No single-user credential.

Environment/build-time runtime auth remains part of the runtime-provisioned category and keeps its existing precedence.

## Storage Model

Extend the non-secret connection metadata with:

```ts
type ApiKeyPersistence = "device" | "session"
type ApiKeySource = "runtime" | "manual"

type SingleUserCredentialMetadata = {
  source: ApiKeySource
  persistence?: ApiKeyPersistence
  serverOrigin: string
}
```

The runtime source is represented only in memory or by non-secret ownership metadata. Its API key value must not be persisted by manual save logic.

For a manual device credential:

- Persist the key in local storage explicitly scoped to the current device.
- Persist `persistence: "device"` and the normalized `serverOrigin` with the connection metadata.
- In the extension, use `browser.storage.local`, never `browser.storage.sync`.

For a manual session credential:

- Persist the key and matching origin in session storage.
- Persist only the non-secret preference `persistence: "session"` in local connection metadata.
- In the extension, use `browser.storage.session` so the key survives extension service-worker restarts without surviving a browser restart.
- In the WebUI shim, session storage must map to `window.sessionStorage`, not `window.localStorage` with a prefix.

The existing `tldwConfig` object remains the connection metadata source of truth. Runtime hydration may expose an `apiKey` on the in-memory config consumed by request code, but storage writers must include the key only for a manual `device` credential.

## Origin Binding

Normalize the configured server URL with `new URL(value).origin`. Every manual key is stored with that origin.

- Credential lookup returns no key when the active origin differs from the stored origin.
- Saving or connecting to a different origin clears both manual key stores before any request can use the previous key.
- The key field is cleared when a populated form changes to a different valid origin.
- Cosmetic URL changes that preserve the origin, such as a trailing slash or path normalization, do not clear the key.
- Invalid URLs never cause a stored key to be attached to a request.

## User Experience

### Runtime-Provisioned Same-Origin WebUI

When runtime auth is available:

- Hide the manual API-key input and persistence checkbox.
- Show a compact informational state: `Connected using this server's configured API key.`
- Do not copy the runtime key into manual storage.
- If runtime auth becomes unavailable, return to the manual setup state without displaying the old key.

### Manual Single-User Setup

Show the same inline control in onboarding and Settings:

- Label: `Remember on this device`
- Default: enabled for a new manual setup.
- Supporting text while enabled: `Stores this API key in this browser until you disconnect or clear browser data. Turn this off on a shared device.`
- Supporting text while disabled: `Keep signed in until this browser closes.`

Use the existing shared checkbox/switch vocabulary, visible focus treatment, and semantic labels. Do not add a confirmation modal.

Existing manual credentials retain their current scope during migration. A session-only credential must not be silently promoted to device storage merely because the new-control default is enabled for new entries.

## Save and Hydration Flow

### Manual Save

1. Validate the server URL and derive its origin.
2. Validate the API key against that server.
3. Clear credential data associated with a different origin.
4. Write the manual credential to the selected storage scope.
5. Write non-secret connection metadata and the persistence preference.
6. Hydrate the in-memory request client from the selected scope.
7. Emit the existing config-updated event so other WebUI tabs or extension contexts refresh.

If the selected persistent write fails, do not claim that the key was remembered. Keep the successful connection session-only and show a warning that the user will need to enter the key after closing the browser.

### Startup

1. Load non-secret connection metadata.
2. Resolve runtime auth first.
3. If runtime auth is absent, load the manual key from the declared scope.
4. Verify the stored key origin matches the active server origin.
5. Hydrate request headers in memory.
6. Treat missing, invalid, or mismatched credentials as unconfigured auth, not as a successful anonymous connection.

### Clearing

Logout/disconnect, reset-all-settings, auth-mode changes, and server-origin changes clear:

- Device-local manual key material.
- Session manual key material.
- Manual credential metadata and runtime overrides derived from manual storage.

Network errors, timeouts, and server `5xx` responses do not clear credentials. A `401` or `403` may mark authentication invalid, but automatic deletion is out of scope; the user can replace or clear the credential explicitly.

## Migration

- Existing valid manual keys already stored persistently remain `device` credentials and gain origin metadata on the next successful load or save.
- Existing session-only runtime bridges remain `session` credentials for the current browser session.
- Existing runtime-owned keys identified by runtime ownership metadata remain runtime-owned and are never reclassified as manual.
- Placeholder keys are ignored.
- Configurations with an unparseable server URL do not hydrate a manual key until the URL is corrected.

Migration must be idempotent and must not move a secret from session storage into local storage without a subsequent explicit manual save.

## Error Handling

- Storage unavailable or quota exceeded: connect session-only and show a non-blocking persistence warning.
- Session storage unsupported: fail closed to in-memory use for the current page/context and disclose that reload persistence is unavailable.
- Runtime-config fetch unavailable: fall back to manual setup; do not display a raw fetch error on the key field.
- Origin mismatch: suppress the key and show the normal API-key-required state.
- Corrupt credential metadata: ignore and clear the corrupt manual credential record without logging secret material.

Logs and diagnostics may report credential source, persistence scope, and origin match state, but never the key value.

## Test Strategy

### Unit and Component Tests

- New manual forms default `Remember on this device` to enabled.
- Existing session metadata renders the choice disabled.
- Runtime auth hides the manual key and persistence controls.
- Device save writes only device-local secret storage.
- Session save writes only session secret storage.
- Runtime keys are never copied to manual storage.
- Origin mismatch returns no credential and clears the form value before connection.
- Logout/reset/auth-mode changes clear both scopes.
- Persistent-write failure falls back to session-only with a warning.

### WebUI Browser Tests

- Manual device save, hard reload, authenticated route remains available.
- Manual device save, close browser, reopen the same persistent profile, authenticated route remains available.
- Manual session save, hard reload, authenticated route remains available.
- Manual session save, close browser, reopen the same persistent profile, manual key is absent.
- Runtime-provisioned same-origin auth reaches an authenticated route with no manual key persisted.

The close/reopen test must launch two browser processes against the same Playwright `userDataDir`; opening another page in the same context is insufficient.

### Extension Browser Tests

- Enter and save a manual device key through the real options/onboarding UI.
- Close the persistent browser context.
- Relaunch with the same `userDataDir` and the same unpacked extension path.
- Verify the extension ID/installation is unchanged, connection config is present, and authenticated UI is available without entering the key again.
- Cover session-only behavior separately by confirming the session key is absent after relaunch.

Tests must exercise the UI save path rather than seeding `tldwConfig` directly.

## Security and Privacy

Device persistence intentionally stores the single-user API key as browser-readable secret material. The checked-by-default control is visible and explains the shared-device risk. This does not expand the trust boundary beyond the existing WebUI/extension JavaScript runtime, but it does increase persistence after local compromise or XSS.

The following controls limit that risk:

- Runtime keys are never copied into manual storage.
- Manual keys are bound to one server origin.
- Extension keys are device-local, not synced.
- Session-only mode is available inline.
- Explicit disconnect/reset clears both stores.
- Secret values are excluded from logs, errors, diagnostics, and test artifacts.

An HttpOnly cookie session remains the appropriate future design if the product later requires persistent single-user auth without exposing a reusable API key to JavaScript.

## Rollout and Compatibility

- Keep existing configuration keys readable during migration.
- Add focused tests before changing storage behavior.
- Ship WebUI and extension support together so shared onboarding copy does not promise behavior one surface lacks.
- Document the device-persistence security trade-off in setup/help text.
- Do not remove legacy readers until the migration has shipped and regression coverage confirms both upgraded and fresh profiles.

