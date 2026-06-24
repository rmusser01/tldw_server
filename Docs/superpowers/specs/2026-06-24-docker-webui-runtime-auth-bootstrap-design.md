# Docker WebUI Runtime Auth Bootstrap Design

Date: 2026-06-24
Status: Draft for user review
Owner: Codex brainstorming session
Backlog: TASK-2360

## Summary

Fix the Docker single-user WebUI startup path by moving WebUI API-key delivery from build-time `NEXT_PUBLIC_X_API_KEY` into a tightly gated runtime bootstrap served by the Next.js WebUI container. Keep the WebUI image generic, preserve same-origin quickstart networking, and configure the backend setup path so Docker WebUI setup writes can succeed with admin auth.

This design also resolves the original review checklist:

- the root `mcp_unified/` Docker install item is stale for this checkout because the active MCP implementation is in `tldw_Server_API/app/core/MCP_unified`, not a root installable package
- the internal WebUI proxy origin should remain `http://app:8000`
- setup onboarding calls should remain authenticated
- the backend app service needs `TLDW_SETUP_ALLOW_REMOTE=1` for Docker setup writes
- Docker quickstart should no longer require users to export `NEXT_PUBLIC_X_API_KEY` before rebuilding the WebUI image

## Problem

The current Docker WebUI overlay still accepts `NEXT_PUBLIC_X_API_KEY` as a build arg. That means the single-user API key can be baked into the WebUI bundle at image build time. Any manual rebuild that omits or mismatches the key can produce a WebUI image that always sends the wrong credential, resulting in repeated `401` responses even when the backend is correctly configured.

The setup flow has a second Docker-specific problem. The backend treats requests from the WebUI container as remote rather than loopback. Authenticated setup requests still need `TLDW_SETUP_ALLOW_REMOTE=1` on the backend app service before setup writes are accepted from the WebUI container network.

The original review list also included a root `mcp_unified/` package install change for the production Dockerfile. In this branch, that directory does not exist at the repo root, and the MCP implementation is packaged inside `tldw_Server_API`. Adding an unconditional `COPY mcp_unified /app/mcp_unified` would break this checkout instead of fixing it.

## Goals

- Make Docker single-user WebUI auth work after runtime key changes without rebuilding the WebUI image.
- Keep `tldw-webui:prod` reusable across single-user installs instead of key-specific.
- Preserve same-origin quickstart browser networking through the existing WebUI proxy.
- Ensure setup writes from the WebUI container are allowed only when admin auth is present.
- Avoid overwriting user-entered WebUI credentials with Docker runtime credentials.
- Keep advanced/static deployments able to use `NEXT_PUBLIC_X_API_KEY` when they deliberately choose build-time public config.
- Document the stale `mcp_unified` checklist item accurately for this branch.

## Non-Goals

- Redesign the entire AuthNZ system.
- Add a new backend endpoint that exposes real API keys.
- Make remote public single-user WebUI deployments expose API keys by default.
- Remove the existing `NEXT_PUBLIC_X_API_KEY` compatibility path for non-Docker or static builds.
- Add a root `mcp_unified/` package where this branch does not currently have one.

## Current State

### Docker

- `Dockerfiles/docker-compose.webui.yml` builds the WebUI with `NEXT_PUBLIC_X_API_KEY` as an optional build arg.
- The WebUI service runtime environment does not receive `AUTH_MODE`, `SINGLE_USER_API_KEY`, or an explicit runtime-auth exposure flag.
- `TLDW_INTERNAL_API_ORIGIN` already defaults to `http://app:8000` in the WebUI overlay and Dockerfile.
- `Dockerfiles/docker-compose.single-user.yml` does not set `TLDW_SETUP_ALLOW_REMOTE=1` on the `app` service.

### WebUI

- `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts` seeds `tldwConfig` from build-time public env values.
- `_app.tsx` decides whether the app is authenticated by loading the configured `TldwApiClient` state.
- `TldwConfig` is a shared client contract with `serverUrl`, `apiKey`, tokens, org ID, and `authMode`; it should not grow Docker runtime ownership metadata.
- Existing config update events use `tldw:config-updated`.

### Backend Setup

- Setup dependencies already require admin auth for remote setup requests.
- The backend also blocks setup writes from non-loopback clients unless `TLDW_SETUP_ALLOW_REMOTE=1` or the equivalent config flag is enabled.
- Setup onboarding calls in the current checkout are not using the old unauthenticated `noAuth: true` pattern for the relevant setup admin calls.

### MCP Unified

- There is no root-level `mcp_unified/` directory in this checkout.
- `pyproject.toml` package discovery covers `tldw_Server_API*`.
- MCP Unified lives under `tldw_Server_API/app/core/MCP_unified`.

## Proposed Design

### 1. Add a WebUI-local runtime config endpoint

Add a Next.js `pages/api` route under a WebUI-private path such as:

```text
/api/_tldw-webui/runtime-config
```

The concrete implementation path should be:

```text
apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts
```

The route returns runtime WebUI bootstrap data from the WebUI server process. It does not call the backend and does not reuse the backend `docs-info` endpoint, because `docs-info` intentionally avoids exposing real API keys.

Runtime auth is available only when all guards pass:

- `AUTH_MODE=single_user`
- `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1`
- `SINGLE_USER_API_KEY` exists and is not a placeholder value such as `change-me`
- the browser request `Host` is loopback/local, including `localhost`, `127.0.0.1`, `[::1]`, and `::1`
- the Docker WebUI port remains bound to loopback in the default compose overlay, currently `127.0.0.1:8080:3000`
- forwarding headers such as `Forwarded`, `X-Forwarded-For`, `X-Forwarded-Host`, or `X-Real-IP` are absent

The route must ignore forwarded host headers for the exposure decision. A reverse proxy can opt into a different deployment later, but the default Docker path must not trust `X-Forwarded-Host` for deciding whether to reveal the single-user API key.

The route should return `200` with an unavailable payload when disabled rather than throwing an application error. Responses must include `Cache-Control: no-store`.

Example shape:

```json
{
  "runtimeAuth": {
    "available": true,
    "authMode": "single-user",
    "apiKey": "..."
  },
  "networking": {
    "deploymentMode": "quickstart",
    "serverUrl": ""
  }
}
```

When unavailable, omit `apiKey` and return `available: false`.

### 2. Bootstrap browser auth before app-level auth checks

Update `runtime-bootstrap.ts` so WebUI startup fetches `/api/_tldw-webui/runtime-config` with:

- `credentials: "same-origin"`
- `cache: "no-store"`

The bootstrap should export a named promise, for example `runtimeBootstrapReady`, that completes before `_app.tsx` calls `getConfiguredAuthState()`. `_app.tsx` should import that promise instead of importing bootstrap only for side effects, and `refreshAuthState()` should await it before reading configured auth state. This avoids an initial unauthenticated render when runtime auth is available but has not yet seeded storage.

The bootstrap should continue if the endpoint is missing, disabled, or fails. Existing manual config and build-time public env fallbacks remain usable.

When runtime auth wins, the bootstrap should also call the existing in-memory auth helper, for example `setRuntimeApiKey(runtimeKey)`, so shared request helpers prefer runtime auth over any stale `NEXT_PUBLIC_X_API_KEY` compiled into the bundle. The current auth helper already checks the in-memory runtime API key before build-time public env values; the implementation must make use of that precedence.

### 3. Track runtime-owned keys outside `TldwConfig`

Do not add Docker/runtime metadata to `TldwConfig`. Instead, store ownership metadata in a separate safe-storage key, for example:

```text
tldwRuntimeAuthMetadata
```

The metadata should be non-secret and only indicate that the current stored API key was written by the WebUI runtime bootstrap. A simple shape is sufficient:

```json
{
  "source": "webui-runtime",
  "authMode": "single-user",
  "fingerprint": "non-secret-key-fingerprint"
}
```

The API key still lives in `tldwConfig.apiKey` because that is how the shared API client currently injects `X-API-KEY`.

Write rules:

- if no API key is configured, write the runtime key and metadata
- if the existing key has matching runtime metadata, replace it when the runtime key changes
- if the existing key has no runtime metadata but matches the compiled `NEXT_PUBLIC_X_API_KEY` value or a known placeholder, treat it as bootstrap-owned and replace it with the runtime key
- if the existing key already equals the runtime key but lacks metadata, add runtime metadata without changing the key
- if an existing key has no runtime metadata, treat it as user-managed and do not overwrite it
- if runtime auth becomes unavailable, leave manual config alone and do not delete user-managed keys

After writing storage, dispatch `tldw:config-updated`. The implementation must update both `tldwConfig` and `tldwServerUrl` consistently when it changes server URL state, matching the existing bootstrap behavior.

### 4. Wire Docker runtime environment explicitly

Update the WebUI compose overlay so the WebUI container receives the backend auth mode and key at runtime:

```yaml
environment:
  - AUTH_MODE=${AUTH_MODE:-single_user}
  - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}
  - TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-1}
```

Keep `NEXT_PUBLIC_X_API_KEY` as an optional build arg for compatibility, but stop presenting it as the Docker quickstart requirement.

Update the backend app service in `Dockerfiles/docker-compose.single-user.yml`:

```yaml
environment:
  - TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}
```

This allows Docker WebUI setup writes from the WebUI container network, while the existing backend setup dependency still requires admin auth for remote callers.

### 5. Keep setup onboarding authenticated

Do not restore `noAuth: true` on setup write calls. The Docker fix is to deliver the runtime single-user API key to the WebUI and allow remote setup writes on the backend, not to bypass admin auth.

The implementation should add or update tests around the current setup admin client calls so setup writes use authenticated request paths.

### 6. Treat the root `mcp_unified` Docker item as stale

Do not add unconditional Dockerfile steps for a root `mcp_unified/` package in this branch. That directory is absent, and the MCP code is already in the main package tree.

Instead, verification should cover the active packaging shape:

- production Docker build installs `tldw_Server_API`
- an import smoke or equivalent package check can import the in-tree MCP Unified module
- docs or task notes should state that the old root-package remediation applies only to branches that actually contain a root installable `mcp_unified/` directory

If a future branch reintroduces a root installable `mcp_unified/` package, the Dockerfile should add conditional packaging support in that branch alongside tests proving the directory exists.

### 7. Update docs and troubleshooting copy

Documentation should say:

- Docker single-user WebUI quickstart uses runtime auth bootstrap by default
- users should not export `NEXT_PUBLIC_X_API_KEY` before normal Docker WebUI rebuilds
- `NEXT_PUBLIC_X_API_KEY` remains an advanced/static-build compatibility option
- `TLDW_INTERNAL_API_ORIGIN` should be `http://app:8000` for Docker compose quickstart
- `TLDW_SETUP_ALLOW_REMOTE=1` is expected for the single-user Docker app service

Remove or revise copy that implies Docker quickstart automatically handles runtime auth while the compose file still bakes or omits the key.

## Security Considerations

The runtime config endpoint intentionally exposes the single-user API key to browser JavaScript, because the existing single-user WebUI auth model already requires the browser to send `X-API-KEY`. The design reduces the build-time persistence problem; it does not make the browser key secret from the browser.

The exposure boundary is therefore:

- loopback/local browser access only by default
- explicit `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1`
- single-user mode only
- no forwarded-host trust
- no forwarded request headers on the default runtime-auth path
- no response caching
- default Docker compose binds the WebUI host port to loopback, not `0.0.0.0`

This is safer than baking the key into a reusable image, and it keeps accidental remote exposure out of the default compose binding.

The Host-header guard is not a substitute for keeping the quickstart port loopback-bound. Documentation should warn operators that if they publish the WebUI port beyond localhost, they should disable runtime auth bootstrap and use an explicit advanced auth configuration instead.

## Test Plan

### Unit tests

- runtime-config route returns unavailable when disabled, non-single-user, placeholder key, or non-loopback Host
- runtime-config route returns unavailable when forwarding headers are present
- runtime-config route returns available with `Cache-Control: no-store` for loopback Host and valid single-user env
- bootstrap writes runtime key only when no manual key exists
- bootstrap replaces a previous runtime-owned key when the runtime key changes
- bootstrap replaces a stale key that was seeded from `NEXT_PUBLIC_X_API_KEY`
- bootstrap does not replace user-managed keys
- bootstrap calls the in-memory runtime auth setter so runtime auth outranks stale build-time public env auth
- `_app.tsx` awaits the named runtime bootstrap promise before first auth-state calculation

### Integration or focused build tests

- WebUI quickstart config route is served by Next.js locally and is not proxied by the `/api/:path*` backend rewrite
- Docker compose config contains `TLDW_INTERNAL_API_ORIGIN=http://app:8000`
- WebUI compose passes runtime auth env values to the WebUI service
- WebUI compose keeps the default host port binding on `127.0.0.1`
- single-user app compose enables `TLDW_SETUP_ALLOW_REMOTE=1`
- production package/import smoke covers the active in-tree MCP Unified module instead of a nonexistent root package

### Manual verification

- Generate or set `SINGLE_USER_API_KEY`
- Start the Docker single-user + WebUI overlay
- Open `http://127.0.0.1:8080`
- Confirm the WebUI reaches authenticated API endpoints without `NEXT_PUBLIC_X_API_KEY`
- Confirm setup writes succeed from the WebUI path
- Rebuild only the WebUI image without `NEXT_PUBLIC_X_API_KEY` and confirm auth still works from runtime env

## Rollout

1. Add the runtime config endpoint and bootstrap storage behavior behind `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH`.
2. Wire Docker compose runtime environment and backend setup remote flag.
3. Update docs and troubleshooting to reflect runtime auth as the Docker default.
4. Add tests for endpoint guards, bootstrap precedence, compose config, and stale `mcp_unified` handling.
5. Run focused WebUI tests, backend/import checks, compose config validation, and Bandit over touched Python scope if any Python files change.

## Open Questions

- Should advanced reverse-proxy deployments have a separate allowlist variable for non-loopback runtime auth exposure, or should they continue using explicit manual auth configuration?
- Should the runtime metadata fingerprint be a short hash of the key, or should it use a non-secret generated marker that changes when the runtime key changes?

The implementation can proceed without answering these for the Docker quickstart path by keeping non-loopback runtime auth unavailable and storing only non-secret metadata.
