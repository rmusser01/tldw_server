# LLM Provider Key Settings Route Design

**Date:** 2026-06-01
**Surface:** Hosted WebUI settings and extension/options settings
**Status:** Approved in-session
**Backlog:** TASK-497

---

## Goal

Make the LLM provider key management page reachable from the WebUI and extension settings surfaces.

Users who click "Add your API key" or provider-key related settings links should land on the existing provider key management screen instead of a 404.

## Problem

The shared settings UI already has an LLM provider key management component and navigation path:

- [ProviderKeysSettings.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx)
- [settings-nav-config.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/settings-nav-config.ts)
- settings-specific shared route entry in [option-settings-route-registry.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/option-settings-route-registry.tsx)

The route is not wired consistently across the actual app shells that can expose the settings navigation. In particular, the hosted Next.js WebUI has no `pages/settings/provider-keys.tsx`, and the extension route registry does not define `/settings/provider-keys`. Links to that path can therefore fall through to a 404 even though the page component exists.

## Product Decision

This flow is for LLM provider key management, not tldw server authentication.

Keep these concepts separate:

- `/settings/tldw`: server URL, single-user API key, multi-user login, and connection health.
- `/settings/provider-keys`: provider credentials for OpenAI, Anthropic, Google, Groq, and other LLM/TTS/provider integrations managed by the existing BYOK page.

Do not redirect provider-key links to `/settings/tldw`; that would fix the 404 while sending users to the wrong credential surface.

## Considered Approaches

### Recommended: Route Parity Fix

Wire `/settings/provider-keys` into every settings shell that can link to it. Reuse the existing `ProviderKeysSettings` component and add route coverage.

This is the lowest-risk fix because it preserves the current settings navigation and backend API behavior.

### Redirect To tldw Settings

Redirect `/settings/provider-keys` to `/settings/tldw`.

This prevents a 404 but conflates server authentication with LLM provider credentials, so it does not match the requested product behavior.

### Fold Provider Keys Into tldw Settings

Move provider key management into the tldw server settings page.

This may improve discoverability, but it expands scope and makes an already dense connection/auth page less clear. It is not needed to solve the broken route.

## Design

Add `/settings/provider-keys` as a first-class route in the hosted WebUI and extension/options route registry.

The page should render the existing `ProviderKeysSettings` component inside the existing settings shell:

- Hosted WebUI: create `apps/tldw-frontend/pages/settings/provider-keys.tsx` following the same dynamic import pattern used by nearby settings pages.
- Extension/options: add the lazy settings route for `ProviderKeysSettings` and a `/settings/provider-keys` route definition in `apps/tldw-frontend/extension/routes/route-registry.tsx`.
- Shared package shell: keep `apps/packages/ui/src/routes/option-settings-route-registry.tsx` as the canonical settings-registry entry. The shared options shell resolves settings deep links by importing this smaller registry through `DeferredOptionsRoute`.

The main shared `apps/packages/ui/src/routes/route-registry.tsx` is not the primary settings deep-link registry in the shared package shell. Do not add `/settings/provider-keys` there unless implementation testing proves a separate consumer needs it.

No new provider-key storage, API contract, encryption behavior, or settings layout is part of this change.

## Data Flow

1. User clicks a provider-key settings link or nav item.
2. Router resolves `/settings/provider-keys`.
3. Settings shell renders `ProviderKeysSettings`.
4. `ProviderKeysSettings` calls the existing `tldwClient.listUserProviderKeys`, `upsertUserProviderKey`, and `deleteUserProviderKey` methods.
5. The backend remains responsible for BYOK availability, key validation, encryption, and fallback to server defaults.

## Error Handling

Reuse current page behavior:

- BYOK unavailable (`403`) shows the existing "Provider key management is not available" information state.
- Fetch failures show the existing provider key load error.
- Save/delete failures show the existing Ant Design message feedback.

The route fix should not convert API failures into navigation errors.

## Testing

Add route coverage for:

- Hosted WebUI has a `/settings/provider-keys` page that imports `ProviderKeysSettings`; follow the page-shim test style in `apps/tldw-frontend/__tests__/pages/settings-mcp-hub-route.test.tsx`.
- Extension route registry defines `/settings/provider-keys`; follow the extension route-registry contract test style in `apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts` or add a focused provider-keys route test.
- Existing shared settings registry remains covered for `/settings/provider-keys`; a targeted assertion in `apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx` can verify settings deep links are resolved through `option-settings-route-registry`.

Manual/browser verification after implementation should open `/settings/provider-keys` in the WebUI and extension/options shell and confirm it renders the provider key manager instead of a 404.

## Non-Goals

- No redesign of settings navigation.
- No changes to provider key CRUD behavior.
- No backend BYOK API changes.
- No merge of provider keys into `/settings/tldw`.
- No migration of existing credentials.

## Success Criteria

- `/settings/provider-keys` resolves in the hosted WebUI.
- `/settings/provider-keys` resolves in the extension/options route registry.
- Existing "Add/configure provider key" links reach the provider key manager.
- Tests fail if the provider-key settings route is removed again.
