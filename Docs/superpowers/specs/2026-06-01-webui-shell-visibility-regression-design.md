# WebUI Shell Visibility Regression Design

**Date:** 2026-06-01
**Backlog:** TASK-490
**Status:** Approved for planning

## Problem

The WebUI and extension are missing the expected top bar and sidebar across normal pages, not only `/chat`. That points to a global shell visibility regression rather than a single route bug.

The likely fault line is the interaction between:

- `apps/tldw-frontend/pages/_app.tsx`, which wraps WebUI pages in the web root shell.
- `apps/tldw-frontend/components/layout/WebLayout.tsx`, which owns WebUI header/sidebar rendering and shell suppression.
- `apps/packages/ui/src/components/Layouts/Layout.tsx`, which is still used by shared route components and must behave as a nested pass-through when the WebUI root shell already exists.
- Extension route shells under `apps/tldw-frontend/extension/routes/`, which need separate options/sidepanel behavior and should not inherit WebUI-only auth gating.

The current behavior suggests the root shell is either being hidden by auth/readiness state after the route content appears, or nested shared layout wrappers are suppressing the root chrome more broadly than intended.

## Goal

Restore the global app shell contract so normal authenticated/configured pages show the top bar and sidebar again across WebUI and extension surfaces.

`/chat` remains a required regression route because it has extra chat rail behavior, but the fix should be shell-first and route-agnostic.

## Non-Goals

- Do not redesign the header, sidebar, chat rail, composer, or page content.
- Do not add route-by-route layout wrappers as the primary fix.
- Do not roll back recent chat rail work unless investigation proves it is the direct cause.
- Do not change auth setup, first-run setup, or backend readiness product behavior beyond preventing accidental shell suppression.
- Do not merge WebUI and extension shell implementations into one new abstraction in this repair slice.

## Shell Visibility Contract

Normal WebUI app routes should render the WebUI root shell once auth/config readiness has resolved successfully.

Routes that may intentionally bypass or hide shell chrome:

- `/login`
- setup or first-run routes that intentionally present a focused setup flow
- settings routes that intentionally use their settings-specific navigation behavior
- error or recovery screens that explicitly need a focused full-screen treatment

For all other normal app routes:

- the top bar should be present;
- the sidebar or chat rail affordance should be present according to the route and viewport;
- route content should render inside the shell, not beside a duplicate shell;
- nested shared `OptionLayout` instances should pass children through under the WebUI root shell;
- nested shared `OptionLayout` hide requests should only affect the current route and only when explicitly allowed by the root shell.

## Architecture

Use the WebUI root shell as the single owner of global WebUI chrome.

The repair should clarify three boundaries:

1. **Root WebUI gating:** `pages/_app.tsx` decides whether a route is public, settings, setup, or normal app content. Normal app content should not hide shell chrome after successful auth/config resolution.
2. **Web shell rendering:** `WebLayout.tsx` renders the actual top bar, sidebar/chat rail, modal hosts, and route content. Its `hideHeader` / `hideSidebar` inputs should be treated as explicit shell policy, not as incidental loading state once a normal route is active.
3. **Nested shared layout handoff:** shared `OptionLayout` should detect the existing shell and avoid mounting a second full shell. It may request route-scoped hide overrides, but those overrides must be cleared on navigation and must respect the root shell's allow flags.

Extension options and sidepanel shells should keep their existing route-shell ownership. The WebUI-specific `_app.tsx` auth/readiness path must not become a dependency for extension sidepanel header/sidebar behavior.

## Error And Readiness Handling

Auth/config/readiness states should choose one of two visible outcomes:

- a focused login/setup/recovery surface when the route cannot be shown yet;
- the normal app shell with route content when auth/config/readiness is satisfied.

Avoid the broken middle state where route content appears but global shell chrome stays hidden.

If the backend is unavailable after the app has enough configuration to render, keep the existing recovery modal/banner behavior inside the shell where possible. Only fatal recovery states should intentionally suppress chrome.

## Testing Strategy

Add focused regression coverage before implementation:

- a WebUI shell test that seeds auth/config and asserts shell chrome on representative normal routes such as `/chat`, `/media`, `/notes`, and `/research-workspace`;
- a `/chat` assertion that the header and chat sidebar or edge affordance are present according to desktop/mobile route state;
- a settings-route assertion that settings-specific behavior remains intentional and does not leak hide overrides to the next normal route;
- an extension options/sidepanel contract test that verifies extension shell ownership remains separate from WebUI `_app.tsx` gating;
- a navigation test that moves from a route allowed to hide chrome back to a normal route and verifies the shell reappears.

Browser verification should use Playwright or the Browser plugin against a local dev server after the code fix. It should check visible shell chrome on at least three representative routes rather than relying only on source guards.

## Acceptance Criteria

- Normal authenticated/configured WebUI pages show the top bar and sidebar/chat rail affordance.
- `/chat` shows the expected header and recoverable chat rail behavior.
- Shell suppression remains limited to intentional public/setup/settings/recovery routes.
- Nested shared `OptionLayout` wrappers do not create duplicate shells and do not globally hide the root shell.
- Extension options and sidepanel routes keep their own header/sidebar behavior.
- Targeted tests and browser verification are recorded in TASK-490.
