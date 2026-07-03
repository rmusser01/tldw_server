---
id: TASK-12103
title: Remove half-wired dead web auth stack and dead extension-routes tree
status: Done
labels:
- tech-debt
- medium
- frontend
- maintenance
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Medium (maintenance trap — realistic-looking dead code).** From the 2026-07-02 frontend audit (§4, §6b). Both were verified to have zero live mounts/importers.

1. **Half-wired web auth stack.** The web `AuthProvider` (`apps/tldw-frontend/hooks/useAuth.tsx:16`) and web `ConfigProvider` (`hooks/useConfig.tsx:237`) are **never mounted** — the `ConfigProvider` in `components/AppProviders.tsx:80` is antd's, not this one; these two appear only in their own definitions and in `__tests__`. Their only consumers, `components/layout/Header.tsx` → `components/layout/Layout.tsx`, are imported by nothing (`WebLayout` uses a different shared Header). So `useAuth`/`useConfig` would throw if rendered, and `api.defaults.baseURL` is never synced from user config. It's ~500 lines of code a maintainer will mistake for live.

2. **Runtime-unused `extension/routes` tree.** `apps/tldw-frontend/extension/routes/*` (`route-registry.tsx`, `app-route.tsx`, all `option-*.tsx`) is not rendered in the web build — pages mount `packages/ui/src/routes/*` via the `@/` alias (resolved in `next.config.mjs`). Editing a component here silently no-ops. **It is NOT safely deletable, though:** ~22 tests reference it (a few direct imports + `readFileSync` parity-guard tests that keep it byte-in-sync with `packages/ui/src/routes/*`). It's a deliberately-maintained mirror, not disposable dead code. Note: the `extension/shims/*` in the same directory ARE live — keep those.

Goal: quarantine with a clear marker (done — `_RUNTIME_UNUSED.md`) so maintainers don't edit/trust it. Genuine removal is a follow-up that must first migrate/retire the ~22 parity tests to target `packages/ui/src/routes/*`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unmounted web `AuthProvider`/`ConfigProvider` and their only consumers (`layout/Header.tsx`, `layout/Layout.tsx`) are removed, or quarantined with a clear "not wired / do not use" marker and a note on intent.
- [ ] #2 The dead `tldw-frontend/extension/routes/*` tree is removed (or documented as intentionally divergent with a guard); `extension/shims/*` is left intact.
- [ ] #3 No live import path breaks (typecheck + e2e smoke pass after removal).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
