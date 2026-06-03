# Developing the tldw Assistant Extension

This document is for contributors working on the browser extension codebase.

## Prerequisites
- **OS**: macOS 13+, Linux, or Windows 11 (WSL2 recommended for Linux tooling).
- **Node.js**: 20.x LTS (>= 20.10). Check with `node -v`. Install via nvm from https://nodejs.org and https://github.com/nvm-sh/nvm.
- **Bun**: >= 1.0 (used for dev/build scripts). Check with `bun -v`. Install from https://bun.sh.
- **npm**: >= 10 (bundled with recent Node). Check with `npm -v`. If you prefer pnpm or yarn, keep them up to date but follow the existing lockfile.
- **Git**: >= 2.40. Check with `git --version`. Install from https://git-scm.com/downloads.
- **Browsers**: Current Chrome, Firefox, and Edge for extension development and manual testing.
- **Optional**: Playwright browsers via `npx playwright install chromium` if you plan to run e2e tests.

## Install & Build
- Install dependencies (respects `package-lock.json`):
  - `npm install`
- Common scripts:
  - Dev (Chrome): `bun dev` or `npm run dev`
  - Dev (Firefox): `bun run dev:firefox`
  - Dev (Edge): `bun run dev:edge`
  - Type-check: `bun run compile`
  - Build all: `bun run build`
  - Force production artifacts: `bun run build:prod`
  - Force development artifacts: `bun run build:dev`
  - Browser-specific production build: `bun run build:chrome:prod`
  - Zip bundles: `bun run zip`
  - Force production zip bundle: `bun run zip:prod`
  - Docs: `bun run docs:dev`, `docs:build`, `docs:preview`

Default artifact profile follows the checked-out branch: `main` builds production artifacts, every other branch builds development artifacts. Exported unpacked installs live under `build/chrome-mv3` and `build/edge-mv3` for production Chrome/Edge builds, `build/firefox-mv2` for production Firefox builds, and the corresponding `-dev` suffixed directory for development builds. WXT’s internal `.output/<target>` roots stay stable for tooling and automated tests.

## Configuring tldw_server for Development

- The extension talks to a tldw_server instance; by default it points at:
  - `http://127.0.0.1:8000`
- You can change the server URL from:
  - Options → Settings → tldw server.

### API key via environment variables

For easier local dev and CI, you can provide the tldw_server API key via Vite/WXT env:

- Supported env vars:
  - `VITE_TLDW_API_KEY`
  - `VITE_TLDW_DEFAULT_API_KEY` (fallback name)
- How it works:
  - On startup, the extension reads these env vars and, if `tldwConfig.apiKey` is missing, seeds the API key from the env value.
  - The key is stored in extension local storage (`tldwConfig`) so both the UI and background proxies use it.
  - A user-entered key in Settings always wins over env; env will not overwrite an existing saved key.
- Example `.env.local`:
  ```bash
  VITE_TLDW_API_KEY=your-local-dev-api-key
  ```

### Security & placeholder keys

- Keys containing `REPLACE-ME` are treated as **placeholder/demo** values.
  - In development builds they are allowed but emit console warnings and are intended only for local demos.
  - In production builds they are rejected: API calls will fail with clear errors instructing the user to set a real key under Settings → tldw server.
- Never commit real API keys to the repository or `.env` files that might be checked in.

## Running Tests

- e2e tests (Playwright):
  - List tests: `npm run test:e2e -- --list`
  - Run a specific spec: `npm run test:e2e -- tests/e2e/serverConnectionCard.spec.ts`
- Type-checks:
  - `bun run compile`

Note: some e2e specs build the extension before running tests. Make sure your local Node/Bun/WXT versions satisfy the prerequisites above.

For release-like local verification on a feature branch, prefer the explicit production entrypoints instead of the branch-aware defaults:

- `bun run build:chrome:prod`
- `bun run zip:prod`

## Offline Mode & Quick Ingest (for contributors)

Offline mode is primarily a UX/testing affordance. It lets the UI behave “as if connected” while the backend is unavailable, and turns Quick Ingest into a staging area instead of failing outright. This section explains how it works so changes stay consistent.

### Connection store & offline bypass

- Source: `src/store/connection.tsx`, `src/hooks/useConnectionState.ts`.
- The central Zustand store tracks:
  - `state.offlineBypass`: whether we’re in “offline but allowed” mode.
  - `state.isConnected`: true only when a real health check succeeds.
- Offline bypass flags:
  - Env flag: `VITE_TLDW_E2E_ALLOW_OFFLINE=true` forces bypass for CI/e2e.
  - Runtime flag: `__tldw_allow_offline` in `chrome.storage.local` / `localStorage`.
- Store helpers:
  - `enableOfflineBypass()` / `disableOfflineBypass()` (exposed via `useConnectionActions()`).
  - These call a shared `setOfflineBypassFlag()` and then `checkOnce()` to recompute connection state.
- Test/debug helpers (browser globals):
  - `window.__tldw_useConnectionStore` – direct access to the store for Playwright and ad‑hoc debugging.
  - `window.__tldw_enableOfflineBypass()` / `window.__tldw_disableOfflineBypass()` – thin wrappers over the store actions.

When `checkOnce()` runs with bypass enabled, the connection state goes to `CONNECTED` with `offlineBypass: true` and a placeholder server URL if needed. This is intentionally separated from the real “healthy server” path.

### ServerConnectionCard behaviour

- Source: `src/components/Common/ServerConnectionCard.tsx`.
- Connection phases map to variants:
  - `UNCONFIGURED → missing`, `SEARCHING → loading`, `CONNECTED → ok`, `ERROR → error`.
- Primary action per state (Options home):
  - Missing/unconfigured: **Set up server** (primary button) → opens `options.html#/settings/tldw`.
  - Error (unreachable): **Health & diagnostics** (primary button) → opens `options.html#/settings/health`.
  - Auth error: **Fix API key** (primary button) → opens `options.html#/settings/tldw`.
  - Connected: **Start chatting** (primary button) → focuses the main composer / sidepanel input.
  - First‑run completed vs not:
    - Before `hasCompletedFirstRun`: primary label is **Finish setup** and routes back into onboarding.
    - After first‑run: primary labels follow the mapping above.
- Offline mode UX:
  - When `offlineBypass` is true:
    - A gold tag is shown: “Offline mode — staging only”.
    - The advanced panel toggle is always available (even if the card appears connected).
    - The first advanced button becomes **Disable offline mode** and calls `disableOfflineBypass()`.
  - When disabled, the tag disappears and the button returns to **Continue offline**.
- Tests:
  - `tests/e2e/serverConnectionCard.spec.ts` exercises:
    - Error → offline toggle.
    - Offline badge visibility.
    - Reverting back to the error state after disabling offline mode.
  - `tests/e2e/options-first-run.spec.ts` expects:
    - Disconnected: a primary **Set up server** button on the card that routes to `/settings/tldw`.
    - Connected: **Start chatting** visible as the primary CTA, focusing the composer when clicked.

### Options / Sidepanel connection UX

If you change card copy or button labels, update the i18n keys in `src/assets/locale/*/option.json` and adjust the Playwright expectations accordingly.

- Options home:
  - Before first‑run is complete, `OptionIndex` renders only the onboarding wizard (no header/card).
  - After first‑run, the main Options view shows the chat Playground with:
    - A single primary CTA on the connection card per the mapping above.
    - Any inline “Set up server” links (e.g., composer banners) styled as secondary text links, not primary buttons.
- Sidepanel:
  - When not connected, the composer remains visible but:
    - Is read‑only with a connection‑specific placeholder.
    - Is wrapped in a muted/dashed container to signal “Not ready”.
    - Has a shared connection banner above it (see `EmptySidePanel`) with a single primary action **Open tldw Settings**.
  - When the connection transitions to connected:
    - `useFocusComposerOnConnect` shifts focus into the composer.
    - No extra, large “Start chatting” button is shown; the composer itself is the next step.
  - Icon buttons in Options and Sidepanel headers should either:
    - Include short visible labels at ≥`sm` breakpoints (e.g., “Quick ingest”, “Model”, “Modes”), or
    - Have descriptive `aria-label` + tooltip text if space is constrained.

### Quick Ingest staging semantics

- Source: `src/components/Common/QuickIngestModal.tsx`.
- Connection wiring:
  - The modal reads `isConnected` and `offlineBypass` via `useConnectionState()`.
  - `ingestBlocked = !isConnected || Boolean(offlineBypass)`.
- Queue vs processing:
  - While offline:
    - An amber banner explains: “Server offline — staging only”.
    - Rows/files show a **Pending — will run when connected** tag.
    - The main action button label changes to **Queue only — server offline**.
    - Clicking the button does **not** hit the network; it shows a toast and leaves items staged.
  - When the server comes back online:
    - The modal computes a `stagedCount` of items with no completed result.
    - If we previously had `ingestBlocked === true` and now `false` with `stagedCount > 0`, a toast fires:
      - “Server back online — ready to process {{count}} queued items.”
    - A secondary **Process queued items** button appears, which runs the normal ingest pipeline for whatever is currently queued.
- Footer hints:
  - While offline, a small footer string reinforces that items are staged and will run after reconnection.

When modifying Quick Ingest:
- Keep the “offline = staging only” contract: never silently send batches while `ingestBlocked` is true.
- Preserve the distinction between:
  - Queue-only label/behaviour in offline mode.
  - Full ingest/process labels when the server is reachable.

Key tests that should remain green after changes:
- `tests/e2e/quick-ingest-ui.spec.ts` – offline banner, pending tags, offline button label, Process‑queued flow.
- `tests/e2e/quick-ingest-workflows.spec.ts` – general UX and Inspector flows.

## Contribution Notes

- Follow the coding guidelines in `AGENTS.md` (TypeScript, 2-space indent, TailwindCSS for styles).
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
- When introducing new user-visible configuration that touches auth, document it clearly (Settings copy + docs) and avoid hard-coding secrets.
