This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/pages/api-reference/create-next-app).

## Getting Started

### Prerequisites

Recommended package manager: Bun.

Install Bun (if needed):

```bash
# macOS/Linux
curl -fsSL https://bun.sh/install | bash

# Windows (PowerShell)
powershell -c "irm bun.sh/install.ps1 | iex"

# Verify
bun --version
```

You can also use npm/yarn/pnpm if you prefer Node package managers.

1) Install dependencies:

```bash
cd apps/tldw-frontend
bun install

# npm fallback:
# npm install
```

2) Configure environment variables (copy `.env.local.example` to `.env.local` and edit as needed):

```
cp .env.local.example .env.local
```

Key variables:

- `NEXT_PUBLIC_API_URL`: Backend URL (default: `http://127.0.0.1:8000`)
- `NEXT_PUBLIC_API_BASE_URL`: Optional. Absolute base URL for static assets and WebUI links. If set, this takes precedence over deriving the base from `NEXT_PUBLIC_API_URL`. Useful when the API is mounted under `/api/vN` behind a reverse proxy.
- `NEXT_PUBLIC_API_VERSION`: API version (default: `v1`)
- `NEXT_PUBLIC_X_API_KEY`: Optional. Single-user mode API key (sent as `X-API-KEY`).
- `NEXT_PUBLIC_API_BEARER`: Optional. Bearer token for chat module when server sets `API_BEARER`.
- `NEXT_PUBLIC_RUNS_CSV_SERVER_THRESHOLD`: Optional. Runs row-count threshold for preferring server-side CSV export (default: `2000`).

3) Run the development server (use port 8080 to match the server's built-in local browser defaults):

```bash
bun run dev -- -p 8080

# npm fallback:
# npm run dev -- -p 8080
```

Open [http://localhost:8080](http://localhost:8080) with your browser.

### Quickstart networking (default Docker WebUI path)

When you use the repository quickstart Docker + WebUI flow, the default browser path stays on same-origin browser API requests through the WebUI proxy. That is the default quickstart networking story and does not depend on browser CORS setup or a custom browser-visible API host.

### Advanced/custom-host networking

Set `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced` together with `NEXT_PUBLIC_API_URL` only when you intentionally need an advanced/custom-host networking path, such as LAN/mobile access, a reverse proxy, or a custom domain where the browser should call a non-default API host directly.

## Branch-Aware Build Profiles

Frontend artifact scripts are branch-aware by default:

- `main` builds production artifacts.
- Any other branch builds development artifacts.
- Use `bun run build:prod` or `bun run compile:prod` when you need a production artifact from a non-`main` branch.
- Use `bun run build:dev` or `bun run compile:dev` when you need a development artifact explicitly.

The production profile forces the quickstart WebUI path. The development profile forces the advanced/custom-host path, so local feature branches keep the browser-visible API configuration developers expect.

### repo2txt Route

The web app exposes the shared repo2txt options UI at:

- `http://localhost:8080/repo2txt`

This page dynamically renders the shared route from `apps/packages/ui/src/routes/option-repo2txt.tsx`.

### Presentation Studio Routes

The WebUI now exposes the shared Presentation Studio surfaces at:

- `http://localhost:8080/presentation-studio`
- `http://localhost:8080/presentation-studio/new`
- `http://localhost:8080/presentation-studio/<projectId>`

Presentation Studio uses the Slides backend and only appears when the server advertises
`hasPresentationStudio`. Video publishing is separately gated by `hasPresentationRender`.

The browser extension also exposes a quick-start handoff route at:

- `chrome-extension://<extension-id>/options.html#/presentation-studio/start`

That extension route creates a server-backed presentation first, then opens the matching
WebUI editor route so the full edit/export flow stays in the browser WebUI rather than in
the extension tab.

Unified streaming (dev)
 - To exercise the unified SSE/WS streaming in the backend, start the API with the dev overlay:
   `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.dev.yml up -d --build`
  and set `NEXT_PUBLIC_API_URL` to `http://127.0.0.1:8000`. If you serve assets from a different origin or path than the API, set `NEXT_PUBLIC_API_BASE_URL` to the origin hosting web assets (e.g., `https://your-domain.example`).

You can start editing the page by modifying `pages/index.tsx`. The page auto-updates as you edit the file.

[API routes](https://nextjs.org/docs/pages/building-your-application/routing/api-routes) can be accessed on [http://localhost:3000/api/hello](http://localhost:3000/api/hello). This endpoint can be edited in `pages/api/hello.ts`.

The `pages/api` directory is mapped to `/api/*`. Files in this directory are treated as [API routes](https://nextjs.org/docs/pages/building-your-application/routing/api-routes) instead of React pages.

This project uses [`next/font`](https://nextjs.org/docs/pages/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Smoke Test

Run a quick connectivity check against the API:

```bash
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_VERSION=v1 \
NEXT_PUBLIC_X_API_KEY=your_api_key \
bun run smoke

# npm fallback:
# npm run smoke
```

The script exercises providers, chat, RAG, audio voices, and connectors (optional). A 404 on connectors is expected if that module isn’t enabled on your server.

## Auth Modes & Profile

This WebUI supports both single-user API key mode and multi-user JWT mode:

- In **single_user** mode, set `NEXT_PUBLIC_X_API_KEY` (and optionally `NEXT_PUBLIC_API_BEARER` for chat) and navigate directly to the app. Most flows (media, search, chat, audio) should work without logging in.
- In **multi_user** mode, use the `/login` page to authenticate against `/api/v1/auth/login`. On success, the app will hydrate the current user from `/api/v1/users/me`.

You can inspect the current user and roles via the `/profile` page:

- Shows identity, usage counters, roles/permissions, and a debug panel with the raw `/users/me` JSON response to aid troubleshooting.

## Linting, Tests & Build

From `apps/tldw-frontend/` (scripts live in `apps/tldw-frontend/package.json`; use `bun run <script>` or `npm run <script>`):

- `lint` – run ESLint against the codebase.
- `test` – run Vitest unit tests (for example, auth error mapping).
- `test:integration` – start the backend (optional) and run frontend tests + smoke checks (runs `Helper_Scripts/run-frontend-integration.sh` for flags and env).
- `build` – branch-aware artifact build. On `main` it produces the production quickstart artifact; on other branches it produces the development artifact.
- `build:prod` / `build:dev` – explicit production or development artifact overrides.
- `compile` – branch-aware webpack validation build using the same profile contract.
- `compile:prod` / `compile:dev` – explicit webpack profile overrides.

## Integration Test Harness

Run the full frontend + backend integration flow (from `apps/tldw-frontend/`):

```bash
bun run test:integration

# npm fallback:
# npm run test:integration
```

Common options:

- `bun run test:integration -- --backend-docker` (start backend via Docker Compose)
- `bun run test:integration -- --skip-backend` (assume backend already running)
- `bun run test:integration -- --no-backend-tests` (skip `pytest -m integration`)

Useful env overrides:

- `TLDW_X_API_KEY=...` (single_user mode)
- `TLDW_AUTH_MODE=multi_user TLDW_API_BEARER=...` (multi_user mode)
- `TLDW_DOCKER_COMPOSE=...` (custom compose file)

These commands should succeed before shipping changes to the frontend.

## UX Release Gates

Frontend PRs should pass these Playwright smoke gates:

- `npm run e2e:smoke:stage5`
- `npm run e2e:smoke:interaction`
- `npm run e2e:smoke:audio`
- `npx playwright test e2e/smoke/all-pages.spec.ts --reporter=line`

Interaction gate coverage (`e2e:smoke:interaction`) includes:

- Stage 1 defect closures (`INT-1`, `INT-5`): chat template leak guard and home theme-toggle discoverability.
- Stage 2 positive regressions (`INT-2`, `INT-3`, `INT-4`, `INT-6`): deterministic search typing/no-results behavior, keyboard-only command palette execution, and settings navigation active-state checks.

Audio gate coverage (`e2e:smoke:audio`) includes:

- Route identity + runtime budget checks for `/tts`, `/stt`, `/speech`, and `/audio` aliasing.
- Template leak guardrails for audio surfaces (`{{...}}` never user-visible).
- Timeout-to-retry recovery checks for ElevenLabs metadata (`/tts`, `/speech`) and transcription-model loading (`/stt`).

Baseline pass criteria for UX gate acceptance:

- Zero uncaught page errors on gated routes.
- Zero unresolved `{{...}}` template placeholders on audited interaction surfaces.
- No unexpected console/request failures beyond the scoped allowlist.
- Stage 5 + Stage 6 + Stage 7 smoke suites pass with route/action-level assertion diagnostics.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn-pages-router) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/pages/building-your-application/deploying) for more details.
