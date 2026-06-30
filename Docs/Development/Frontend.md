# Frontend Development

This page covers local WebUI development workflows that need more detail than the
repo README. Keep feature-specific runbooks here when they affect repeatable
frontend testing.

## Research Workspace UAT Harness

Use this profile when validating `/research-workspace` against a running local
API with fresh browser state for beginner and power-user personas.

### Assumptions

- The API is already running at `http://127.0.0.1:8000`.
- The API is in single-user mode when testing the power-user persona.
- Do not paste or print real API keys in test logs. The E2E auth helper resolves
  the key from local config and seeds browser storage without attaching the value.
- Run frontend dependency installation from `apps/`, not from
  `apps/tldw-frontend/`.

### Install

The sandboxed macOS temp directory can reject Bun writes. Use explicit temp and
cache paths when installing from a clean worktree:

```bash
cd apps
TMPDIR=/private/tmp BUN_INSTALL_CACHE_DIR=/private/tmp/bun-install-cache bun install --frozen-lockfile
```

The Research Workspace page imports the OpenUI packages through
`apps/tldw-frontend/package.json`; if the page reports missing
`@openuidev/react-*` modules, rerun the root `apps/` install before debugging
application code.

This repository currently has some tracked `apps/packages/ui/node_modules`
symlinks. Bun can rewrite those symlink hashes during install; treat that as
dependency-install churn and do not include it in a Research Workspace UAT
change unless the task is intentionally updating vendored frontend
dependencies.

### Local Dev Server

Use webpack for Research Workspace UAT unless a task is explicitly validating
Turbopack. The webpack profile is slower to start, but it avoids hiding UAT
results behind Turbopack cache issues.

```bash
cd apps/tldw-frontend
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_VERSION=v1 \
bun run dev:webpack -- -p 8080
```

If file watching fails with `EMFILE: too many open files, watch`, raise the
shell file-descriptor limit before starting the dev server:

```bash
ulimit -n 8192
```

If the warning persists after raising the limit, keep the failure attached to
the UAT task and rerun with a prestarted WebUI plus
`TLDW_WEB_AUTOSTART=false`. That separates page behavior from dev-server watch
capacity.

### Focused UAT Evidence Check

The real-backend Research Workspace suite includes an entry-evidence test that:

- creates one fresh no-key browser context for the beginner persona;
- creates one separate API-key browser context for the power-user persona;
- captures screenshots as Playwright attachments;
- captures console messages, page errors, failed requests, and HTTP 4xx/5xx
  responses;
- records cold route timing for both personas and a warm route timing sample for
  the authenticated power-user persona.

Run the focused check with the default web-server autostart:

```bash
cd apps/tldw-frontend
TLDW_SERVER_URL=http://127.0.0.1:8000 \
TLDW_WEB_URL=http://localhost:8080 \
TLDW_WEB_CMD='ulimit -n 8192 && NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 NEXT_PUBLIC_API_VERSION=v1 bun run dev:webpack -- -p 8080' \
bunx playwright test ./e2e/workflows/research-workspace.real-backend.spec.ts \
  --project=chromium \
  --grep "UAT entry evidence" \
  --reporter=line
```

If the WebUI is already running:

```bash
cd apps/tldw-frontend
TLDW_WEB_AUTOSTART=false \
TLDW_SERVER_URL=http://127.0.0.1:8000 \
TLDW_WEB_URL=http://localhost:8080 \
bunx playwright test ./e2e/workflows/research-workspace.real-backend.spec.ts \
  --project=chromium \
  --grep "UAT entry evidence" \
  --reporter=line
```

Route timing is evidence, not a release gate by itself. For follow-up UAT, split
a performance task if a local run records:

- cold entry over 30 seconds to first useful Research Workspace UI;
- warm entry over 10 seconds;
- repeated browser navigation timeouts after the page has already compiled.

### Browser Launch Caveat

Some sandboxed macOS environments block Playwright before any test body runs.
Observed launch failures include:

- bundled Chromium headless shell failing
  `bootstrap_check_in ... Permission denied (1100)`;
- headed Chrome for Testing failing during crashpad startup.

When this occurs, record the launch error in the Backlog task and run the same
persona walkthrough with the Codex in-app browser/CDP harness. Do not treat a
browser launch failure as product evidence for or against Research Workspace
behavior.
