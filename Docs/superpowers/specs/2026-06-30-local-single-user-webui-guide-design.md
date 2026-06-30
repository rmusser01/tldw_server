# Local Single-User WebUI Guide Design

Date: 2026-06-30
Status: Implemented
Backlog: TASK-12075

## Summary

Make `Docs/Getting_Started/Profile_Local_Single_User.md` a self-contained
local full-stack setup guide. A reader should be able to install dependencies,
configure single-user auth, start the FastAPI server, start the Next.js WebUI,
verify both services, and understand common local troubleshooting without
needing to jump to the README add-on section.

The README and frontend README remain useful deeper references, but the local
profile should carry the complete happy path.

## Goals

- Document the local API and WebUI setup in one canonical guide.
- Keep the current API install, start, verify, and first-value flow intact.
- Add WebUI prerequisites, environment configuration, install/start commands,
  URLs, verification checks, and troubleshooting.
- Use `http://127.0.0.1:8000` as the API URL in this guide because the local
  profile starts Uvicorn on `127.0.0.1`.
- Use WebUI port `8080` to match existing quickstart and local browser defaults.
- Include Bun as the recommended frontend package manager and npm as a fallback.
- Keep advanced LAN, custom-host, reverse-proxy, and deeper frontend development
  details as links rather than expanding the local profile into a deployment
  guide.

## Non-Goals

- No changes to server behavior, frontend behavior, Makefile targets, Docker
  compose files, or environment templates.
- No attempt to redesign the whole getting-started index.
- No dedicated local WebUI Makefile target in this slice.
- No live dependency installation or local service startup as part of the docs
  update.

## Current Repo Fit

`Docs/Getting_Started/Profile_Local_Single_User.md` covers the local API
profile and, on the current base branch, already includes a concise WebUI start
block. The guide still needs the complete local WebUI environment, auth warning,
npm fallback, verification, troubleshooting, and deeper-reference cleanup. The
README already contains the core local WebUI commands:

- copy `apps/tldw-frontend/.env.local.example` to `.env.local`;
- configure `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_VERSION`;
- optionally configure `NEXT_PUBLIC_X_API_KEY`;
- run `bun install`;
- run `bun run dev -- -p 8080`.

`apps/tldw-frontend/README.md` also documents Bun installation, npm fallback,
advanced deployment mode, and auth-mode behavior. The local profile should
inline the essential commands and point to those files for deeper details.

## Approved Approach

Use a fully self-contained local full-stack guide.

The page keeps the existing API-first shape and extends the WebUI path in the
existing `## Start` section. Verification then checks both services. This keeps
the final document aligned with the current base branch and avoids two separate
WebUI startup sections.

## Document Structure

Update `Profile_Local_Single_User.md` as follows:

- **Prepare**: add Bun to prerequisites with Node/npm as an optional fallback.
- **Start**: keep API startup unchanged, state that the API runs at
  `http://127.0.0.1:8000`, and extend the existing WebUI startup block with:
  - `cd apps/tldw-frontend`;
  - `cp .env.local.example .env.local`;
  - required `.env.local` values for the local API:
    `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` and
    `NEXT_PUBLIC_API_VERSION=v1`;
  - optional `NEXT_PUBLIC_X_API_KEY` guidance for single-user mode;
  - `bun install`;
  - `bun run dev -- -p 8080`;
  - npm fallback commands;
  - browser URL `http://localhost:8080`.
- **Verify**: keep API verification unchanged and add a WebUI spot check:
  `curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"`.
- **First Value**: explain that users can start from the WebUI or rerun
  `make verify-local-single` for provider-independent ingest/search
  verification.
- **Troubleshoot**: add local WebUI issues for port `8080`, dependency install
  failures, API URL mismatch, and single-user auth failures.
- **Optional Add-ons**: remove the old "Add the WebUI" link as the primary path
  and replace it with links to deeper WebUI development and advanced networking
  docs.

## Verification

This is a documentation-only change. Verification should include:

- Markdown review of the edited local profile for clear command order and no
  broken internal references.
- Link/reference spot checks for:
  - `apps/tldw-frontend/README.md`;
  - `apps/DEVELOPMENT.md`;
  - README WebUI advanced networking section if referenced.
- Targeted docs hygiene command if an existing onboarding docs checker applies.
- Bandit is not applicable because no Python code changes are planned; record
  that skip in the Backlog task during finalization.

## Risks

- The frontend example `.env.local.example` currently uses `localhost` while the
  local API guide uses `127.0.0.1`. The guide should explicitly say the value
  must match the running API host and use `127.0.0.1` for consistency with
  `make start-local-single`.
- `NEXT_PUBLIC_X_API_KEY` is browser-visible by design. The guide should keep
  it framed as local single-user convenience and avoid suggesting this setup for
  public internet exposure.
- Duplicating core WebUI commands creates a maintenance surface. Keep the
  self-contained path concise and link to the frontend README for details that
  are likely to change independently.

## Success Criteria

- A new local user can follow only `Profile_Local_Single_User.md` to run both
  the API and WebUI locally.
- The guide names both local service URLs:
  - API: `http://127.0.0.1:8000`;
  - WebUI: `http://localhost:8080`.
- The guide contains enough WebUI troubleshooting to diagnose the common local
  setup failures without searching the README.
- The README remains a reference, not a required continuation step for the
  local WebUI happy path.
