# PRD: tldw-frontend Static PyPI Bundle

## Overview

Ship a clean static build of `apps/tldw-frontend` inside the `tldw-server` PyPI package so a user can install the backend package, run `tldw-server`, and open the WebUI from the same FastAPI process.

This PRD covers `tldw-frontend` only. `admin-ui` is explicitly out of scope because it currently depends on Next.js server behavior such as middleware and `app/api/*` routes.

## Goals

- Provide a bundled first-run WebUI at `/ui` for PyPI-installed `tldw-server`.
- Build only static browser assets; never package `.next`, `node_modules`, caches, backend databases, model files, or frontend source.
- Preserve existing non-PyPI frontend release paths.
- Add package-content guards so accidental large or unsafe artifacts fail release checks.
- Validate the installed wheel by serving the bundled UI and exercising at least one API-backed page.

## Non-Goals

- Shipping a Node.js runtime, Next.js standalone server, or SSR behavior in the Python package.
- Bundling `admin-ui`.
- Replacing Docker or release-artifact WebUI distribution.
- Solving every deep-link/dynamic-route edge case in the first implementation slice.
- Exposing runtime auth secrets through static assets.

## Target Users

- Local/self-hosted users who install with `pipx`, `pip`, or a virtualenv and want a usable UI without separately building Node assets.
- Release operators who need a deterministic wheel content boundary.
- Maintainers who need PyPI packaging to remain small and auditable.

## Product Requirements

1. `tldw-server` serves the bundled WebUI at `/ui` when static assets are present.
2. `/ui` works from an installed wheel without Node.js, Bun, `node_modules`, or the source checkout.
3. The PyPI wheel includes only the exported static WebUI artifact under `tldw_Server_API/app/static/webui/`.
4. The package build fails if forbidden paths or artifact classes appear in wheel/sdist contents.
5. The packaged UI uses same-origin API calls against `/api/v1`.
6. Existing setup behavior remains intact; `/setup` remains gated separately from `/ui`.
7. Static WebUI bundling is optional at build time for local developer builds but required for release builds that claim UI bundling.

## Technical Requirements

### Static export mode

- Add a package build mode for `apps/tldw-frontend` that uses Next.js `output: "export"`.
- Keep the existing `output: "standalone"` mode for non-PyPI distribution paths.
- Avoid copying `.next`; copy only the generated export output.
- Use a package-specific output directory before copying into `tldw_Server_API/app/static/webui/`.

### Next server-feature removal

The static package mode must not depend on features that require a Next.js server:

- Replace or delete `pages/api/hello.ts`.
- Move `pages/api/_tldw-webui/runtime-config.ts` behavior to a backend endpoint only if still required.
- Replace `pages/api/documentation/manifest.ts` and `pages/api/documentation/content.ts` with either static generated JSON/docs assets or FastAPI-backed endpoints.
- Audit dynamic routes under `apps/tldw-frontend/pages/**/[param]*` and choose the smallest static-safe strategy per route.

### Backend serving

- Mount the exported WebUI with FastAPI `StaticFiles(..., html=True)` under `/ui`.
- Return a clear 404 or operator-facing log if `/ui` assets are missing.
- Do not expose the setup HTML through generic static mounts.
- Optionally redirect `/` to `/ui` after setup is complete; keep setup-required redirect behavior first.

### Package guard

Extend the PyPI artifact checker to fail on:

- `.next/`
- `node_modules/`
- `apps/tldw-frontend/` source paths
- `admin-ui/`
- model artifacts
- database files
- frontend caches
- oversized WebUI bundles above an agreed threshold

## Proposed Implementation Slices

1. Static export build mode and copy target.
2. Server-feature migration for the small `pages/api` surface.
3. FastAPI `/ui` static mount and setup/root routing behavior.
4. Packaging guard updates and release workflow wiring.
5. Installed-wheel smoke test for `/ui`, static assets, and one API-backed workflow.

## Acceptance Criteria

- `make pypi-check` or an equivalent release gate builds a wheel containing `tldw_Server_API/app/static/webui/**`.
- The wheel and sdist contain no `.next`, `node_modules`, frontend source tree, model files, or database files.
- An isolated virtualenv can install the built wheel, run `tldw-server`, and fetch `/ui/`.
- A smoke test proves a bundled UI route can call the same-origin backend API.
- Documentation states that `admin-ui` remains a separate future packaging decision.

## Risks

- Static export may fail on dynamic routes until those pages are converted to static-safe shell routes.
- Runtime config previously handled by Next API routes may need a small FastAPI equivalent.
- Large browser chunks could push PyPI artifact size close to limits if asset discipline regresses.
- Root-route redirects can conflict with setup flow if route ordering is careless.

## Follow-Up Tasks

- Implement static export mode for `apps/tldw-frontend`.
- Move WebUI runtime-config/documentation API behavior out of Next server routes.
- Mount `/ui` in FastAPI with release-safe setup/root routing.
- Extend package artifact guards for static WebUI assets.
- Add isolated installed-wheel UI smoke coverage.
- Create a separate PRD/ADR for `admin-ui` if the frontend path proves successful.

## References

- Related task: `TASK-12158`
- ADR: `Docs/ADR/029-tldw-frontend-static-pypi-bundle.md`
- Next.js static export documentation: https://nextjs.org/docs/app/guides/static-exports
