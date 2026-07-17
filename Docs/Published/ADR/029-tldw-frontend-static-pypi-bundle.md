# ADR-029: tldw-frontend static PyPI bundle

**Status:** Proposed
**Date:** 2026-07-06
**Backfilled from:** not backfilled
**Decision owner:** human/session
**Related task:** TASK-12158
**Related spec/plan:** `Docs/Product/WebUI/TLDW_Frontend_Static_PyPI_Bundle_PRD.md`

## Decision

Allow `tldw-server` PyPI releases to include a clean static export of `apps/tldw-frontend` under backend package static assets, while continuing to forbid frontend source, `.next`, `node_modules`, caches, model files, databases, Next standalone server artifacts, and `admin-ui`.

## Context

The current PyPI release boundary was hardened as backend/API/CLI-only to avoid accidentally publishing the frontend source tree, Node dependencies, generated caches, model files, and local databases. That boundary remains correct for unsafe or server-bound artifacts.

However, a static browser export is materially different from a Next.js standalone build. A static export can be served by FastAPI as ordinary HTML/CSS/JS assets and does not require Node.js at install or runtime. This gives PyPI users a usable first-run WebUI without making `pip install tldw-server` execute frontend build tooling.

The existing `apps/tldw-frontend` package currently uses `output: "standalone"` for non-PyPI distribution paths. The PyPI path needs a separate static export mode and release guard.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep PyPI backend/API-only forever | Leaves pip-installed users without the primary WebUI even though static assets can be safely served by the backend. |
| Ship the Next.js standalone output | Too large and unsafe for PyPI; the traced output can include repo-root models, databases, backend code, and caches. |
| Build frontend assets during `pip install` | Slow, fragile, network/toolchain-dependent, and inappropriate for Python package installation. |
| Create a separate `tldw-server-webui` package first | Reasonable later if size becomes a problem, but premature before measuring a guarded static export. |
| Bundle `admin-ui` in the same slice | `admin-ui` currently uses middleware and `app/api/*` routes, so it needs a separate design and migration path. |

## Consequences

- Release builds may include `tldw_Server_API/app/static/webui/**` when produced from a clean static export.
- Package-content validation must distinguish allowed static WebUI assets from forbidden frontend/build/runtime artifacts.
- Backend serving must mount `/ui` without weakening `/setup` gating.
- The PyPI package remains Python-only at install/runtime; Node/Bun are build-time release dependencies only.
- `admin-ui` remains outside this decision.

## Follow-up

- Implement the PRD in `Docs/Product/WebUI/TLDW_Frontend_Static_PyPI_Bundle_PRD.md`.
- Add package artifact checks for the new allowed/forbidden boundary.
- Add installed-wheel smoke coverage for `/ui`.
- Revisit a separate package only if the static export materially exceeds release-size expectations.
