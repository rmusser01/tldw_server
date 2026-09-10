# Release Notes

Published release notes entry point.

## 0.1.42 - 2026-09-10

- See the repository `CHANGELOG.md` for the full `0.1.42` rollup through
  PR #2939 plus the trusted license-gate bootstrap on `main`.
- This candidate includes chat/service prompts, notes and personal-context
  sync, research and presentations, audio/persona workflows, MCP transports,
  durable webhooks, production deployment checks, and reliability fixes.
- Back up persistent data before upgrading; the accumulated train includes
  schema changes. See the repository deployment and migration runbooks.
- Candidate repairs cover speech preference persistence, exported media
  artifacts, extension error copy and the required WebUI TypeScript check.
- Verify installed artifact digests: repository rollups 0.1.39–0.1.41 do not
  establish publication. At preparation, GitHub/GHCR app latest was 0.1.38
  and public PyPI listed 0.1.32.
- The source release's protected frontend material remains source-available
  under PolyForm Perimeter 1.0.1. Its release-specific Countdown grant adds
  `AGPL-3.0-only` on September 10, 2028 at 12:00 UTC; see
  `LICENSES/releases/0.1.42/`. No protected frontend binary is published.

## 0.1.41 - 2026-07-16

- See the repository `CHANGELOG.md` for the full `0.1.41` rollup from the
  frozen `dev` snapshot through PR #2744 into `main`.
- This patch highlights Research Workspace and source-grounded learning
  expansion, Skills catalog/rendering parity, Quick Ingest and Chatbooks
  hardening, single-user auth recovery, Watchlists/Jobs operations, and checked
  local/custom provider egress.

## 0.1.40 - 2026-07-10

- See the repository `CHANGELOG.md` for the full `0.1.40` rollup from
  `dev` into `main`.
- This patch highlights Chatbooks full-account backup/restore, chat document
  processing choices, CodeQL/security hardening, media ingest worker startup
  fixes, and release CI stabilization.

For release process details, see `Docs/Development/Release_Process.md`.
Use [Release Process](https://github.com/rmusser01/tldw_server/blob/main/Docs/Development/Release_Process.md) as the authoritative operator path and [Release Checklist](https://github.com/rmusser01/tldw_server/blob/main/Docs/Release_Checklist.md) as the broad readiness checklist.
