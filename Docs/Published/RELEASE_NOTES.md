# Release Notes

Published release notes entry point.

## 0.1.43 - 2026-09-20 (draft)

This candidate includes all 616 commits merged into the frozen development
snapshot through PR #2970 after `v0.1.42`, preserving the release-only repairs.

- OSCE scenario practice, advanced quiz fixtures/metrics, structured prompt
  recipes and Writing Predict/Fill service prompts.
- Provider-aware scheduled-task authoring, Explainer navigation and visual-novel
  asset generation preflight/recovery.
- Chat drafts, tab restoration, regeneration, provider/model selection, source
  evidence and account-isolation repairs.
- PostgreSQL ownership, queries, migrations, timestamps and transaction/shutdown
  repairs across Notes, Study, Characters, World Books, media, prompts and auth.
- Fresh-install usability/recovery fixes and real macOS guest recovery drills.

Back up persistent data before upgrading. UAT261 remains open by requester
direction; the previous full four-configuration UAT matrix failed and subsequent
targeted repairs do not establish a new full-matrix pass. Wider certification
remains separately tracked. The candidate requires exact-head CI/review and a
requester-written Change summary before merge.

The new protected source record proposes a September 20, 2026 release date and
September 20, 2028 at 12:00 UTC Countdown start. Requester review is pending.
The prior 0.1.42 grant is unchanged. Server packages/images exclude protected
frontend material; no protected frontend binary publication is planned.

See the repository [complete change inventory](https://github.com/rmusser01/tldw_server/blob/main/Docs/Development/releases/0.1.43-change-inventory.md)
and [execution plan](https://github.com/rmusser01/tldw_server/blob/main/Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md)
for all merged PRs, commits, verification and outstanding release decisions.

## 0.1.42 - 2026-09-10

- See the repository `CHANGELOG.md` for the full `0.1.42` rollup through
  PR #2941 plus the trusted license-gate bootstrap on `main`.
- This candidate includes chat/service prompts, notes and personal-context
  sync, research and presentations, audio/persona workflows, MCP transports,
  durable webhooks, production deployment checks, and reliability fixes.
- Back up persistent data before upgrading; the accumulated train includes
  schema changes. See the repository deployment and migration runbooks.
- Candidate repairs cover speech preference persistence, exported media
  artifacts, extension error copy, required WebUI TypeScript checking, and
  the missing local Python package needed for API container startup.
- Worker images now include their local packages and configuration, with backend
  import checks in CI. Erasure honors SQLite foreign keys; ACP health, embedding
  requeue warnings and erasure logs exclude private exception details. Shared
  frontend dependency majors are aligned, with strict URL/API-key guard and
  timeout checks. Shared hook enforcement covers corrected moderation/workflow
  clocks. Local model and audio input validation rejects symlink aliases before
  resolution; manual CI comparisons honor the selected base commit.
- Delayed voice-message saves preserve newer turns; empty conversations retain
  zero token counts. Additional DSR failure diagnostics exclude private text.
- Notification counts recover safely after account changes, notification APIs
  remain immutable, and web-clipper extension storage has strict type coverage.
- DSR previews query selected categories and fail on unavailable embedding counts;
  RAG input focus retains its behavior under compiler ref validation.
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
