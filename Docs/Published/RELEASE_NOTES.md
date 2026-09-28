# Release Notes

Published release notes entry point.

## 0.1.45 - 2026-09-27

Patch release rolling up three changes merged after 0.1.44. [Complete inventory](https://github.com/rmusser01/tldw_server/blob/main/Docs/Development/releases/0.1.45-change-inventory.md).

- **License gate** — A license-gate run cancelled before it reached a verdict, including one superseded by a newer run for the same PR, no longer posts a false policy failure; the status stays pending, which remains fail-closed (#3029, #3032).
- **SQLite startup under sqlglot 30.20** — The canonical SQLite users table is accepted under sqlglot 30.20.0, and one shared DDL constant is pinned by a positive test (#3030).

No schema migrations or upgrade steps beyond 0.1.44's. The protected frontend source is unchanged since 0.1.44.

## 0.1.44 - 2026-09-27

Includes the complete frozen development range since v0.1.43: 272 commits across 37 merged PRs. [Complete inventory](https://github.com/rmusser01/tldw_server/blob/main/Docs/Development/releases/0.1.44-change-inventory.md).



- **Chat Macros v1.1** — Guided and YAML authoring, import/export, cloning, validation and named output-profile editors; built-ins remain immutable (#2951).
- **Workspace Persona provenance** — Persist local startup selection, opt-out and origin together, redact inaccessible origins and prevent forged import provenance (#2963).
- **Chat history and fork foundations** — Owner-validated history selection and independent local-copy identities/assets in the WebUI and full-page extension (#2968). Native fork contracts, retained-context projection and operation storage are groundwork; no public native-fork flow is exposed yet (#3002).
- **VN generation recipes** — Snapshot generation settings at acceptance and replay failed-slot recipes on Retry; Regenerate uses current settings (#3015).



- **Authentication and tenant isolation** — Enforce cross-user ownership across endpoints, storage, workflow and worker access; PostgreSQL core-chat RLS, owner-isolation tests and auth/scope/RLS ratchets. Audio diagnostics and warm-up require admin access; provider-initializing TTS health/catalog requires authentication (#2985, #2986, #2991, #2993, #2995, #2998, #2999, #3005, #2968).
- **CI and licensing** — Separate event concurrency so required gates report, run licensing admission before dependent gates, reduce audit clone depth, add Kanban/ACP/MCP coverage and timezone guards, and ship the canonical MCP GPL text (#2981, #2987–#2990, #2997, #3004, #3007, #3010, #3013).
- **Release records** — Carry completed 0.1.43 post-publication review records and prepare consistent 0.1.44 metadata, source manifest and proposed legal dates. Update architecture/workflow inventories and record reproducible missing-agent macOS VM startup recovery (#3014, #3017).



- **Post-release review repairs** — Clear denied chat selection safely, improve sign-in recovery copy, normalize image detail, restore readiness compatibility, include production configuration in packages, and isolate test fixtures and diagnostics (#2978).
- **Core reliability** — Correct share-link signing/error handling, PostgreSQL Notes slides candidates, duplicate RAG metric counting, MCP tab/carriage-return preservation, UTC timestamp interpretation, MLX cache ownership and other independently reproduced core defects (#2980).
- **Cancellation, OCR and Sync** — Propagate cancellation, preserve OCR input until consumed, keep withheld Sync envelopes reachable, and expire abandoned blob uploads so quota is released (#2982, #3006).
- **MCP and integration** — Remove ineffective SQL denylist filtering while retaining parameterized-query boundaries; fix MCP test failures and media auth import boundaries; remove production branching on pytest environment state (#2996, #2997, #3012, #2983).

- **Audio resampling and MCP tests** — Buffered audio without librosa now actually resamples through the existing scipy/linear fallback; invalid rates are rejected and empty audio is retained (#3024). MCP assertions, workspace checks and optional-parser requirements are corrected (#3025). Chat NetworkError status/retry translation remains a separately tracked known defect (TASK-13381, #3026).

### Upgrade requirements and limits

- Back up persistent data and **drain all API, worker and direct database writers** before registered per-user schema migrations. Restart only compatible binaries. Mixed-version rolling upgrades and data-preserving rollback to old binaries are unsupported (#2963, #2968, #3002).
- PostgreSQL application credentials must use a **NOSUPERUSER, NOBYPASSRLS role that does not own content tables**; migrations still run as the owner. Existing deployments using a privileged application role will fail startup until corrected. Existing pgvector tables require owner-prefixed migration to be listed; previously issued ownerless Chatbook signed URLs stop verifying (#2985).
- Native fork storage/projection is preparatory; later fork recovery/synchronization remains separate. Broader UAT and certification are separately tracked; targeted regressions do not establish a fresh full-matrix pass. See [#2963](https://github.com/rmusser01/tldw_server/pull/2963), [#2985](https://github.com/rmusser01/tldw_server/pull/2985), and [#3002](https://github.com/rmusser01/tldw_server/pull/3002) for migration/compatibility details.

Released September 27, 2026; the protected frontend Countdown starts September 27, 2028 at 12:00 UTC. Older release grants stay unchanged. Server artifacts exclude protected frontend material; no protected frontend binary is published.

Release review additionally hardens PostgreSQL startup policy verification, required isolation gates, commit-bound merge checks and corrupt history-metadata errors, retains invalidated history leases during automatic restoration, and localizes macro settings controls.

## 0.1.43 - 2026-09-20

This release includes all 616 commits merged into the frozen development
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
- Prompt Studio optimization events recover correctly when optional WebSocket
  transport finishes loading after a startup import cycle.
- Release review fixes preserve image quality options during chat retry and
  regeneration, retire media requests/actions on account changes, isolate saved
  media collections/favorites, limit OSCE traffic, and keep local model paths out
  of anonymous setup progress while preserving administrator resume.

Media collections and favorites now belong to an account on a server. Existing
unscoped local entries are retained in browser storage but are not automatically
assigned to a signed-in account, because their original owner cannot be verified.

Back up persistent data before upgrading. UAT261 remains open by requester
direction; the previous full four-configuration UAT matrix failed and subsequent
targeted repairs do not establish a new full-matrix pass. Wider certification
remains separately tracked. The release passed all 73 reported CI checks and final protected-source verification;
the requester approved the release and its PR-specific human-summary waiver.

The approved protected source record uses a September 20, 2026 release date and
September 20, 2028 at 12:00 UTC Countdown start.
The prior 0.1.42 grant is unchanged. Server packages/images exclude protected
frontend material; no protected frontend binary is published.

See the repository [complete change inventory](https://github.com/rmusser01/tldw_server/blob/main/Docs/Development/releases/0.1.43-change-inventory.md)
and [execution plan](https://github.com/rmusser01/tldw_server/blob/main/Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md)
for the full change inventory, verification, and separately tracked readiness work.

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
