# Moderation Dev Integration

Associated task: TASK-15466 (formerly TASK-15417).

## Scope and Rebase

The requester authorized rebase, conflict resolution, verification and a PR
against dev on 2026-11-07. No merge or auto-merge was requested.

- Fetched dev: `dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c`.
- Pre-rebase reviewed tip: `124ec3bc7a4176800bac0b8b75f00776ae63f037`,
  preserved at `codex/moderation-pre-rebase-20261107`.
- Replayed all 27 existing commits plus the integration tracking checkpoint.
- Explicit conflicts: ADR index twice (proposal/acceptance); Chat endpoint and
  Chat service import blocks. Dev credential imports and the single existing
  HistorySelectionError import are preserved alongside Moderation imports.
- Range-diff inspected: other differences are contextual dev changes and
  expectations already repaired upstream. No moderation caps, deadlines or
  runtime contracts were intentionally changed by integration.
- Fixed the newly introduced endpoint import-order finding; remaining lint
  debt is compared against the same paths on dev, not declared clean.

Dev independently assigned TASK-15417 and ADR-086 to other work. Paginated
open-PR file inventory plus dev inspection found maximum task ID 15465 and
ADR 095. Only this branch's implementation task and Moderation ADR were moved
to TASK-15466 and ADR-096. Old commit/task notes, log paths and evidence retain
historical aliases. Accepted ADR rationale was not changed. ADR assessment:
existing ADR-096 and ADR-021 govern; integration requires no new decision.

## Independent Integration Review

Reviewer Plato inspected the range-diff from the reviewed pre-rebase tip to
`dc7550704404fb227f62ddae8aa260da95585e80` against the fetched dev. No actionable
newly introduced or newly exposed integration findings. Review specifically
covered credential import conflicts, streaming usage collection before
transformation and failure finalization without successful completion/saves,
Audio/realtime credential auto-merges, and independent-field batching.
Subsequent identity corrections and import sorting are semantically neutral.
This is a static integration review, not a new whole-implementation review or
unconditional merge/platform approval.

## Verification

Commands run from the isolated worktree; Python commands activate the main
checkout's `.venv`. Existing warnings remain visible.

- Native benign containment prerequisites: 2 passed, 13 deselected, 18 warnings,
  exit 0. Run before the broad suite's independently contained hostile probes.
  Log: `/tmp/moderation-integration-containment-preflight.log`.
- Moderation shared UI: 7 files, 37 tests passed, exit 0.
  Log: `/tmp/moderation-integration-ui-tests.log`.
- Frontend `bun run typecheck` with the existing 8 GiB heap: exit 0.
  Log: `/tmp/moderation-integration-ui-typecheck.log`.
- Adjacent Chat credential policy/fallback taxonomy and Audio TTS credential
  boundary regressions: 97 passed, 307 warnings, exit 0.
  Log: `/tmp/moderation-integration-credential-regression.log`.
- OpenAPI fingerprint and committed path-union check: both match, exit 0.
  Log: `/tmp/moderation-integration-openapi-check.log`.
- All 68 cumulatively changed Python paths compile with SyntaxWarning as error.
- Production Bandit: 0 findings, 0 scan errors, exit 0. Existing unused-nosec
  advisories remain visible. `/tmp/bandit_moderation_integration_prod.json`.
- Test Bandit, only B101 excluded: 32 previously reviewed findings, 0 scan errors,
  exit 1. Same nine B404/nineteen B603 fixed-vector subprocess heuristics,
  one B105/two B106 dummy credential fixtures, one B608 constant table-name
  migration fixture; no new suppression. No input-controlled shell/executable
  or SQL is introduced. `/tmp/bandit_moderation_integration_tests.json`.
- Backlog task-ID check and scoped backlog-py normalization: exit 0.
- Shard coverage: 871 shards, 5,392 files, 4 ignored, 44 baseline,
  0 newly uncovered; exit 0. No CI workflow changes needed.
- HTTP-client-patching and nonempty-legacy-complete guards: exit 0.
- Same-path Ruff comparison against dev: 89 baseline, initially 71 current,
  one new endpoint import-order diagnostic. After moving only the new imports,
  endpoint comparison shows zero new; the other 67 paths are unchanged from
  that comparison. Both edited probe test files are Ruff-clean. Remaining
  inherited findings are not silently suppressed or described as a clean scan.

Backend focused suite and final committed-head task-format results are pending.
An initial task-format run referenced the old committed task filename after
its working-tree rename; actual renamed task contents pass canonical checking.
The committed-head ratchet must be rerun after committing the rename.

## Blocked-Pipe Probe Correction

The first broad backend manifest run finished with 2,150 passed, two preexisting
skips and one failure (8,242 warnings, 417.69 seconds, exit 1). The blocked-write
probe used a test-only 100 ms execution scope for a 1 MB request. Under load,
serialization exhausted that scope before Popen. The runner correctly returned
deadline_exceeded without launching a child; the probe then failed its required
assertion that a real child had been exercised. Raw failure retained at
`/tmp/moderation-integration-backend-focused.log`.

Added an independent-controller regression wrapping encoding with a controlled
150 ms delay. Before the harness correction it failed with the same no-child
assertion (one failed, exit 1); `/tmp/moderation-integration-blocked-write-red.log`.
The harness now uses the existing default two-second execution scope for this
real blocked-pipe check. It still requires a real launch, deadline failure,
reaped child, closed descriptors, empty registry and a successful next request.
Independent tree containment and the separate tightened-deadline regression are
unchanged. Production code, caps and deadlines are untouched by this correction.

## Publication Gates

Preserve the worktree for PR feedback. Required remote CI must run on the PR;
local checks are not substituted for those statuses. A requester-owned Change
summary explaining what changed and why remains required before merge. The
AI-authored PR description/task summary does not satisfy that gate.

Windows runtime remains unverified. Earlier Linux execution evidence remains
revision-qualified in the prior release record, not relabelled as fresh here.
No whole-project pass is claimed; prior whole-tree collection was blocked by
missing httpx2 in the unrelated Workspaces suite. Neutral-phase rollout still
requires compatible quiesced readers/writers, backup and pre-traffic preparation.
