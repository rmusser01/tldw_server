# PR3084 current UAT checkpoint — 2026-10-20

Task: TASK-13260.278.18.83.45. Human requests publication of the current work set, progress toward merging it, and a separate PR for remaining UAT work when ready. Active UAT checkout and runtime remain untouched. No subagents or main-thread messaging.

ADR check: checkpoint publication adds no durable architecture rule. Preserve existing governing ADR030/043/059. Review incoming C09 PostgreSQL whole-database recovery ADR and resolve numeric ADR collisions against current dev without rewriting accepted rationale.

## Stage1: Isolate and review the actual increment
**Goal:** Capture a stable, source-only checkpoint against published2d753c9.
**Success Criteria:** Preserve all published commits and independent fixes; exclude runtime/build/model/credential files; review every changed source hunk and test body; task, compile/lint/security and affected tests pass.
**Tests:** Fresh diff/ancestry and snapshot stability, touched Python/TypeScript/Node and official isolated PostgreSQL checks as applicable.
**Status:** Complete

## Stage2: Publish the reviewed checkpoint
**Goal:** Update existing PR3084 with a normal reviewed commit and exact-lease publication.
**Success Criteria:** Remote/PR head matches, original human Change summary unchanged, PR verification describes current scope and limits.
**Tests:** Normal hooks, actual/staged review, fresh source/head guards and exact readback.
**Status:** In Progress

## Stage3: Current-dev integration and merge readiness
**Goal:** Reconcile the checkpoint with live dev and qualify the final head.
**Success Criteria:** All conflicts resolved by source review with published behavior retained, seven required current-head contexts successful, reviews settled, human-written summary covers the final UAT change and rationale. Merge only after all gates pass.
**Tests:** Queue-mode read, final affected checks/security, current-head hosted checks and review census, human-summary readback.
**Status:** Not Started

## Stage4: Handoff the remaining work
**Goal:** Keep current checkpoint and later UAT work distinct.
**Success Criteria:** Record the exact checkpoint/merge boundary; later UAT work uses a separate PR when it is ready. No empty follow-up PR or claim that current C10/C11 acceptance is complete. Remove only this owned plan once every stage is complete.
**Tests:** Compare publication boundary and concise task/PR status.
**Status:** Not Started

## Initial facts
- Published head2d753c9585257f3607e6776231841ae237383ecb includes the C08 checkpoint;939candidate files already byte-match it.65paths differ.
- Active source base523ef323e9f56bc31559f6c5b8e506c27e599dcb has701tracked dirty files plus source additions; most are already published, not new changes. Build trees and runtime data are excluded.
- Live dev e1ca22f2986554f85a7f9473104bfd3e984b95ae differs from stale local origin/dev. Queue variable unset; PR is unarmed and conflicting.
- Last recorded selected UAT:30accepted/13open, C10active. Publication does not close those workflows.
- Existing human Change summary covers ten stateless main.py handlers only. Preserve it; human-owned expanded rationale remains a merge requirement after the concrete checkpoint is ready.

## Checkpoint qualification
- Captured source is stable against published2d753c9. Whitespace-only changes were excluded; published fixes retained.
- Python280unique cases passed across scoped runs, including corrected official per-test restricted PostgreSQL restore targets. Initial credential/DSN setup errors were not accepted; the passing274cases were not replayed.
- UI175unique and WebUI268unique cases passed across their full runs and focused completions. Three multi-step invalid-target cases use the existing neighboring30-second case budget; assertions retained. Ten real-socket notification relay cases passed.
- Configured changed-source semantic checks: UI21files and WebUI26files,0diagnostics with actual Vitest setup. This is not a whole-project typecheck.
- Scoped security: Bandit239baseline/238current,0new AST-aligned findings and0errors. Ruff no new diagnostics; ESLint28files854baseline/849current,0new and0ignored. Python27files compile; credential-shaped scan and diff whitespace clear.
- Supported canonical OpenAPI export/APItypes passed:2117paths3271schemas, fingerprint3a3a8755af5e1d165ced795c2efd8d94b64319d1d8ab99649b071d8ab8dbfd85. Original three public descriptions retained.
- Independent hosted review and exact-current-head CI remain open. No new UAT/native/model/runtime acceptance is claimed. The expanded human Change summary is still required before merge.
