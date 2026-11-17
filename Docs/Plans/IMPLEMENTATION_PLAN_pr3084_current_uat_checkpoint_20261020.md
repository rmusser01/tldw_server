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
**Status:** Complete

## Stage3: Current-dev integration and merge readiness
**Goal:** Reconcile the checkpoint with live dev and qualify the final head.
**Success Criteria:** All conflicts resolved by source review with published behavior retained, seven required current-head contexts successful, reviews settled, the original human summary remains exact and the requester’s PR-specific expanded-summary waiver is recorded. Merge only after all gates pass.
**Tests:** Queue-mode read, final affected checks/security, current-head hosted checks and review census, human-summary readback.
**Status:** In Progress

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
- Existing human Change summary covers ten stateless main.py handlers only. Preserve it; requester explicitly waived the expanded-summary requirement on2026-10-20.

## Checkpoint qualification
- Captured source is stable against published2d753c9. Whitespace-only changes were excluded; published fixes retained.
- Python280unique cases passed across scoped runs, including corrected official per-test restricted PostgreSQL restore targets. Initial credential/DSN setup errors were not accepted; the passing274cases were not replayed.
- UI175unique and WebUI268unique cases passed across their full runs and focused completions. Three multi-step invalid-target cases use the existing neighboring30-second case budget; assertions retained. Ten real-socket notification relay cases passed.
- Configured changed-source semantic checks: UI21files and WebUI26files,0diagnostics with actual Vitest setup. This is not a whole-project typecheck.
- Scoped security: Bandit239baseline/238current,0new AST-aligned findings and0errors. Ruff no new diagnostics; ESLint28files854baseline/849current,0new and0ignored. Python27files compile; credential-shaped scan and diff whitespace clear.
- Supported canonical OpenAPI export/APItypes passed:2117paths3271schemas, fingerprint3a3a8755af5e1d165ced795c2efd8d94b64319d1d8ab99649b071d8ab8dbfd85. Original three public descriptions retained.
- Independent hosted review and exact-current-head CI remain open. No new UAT/native/model/runtime acceptance is claimed. The requester’s explicit expanded-summary waiver is recorded; all other final-head gates remain open.

## Current-dev integration progress
- Published checkpoint dc61e0c99a135a4f910092a3ed01c0cd5adeeea1; original human summary and fresh Cubic suffix preserved. User reply “waived” is the PR-specific expanded-summary exception.
- Local rebase completed on dev e1ca22f2986554f85a7f9473104bfd3e984b95ae, integration HEAD0589e7c10c. This integrated head is unpublished and unqualified; configured semantic/behavioral, PostgreSQL migration/API, canonical generation, security and final source/staged review remain open.
- Proposed ADRs renumbered077/079 to avoid current dev067/068; accepted rationale untouched. PostgreSQL dictionary migration moved to86, preserving native83/84/85.
- Scoped Chat units209passed; four specific refusal-code cases4passed after the exact error-contract correction. No UAT/runtime acceptance added.

## Integration qualification update — 2026-10-20
- Current normal Chat/history and full-page interrupted-reply checks:320 unique cases per UI/WebUI configuration, composed of318 passing full-run cases and two passing focused completions. No claim of one320-case green run. Final two production semantic roots have zero diagnostics.
- Ordinary failed native saves retain their unknown-write classification and require explicit recovery; experimental completion-classification changes were reverted. The separate first-send repair publishes acknowledged create-response identity under the current-owner guard.
- B02 final bounded cohort is complete. Boundary trace exposed unhandled TypeError flushDraft is not a function in beginPromptAssistReset: the isolated persistence mock omitted that current method. One missing flushDraft mock added; original draft/loading/recovery/no-extra-send/one-explicit-send assertions retained. UI2passed and WebUI2passed, each natural0/219filtered. Four diagnostic production files restored exactly. All grants consumed; no pending approval or product/live-UAT finding.
- Full official history API/PG backup and interruption scope:116 passed. Six touched Bandit modules:1071 baseline/1064 current findings, zero new aligned findings/errors; other315 Python paths unchanged after the whole321-path assessment. Narrow Ruff cleanup is clear. Final security/lint review remains required.
- Pending normal merge now includes exact dev00e61ed88cbbc944a43c5595c4a3f4d55f05c4f4 (40 incoming paths). Recoverable pre-merge WIP ref refs/codex/pr3084-integration-wip-20261020-6fb051ab preserves the exact65-path patch; all saved source bytes survived3way reapplication, with reviewed ADR-index union. Accepted ADR078 remains byte-exact and proposed whole-PG recovery ADR is079. All12production and10test patches/complete added bodies are covered; two long plan narratives and final combined review remain open.
- No integrated publication or merge readiness is claimed. Main UAT checkout/runtime unchanged.

## Latest merged-candidate checks
- Canonical supported check with normal file access passes checked-in fingerprint7c30dbce614144a1267e1a05a8ac9564bf1a9a96fbc4129121f26ae5d20d2a9b,2135paths3306schemas and generated path union. The sandbox-only missing MediaWiki routes were not accepted as source drift. Three original public descriptions retained.
-27 merged/touched Python paths: Bandit637baseline/637current, zero new AST-aligned findings/errors; Ruff42baseline/23current, zero new; parse/diff whitespace clear.
- Media-summary and Watchlists suites122passed4765warnings186.49pytest195.17outer using normal file access. Five other incoming quota suites21passed1276warnings14.12pytest19.66outer:143distinct incoming quota/retention checks across both runs. Three RG suites exited0, but pass/skip counts were lost behind OTEL output; real Redis behavior is not accepted from that result.
- Known Admin/template failures from the93-case cohort are cleared across scoped completions; no claim of one green93-case run. Eleven production frontend files introduce no new warnings against published checkpoint. Four Admin/template semantic roots have zero diagnostics.
- Seven changed Admin/template test roots have zero configured semantic diagnostics in UI and WebUI. Eleven ignored ByRole exact options removed from the Rbac fixture; literal accessible-name behavior/assertions unchanged. Twenty integration test files have no new diagnostic counts against parent/upstream/published baselines; inherited warnings remain.
- Retained original first-full-import ownership, uninstrumented whole macOS Prompt natural exit and installer release acceptance are not closed by these source checks. Main runtime/worktree remains untouched; merge readiness needs current-head hosted/review gates and an authoritative disposition of retained release gates.
- Final local integration review covers affected production/caller contracts and changed test/fixture bodies. No new substantiated issue in covered source; no independent agent used under this side conversation rule. Two long incoming foreign plan narratives have selected requirements/AC/source-consistency review, not exhaustive narrative review. Configured pre_commit on104stagedpaths passed natural0/2.736s; wizard-only formatter hooks reported nofiles. Fresh live dev00e/published dc61/OPEN/unarmed and original human-summary hash match. Normal commit/exact-lease publication is next; current-head hosted qualification remains open.

## Integrated publication
- Published14fa8c4a78da39af70df02c15fb2921356d7d200, normal exact-lease push from dc61. Branch/PR/GraphQL read back exact, OPEN/MERGEABLE/unarmed. No repush after initial API lag. Commit first-parent patch exactly matches reviewed index SHA256ea86109346a80f4210e59416c7d480f2969ace2b82f0692e3a571325f67265cf. Original human paragraph/current Cubic suffix preserved through exact body readback.
- First full14fa census59checks1page/30completed29queued; seven required contexts not yet emitted. Actual CI39543972882 is pull_request/attempt1/exact14fa. No new test-failure evidence; superseded license-audit cancellation only. Four reviews threads/comments complete, deliberate private classifier unresolved/intact. No old logs read or Actions/queue mutation.
- Stage3 stays In Progress pending exact-head hosted/review acceptance and retained release-gate disposition. Stage4 remains open; no empty follow-up PR. Post-publication task/plan notes are local only and are not a tracking-only push.

## Pytest follow-up and requested merge
- User retained pytest>=9.0.3,<9.1 after the candidate9.1.1 logging-race probe failed and the9.0.3 control passed. Dedicated TASK-13895 records upstream-fix adoption, the bounded reproducer, retained log capture/plugin behavior and coordinated project/e2e constraint updates. This Markdown-only follow-up changes no dependencies or source.
- User explicitly requested merging this PR. Current head14fa is conflict-free with dev00e and queue unset, but the trusted license job/required workflows are still queued. Wait for exact-head required results; no bypass or cancellation of unrelated work.
