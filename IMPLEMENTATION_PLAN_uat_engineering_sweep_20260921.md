# Engineering sweep before complete UAT

> **Execution:** Continue in the main UAT task, using the existing debugging, test-first, review and verification workflows. The requester already approved this sequence and its revised review recommendations; no repeat plan approval is needed. Preserve the shared checkout before creating the integration checkout.

**Goal:** Find and repair defects across the complete A/B/C workflow catalog before another full four-configuration UAT.

**Architecture:** Reuse the release playbook, existing shared state/transport/persistence mechanisms, official PostgreSQL fixtures and runner/report helpers. Keep each repair as a separate reviewable Backlog unit. Run deterministic application checks separately from live provider quality and actual-image delivery, then qualify one frozen production candidate.

**Stack:** Existing TypeScript/Vitest/Playwright frontend tools; Python/pytest/Bandit backend tools; SQLite; official PostgreSQL fixtures; existing llama.cpp/mmproj service.

**Approved specification:** [Release UAT playbook](Docs/Development/RELEASE_UAT_PLAYBOOK.md), plus the migrated user-approved sweep sequence recorded in TASK13260.278. Task13262 created the playbook only; it did not implement fixtures or a complete runner.

## Current execution order — requester update 2026-09-22

The requester now requires a PR for this task's work missing from dev, followed by root-cause disposition, repairs and PR review resolution for all identified findings before any additional UAT. Native/full UAT stages below remain planned and are suspended until that repair/review work is handled. Focused causal unit/integration verification remains part of each repair. Do not launch the already-built UAT415 native package in the meantime. Create the PR as a draft while investigation or engineering gates remain incomplete. Keep existing acceptance gaps and provider-quality exceptions explicit; do not relabel them as verified.

## Latest checkpoint — 2026-09-22

PR2979 is published at `33e545ccd8`, including current dev `8045fa2` (zero missing commits at publication). The earlier exact frontend engineering run on `ddcb99d988` passes 46/46 without skips/retries, including canonical C03 two-sample scores and reload.

The next reviewed batch repairs stale Character ownership/random-selection fixtures (251 checks), request-time summary configuration fixtures (136), Slides winner clocks (113), historical migration fixtures (21, including four actual PostgreSQL checks), notification permissions (three actual PostgreSQL checks), registration rollback (one actual PostgreSQL check) and the comment-only HTTP guard false positive. Tracker at that checkpoint: 463 findings /441 verified /22 open. Full/native UAT remains paused while remaining AuthNZ/migration causes, CI and review gates are handled. The final requester-owned change summary remains pending.

The next transaction/lock batch adds real PostgreSQL magic-link lock release and atomic verification (33 combined auth/setup checks), canonical admin/budget seeds (22 checks in each bounded group), exact historical PostgreSQL migration coverage (13 checks), the remaining webhook schema oracle (110 checks), and acquired-only system-log lock cleanup (59 checks). UAT464 is locally verified; current tracker totals are464/442/22. Fresh dev fetch remains8045fa2 with zero missing commits. Hosted Jobs SQLite passes on33e545ccd8; PostgreSQL is queued and remains a separate required result. Publication is held during useful local repairs to avoid cancelling it again. Full/native UAT remains paused.

The follow-on budget/monitoring and PostgreSQL allowlist subgroups verify another51 adjacent and6 actual PostgreSQL checks, respectively. The latter also fixes two tests that swallowed assertions and crossed the PostgreSQL event loop; four forced forbidden responses now fail those assertions. Both stay within broad UAT451. BYOK PostgreSQL fixtures and remaining migration-routing doubles are tracked as unfinished children45/46, and an observed Windows SQL-migration encoding failure still needs its own bounded repair. The next PR commit contains completed reviewed units only.

Children45–47 now have causal red/green evidence: the BYOK PostgreSQL module moves from18 guarded-user-write failures to19/19 actual PostgreSQL passes; four stale v33/v65/v67 coordinator fakes now pass with backend/lock/connection observability; and three Windows-codec BOM migration failures become55 adjacent passes after explicit UTF-8 loading. UAT451 and UAT457 remain open umbrellas; UAT465 is locally verified, with hosted Windows acceptance pending under UAT419. Full/native UAT remains paused. The published PR still precedes these local repairs while its earlier PostgreSQL Jobs shard remains queued.

The next PostgreSQL AuthNZ checkpoint repairs seven more guarded-user fixture modules (eight actual PostgreSQL passes) and exposes UAT466: the auth service passed a pool rather than its active transaction connection into the profile-version gateway. Binding that connection through the write transaction passes two actual PostgreSQL service checks, one real SQLite pool check,26adjacent service checks and six caller-owned PostgreSQL magic-link checks. The UAT451 umbrella and hosted CI acceptance remain open; no full/native UAT has resumed. A fresh fetch and explicit rebase after local commit7f18655aad report zero missing dev commits.

The profile-version migration fixture subgroup moves from seven failures/one pass to eight actual PostgreSQL passes. Historical schema and indirect-write setup use the official isolated fixture database directly; guarded migration paths and all substantive assertions remain. The startup-corruption oracle now checks the existing `False` result and direct migration error. TASK13260.278.17.50 is locally verified; UAT451 remains open for other failures and hosted acceptance.

The adjacent candidate-schema fixture moves from one failure/two passes to three actual PostgreSQL passes. Its shadow-FK and missing-default DDL is confined to rolled-back transactions on a direct connection to the official isolated fixture DB; runtime validation and the production guard remain unchanged. TASK13260.278.17.51 is locally verified. Next focus is outstanding hosted AuthNZ, historical migration and Windows CI failures before resuming any full/native UAT.

The complete AuthNZ PostgreSQL directory now passes 56/56 with no skips and normal exit on the local branch. Hosted PostgreSQL CI on the published older revision remains queued, so it is not counted as passing. Windows core-utils triage identifies UAT467 (missing `fchmod` and an owner-only security constraint) and UAT468 (portable forbidden-path and bind-mount parsing). A cross-platform drive-source test fails before UAT468's repair; the whole preflight module then passes 241 checks locally. UAT467 remains open because automatic security review rejected a chmod fallback that would not enforce owner-only access on Windows. Hosted Windows acceptance remains required; full/native UAT remains paused.

The hosted import-boundary failure is UAT469: six media endpoints used the core AuthNZ import path. Routing the exact re-exported `User` and `get_request_user` through `API_Deps.auth_deps` moves the boundary module from one failure to three passes; all six modules compile/import and 15 adjacent endpoint checks pass (two unchanged optional `pypff` skips). Hosted CI remains pending. A fresh fetch and explicit rebase onto dev `8045fa2` still show zero missing dev commits.

## Global constraints

- Four cells: sqlite-single, sqlite-multi, pg-single, pg-multi. Record each domain's actual database engine and normal actor role.
- WebUI is primary; extension options/sidepanel and cross-surface handoffs remain in scope. Name browser/viewport and capability variants before execution.
- Keep generated Playwright output, private credentials and runtime profiles out of the PR. Never erase failed evidence or stop shared services to simulate failure.
- Evidence has an exact source/base/artifact hash. Older passing evidence never certifies a changed candidate.
- Track code review, durable regression and native UAT separately. Report planned/passed/failed/blocked/not-run/N/A identities; issue-closure percentages cannot establish UAT completion.
- Retain UAT261's explicitly accepted exception as an open failed live-quality outcome. No other failure is waived by changing sequence.
- Preserve the first attempt and any recovery attempt. No skipped, flaky, missing or silently returned required case counts as a clean pass.
- Clean-machine installation, fresh application state and supported upgrades are separate phases. Reused dependencies and caches must be declared.
- Approval review recovered on2026-09-21; the previously blocked native action was never bypassed. Current-dev fetch and isolated worktree creation succeeded. Keep past rejection evidence and use the approved isolated checkout for subsequent edits.

## Stage 1: Baseline, ownership and coverage audit

**Goal:** Preserve all current work and establish exactly what the next sweep must prove.
**Success Criteria:** Verified source/commit checkpoint; adjudicated ownership inventory; all43 playbook families visible; registered-test inventory and known gaps; verified current-dev integration in an isolated checkout before adopting repairs there.
**Tests:** Checkpoint copy hashes and bundle verification; unchanged HEAD/index; workflow-ID uniqueness and source links; Playwright collection output separated from execution; planned-case reconciliation before freezing an executable manifest.
**Status:** In Progress

Files: this plan, `Docs/Reviews/UAT_ENGINEERING_SWEEP_COVERAGE_2026_09_21.md`, `Docs/Reviews/UAT_ENGINEERING_SWEEP_INVENTORY_2026_09_21.json`, its exact `.gitignore` allowlist entry, running UAT tracker. Owner: TASK13260.278.1.

- [x] Preserve tracked edits, relevant untracked source/tracking files, commit delta and original evidence provenance in a private checkpoint without altering the shared index/branch.
- [x] Inspect playbook, existing package/config entry points and the three known weak journeys.
- [x] Complete the durable family/source/ownership audit and record test-collection failures. All43 families are inventoried;93 local links/source references validate; private evidence stays ignored and the durable JSON is explicitly allowlisted.
- [ ] Enumerate executable variants, dependencies, actors, modes and canonical oracles; reconcile them with registered tests. A family inventory is not a completed case manifest.
- [x] Fetch current dev when approval review is available; record the fetched SHA and create a `codex/` isolated checkout from it. Verify ancestry and port/source ownership. Never rebase the dirty shared checkout.
- [x] Adopt only attributable UAT repairs into the isolated checkout, using checkpoint files and task evidence. Review overlaps and rerun affected checks; retain unrelated Chatbook/TTS/roadmap work in its original ownership.

## Stage 2: Systematic workflow and shared-mechanism review

**Goal:** Review every live caller along UI → shared state → request → persistence → reload across the whole agreed catalog.
**Success Criteria:** Each required workflow/variant has a reviewed call path and linked regression/native obligations. Findings are deduplicated and assigned to repair tasks.
**Tests:** Exact owner/actor, object ID, content/version/count and reload oracles; controlled timing for stale responses and queue/cancellation races. Structural source checks remain structural evidence only.
**Status:** In Progress

- [ ] Start with privacy/data loss/startup: auth target/scope changes, drafts, account/tab restoration, queued dispatch and duplicated writes/history.
- [ ] Review ordinary Chat and Character Chat through all live WebUI/extension callers, including image send/Retry, temporary/saved conversations, metadata errors, cancellation and late saves.
- [ ] Trace A-05→A-07→A-09→A-11 and A-08→A-10→A-11 using the same returned source/Note/card/conversation identities.
- [ ] Review B/C workers, permissions, result artifacts and recovery, then X-01…X-04 and all shipped supplemental variants. Do not treat page load or outer job completion as the requested result.
- [ ] Track each concrete new defect before editing it. Existing UAT375/388 and other open tasks remain active with their earlier evidence.

## Stage 3: Causal repairs and durable regressions

**Goal:** Repair every confirmed issue with the smallest change at its real owner and durable positive/negative controls.
**Success Criteria:** Meaningful regressions fail before the repair and pass after it; affected SQLite and official PostgreSQL checks pass without hidden skips; independent review and touched-scope lint/type/Bandit are complete.
**Tests:** Versioned deterministic downstream responses for application semantics; separate live llama.cpp/mmproj and real PNG delivery checks. Fault only owned requests/proxies, never shared services.
**Status:** In Progress

- [ ] UAT389: prepare owned Content Review drafts; make absent required actions and failed AI/commit requests explicit; verify edited content, diff, commit ID/version and reload.
- [ ] UAT390: use F-SOURCE/F-DISTRACTOR, require the exact ingested result and supported citations in the saved/reloaded grounded answer.
- [ ] UAT391: preserve Note/source IDs through exactly five distinct generated/saved cards, five reviews and scheduling/analytics controls.
- [ ] UAT392: repair discovery boundaries and explicit collection setup; reuse `scripts/assert-playwright-no-skips.mjs` and `scripts/live-tier-uat/report.mjs` for accounting. Add exact identity reconciliation instead of a duplicate pass-count checker.
- [ ] For each application defect found in Stage2, create a bounded plan with its exact failing test, causal edit and verification commands. Commit complete reviewed units with their task records; never stage unrelated files or bypass hooks.

## Stage 4: Integrated frozen-candidate sweep

**Goal:** Run integrated regressions and targeted browser recovery against one owned production artifact set.
**Success Criteria:** Every declared case has an attributable outcome; no untriaged failures, missing required results or unresolved critical privacy/data-loss/startup defect. Candidate code does not change during the run.
**Tests:** Deterministic four-cell semantics, targeted native recovery, actual images and live capability actions; first-attempt and retry results retained separately.
**Status:** Not Started

- [ ] Finish upstream adoption before the Stage4/5 freeze. Build server/WebUI/extension from that adopted isolated revision; freeze manifest, fixture/provider contracts and artifact hashes.
- [ ] Start owned four-cell profiles using the official PostgreSQL fixture adapter and restricted application identities. Verify actual stores/roles; normal UI login supplies acceptance identity.
- [ ] Set retries0 and disable reuse of unrelated dev servers. Reuse existing owned-runtime and bounded reporting helpers, extending only the missing workflow/cell orchestration.
- [ ] Finish UAT375 metadata/queue/account-return and UAT388 Casual picker acceptance on both SQLite/PostgreSQL, plus all remaining original-failure controls.
- [ ] Reconcile planned versus observed immutable case IDs, including blocked dependencies, unrun modes and explicit exceptions. If a repair changes the candidate, start a new frozen run for the affected gate.

## Stage 5: Complete agreed UAT and release integration

**Goal:** Repeat the full agreed catalog only after Stage4 gates; complete authorized review/merge work against verified current dev.
**Success Criteria:** Complete required four-cell workflow variants, fresh/installation/upgrade phases and UX outcomes on the named artifacts; review comments and CI addressed; requester-owned change summary satisfies the repository merge policy for the final scope.
**Tests:** Full release-playbook case manifest, published installation procedure, previous/oldest supported upgrades, X-03 UX/browser review and live capability oracles.
**Status:** Not Started

- [ ] Identify supported installation and upgrade starting points from published product support; record exact versions and separate unrun/unsupported paths.
- [ ] Execute the full required A/B/C/X/S cases in all applicable cells/surfaces; continue independent cases while recording blocked dependencies.
- [ ] Preserve exact failures and accepted exceptions in the final report. Do not turn narrower repair evidence or the number of closed findings into release approval.
- [ ] Recheck dev and address Qodo/CI/review comments in isolation. If later upstream adoption or repairs change candidate source content or built artifact hashes, freeze a new candidate and repeat the required integrated and complete four-cell UAT gates before merge. Earlier results stay attached to their original candidate; a partial retest cannot certify the new one. Create/update the appropriate PR and attach it to this task.
- [ ] Apply the repository's human Change summary gate to the actual final PR scope, then perform the already-authorized merge only when all required gates are satisfied.

## Checkpoint and isolated integration

Private checkpoint: `.tmp/uat-engineering-sweep-20260921/checkpoint/`. HEAD `3c871df7173d7b87b64bbb421882c2088542cb7d`; cached origin/dev `08e980a453d12155cccf10d0c5eb7fe32d05ac75`; common ancestor `d72b1d2850ea947b6d12cac19f6b95867b68a580`. The bundle requires that ancestor and was verified locally. There are184 copied files,21 runtime/cache entries retained in place, and0 copy mismatches. No remote fetch was performed; cached origin/dev is not called latest. Candidate12 and all prior native evidence remain unchanged. UAT388 source is reviewed/166 regressions pass, but it has not been packaged/natively accepted. Native/remote integration requiring approval review remains blocked by the recorded Codex usage-limit failure.


2026-09-21 integration update (TASK13260.278.6): fetched dev08e980a453 verified; native-managed worktree `/Users/macbook-dev/.codex/worktrees/uat-engineering-sweep-20260921/tldw_server2`, branch `codex/uat-engineering-sweep-20260921`. Refreshed checkpoint copies194 files/0mismatches, retains21 runtime/cache entries.213 attributable paths transferred by three-way merge; three textual conflicts resolved and13 upstream overlaps independently reviewed.61 clean-baseline tests,1393 integrated frontend tests,22 theme tests and187 SQLite/official PostgreSQL regressions pass. Adoption verification is complete: UAT393 details/generated-placeholder and UAT394 diagnostics are repaired and independently reviewed; final245 backend and1393 frontend tests pass, zero skipped/pending. Both production builds pass. UAT393 native acceptance and exact full workflow case accounting remain open. No candidate freeze or new UAT pass yet. The old approval blocker paragraph above describes the original checkpoint only. The shared tracker was not overwritten after approval review rejected that copy; the isolated tracker is current.

## Restored CI execution checkpoint — 2026-09-22

PR2979 remains a draft on cec0e56; full/native UAT stays paused. Backend groups now execute and expose previously hidden cross-platform failures. UAT432–441 have bounded Backlog ownership and tracker entries. The current batch repairs missing-user boundary semantics, evaluation polling feedback, exact model selection, and stale permission/egress/schema fixtures. Canonical user seeding and portable admin persistence remain in progress. Keep restored execution failures under UAT419 until each has a causal disposition; publish reviewed increments without interpreting aggregate or skipped jobs as acceptance. Strict C-03 run/readback, actual Prompt model selection and hosted PostgreSQL/Windows results remain required.

2026-09-22 verified repair checkpoint: currentdev remains8045fa2956 (freshfetch/API/ancestryconfirmed,0behind46ahead). Local fixes cover UAT432–446 except explicit hosted acceptance for polling/Windows. Canonical userseeds retain193passing cases includingactualPG and1separate revoked-key error-boundary failure (UAT447). CI now schedulesJobsPostgreSQL afterSQLite; actualresult stillrequired. The runningtracker holds449findings with28open engineering/acceptance gates. Root and independent reviews are clear for the completed batch; publish as incremental draft work while447–449 and othernewshard causes remain underinvestigation. Full/nativeUAT doesnotresume.
