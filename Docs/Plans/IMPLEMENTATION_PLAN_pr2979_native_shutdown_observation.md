# Native Prompt Studio shutdown observations — UAT568

Task: TASK-13260.278.18.83.57. Related UAT556/.45 and UAT546/.35; PR2979 remains blocked. This diagnostic branch captures evidence and does not accept or repair the native timeout.

## Stage 1: Freeze source and native contract
**Goal**: Run immutable published b38c9f725748c5addd5ed44f7526bea55ee7621a after copying observer files into RUNNER_TEMP.
**Success Criteria**: Original macOS3.12 setup actions/environment, Prompt Studio scope, pytest arguments/plugins, 300-second case timeout, warning configuration and 60-minute job maximum preserved. Both useful published runs terminal before diagnostic dispatch; prior1190 identities/95 skip reasons retained for comparison.
**Tests**: Frozen workflow/action/source hash and semantic contract comparison; current refs and terminal inventories.
**Status**: Complete

## Stage 2: Observe without changing lifecycle
**Goal**: Reuse the private runpy launcher/probe and stdlib parent in this isolated worktree. Stream monotonic phase timestamps, content-free thread/Python stack symbols and filtered native call graph. Observe collection/sessionfinish/unconfigure/SystemExit/early and late atexit.
**Success Criteria**: Delegate every original unconfigure call exactly once and preserve exceptions/natural exit. Only the parent-owned unreaped child can receive SIGUSR1/native sample at15/30/60/90 seconds after sessionfinish. No observer termination cap, GC/cleanup/exit bypass, warning filter or dependency/budget change. Observation failures are explicit and never replace pytest results.
**Tests**: Original probe lacks streamed phase evidence; synthetic pass/fail subprocess and cleanup/error/privacy/ownership checks.
**Status**: Complete

## Stage 3: Verify and review observer
**Goal**: Qualify only the observer using small synthetic pytest subprocesses; no further whole local Prompt Studio run.
**Success Criteria**: Actual pass0/fail1, cleanup delegation/error preservation, no synthetic secret sentinel in observer records, owned child stack capture and no sampling after natural exit. Compile/Ruff/Bandit/Actionlint dispositions and independent immutable source/evidence review before normal-hook diagnostic commit/publication.
**Tests**: Helper_Scripts/diagnostics/test_pr2979_native_shutdown.py; static checks; frozen source/CLI parity. PostgreSQL unaffected and not rerun.
**Status**: Complete

## Stage 4: Capture native causal evidence
**Goal**: Publish only a separate diagnostic branch and dispatch its existing ci.yml workflow path against frozen b38. Preserve the nine held PR commits.
**Success Criteria**: Actual native source/dependency origins, XML identities/skip reasons, natural exit or unchanged automatic job maximum, phase/Python/native stacks retained. Any missing observations/causal attribution remain explicit. Final uninstrumented actual-head UX/strict/native/whole Prompt/Character unit/MCP wheel+sdist/Character Chat Rate-Limits and pending Chatbook disposition remain open.
**Tests**: Bounded native metadata/log/artifact verification and causal disposition before any corrective edit.
**Status**: In Progress

No new ADR. Original private observer evidence remains immutable. Full UAT paused; UAT261 open.

Local qualification: final immutable candidate4passes/0skips/failures/errors18.94s naturalexit0; real owned child native sample15 exit0/107 filtered frames then natural child0. Original cleanup return/RuntimeError/SystemExit11 preserved under unavailable/closed diagnostic I/O. Privacy sentinel absent. Compile/Ruff/Actionlint/diff0; Actionlint shellcheck integration disabled. Bandit25LOW notices (18test assertions,2subprocess imports,5fixed shell-free argv), no errors/exit1 retained, no security pass/suppression claim. Effective frozen b38 setup/env/pytest contract equal aside from launcher+one observer plugin; original source unchanged. Initial stream-red14 private default-cache/temp warnings and invalid pre-session sampling fixture failure retained/excluded. Independent immutable review pending; actual native evidence/cause and final gates OPEN.

Final v2 qualification/review supersedes historical v1 paragraph: sixpasses0skip/error/failure20.01s/natural0 after causal parent artifact red2fails1pass; ownedchild87878sample15exit0/94filteredsymbols/natural0/privacy guard retained. Compile/Ruff/Actionlint/diff0; Bandit26LOW19B101/2B404/5B603/zeroerrors/exit1 retained. Independent final review CLEAR SHAd2cd69c60345fd89cefa1dea9bfd6cf560c491625fec222da989a54e7a31f2a1. Original twoP2 review retained; parent artifact errors preserve natural exit and raw headers stage outside uploaded results. Stage4 native publication/dispatch/attribution pending; actual native and final gates remain OPEN.
