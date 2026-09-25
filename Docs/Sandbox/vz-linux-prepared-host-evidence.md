# VZ Linux Prepared-Host Evidence Tracker

**Status:** Active tracker for prepared Apple silicon `vz_linux` acceptance evidence.
**Scope:** Real `vz_linux` execution evidence from manual operator runs or the host-gated workflow on trusted refs.
**Policy:** `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`.
**Drill contract:** `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md`.
**Operator entrypoint:** `tools/macos-vz-helper/scripts/vz-helperctl.py smoke` or `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`.

## Purpose

This tracker makes prepared-host acceptance evidence reviewable without making
real VM execution part of normal CI. It records what a prepared Apple silicon
host proved, which expected skips were accepted, which artifacts were preserved,
and what residual gaps remain.

Normal PR checks should continue to use portable unit tests, workflow contract
tests, fake/scaffolded paths, and docs checks. Real `vz_linux` execution remains
manual or host-gated only:

- local operator run on a prepared Apple silicon macOS host
- `workflow_dispatch` on `main` or `dev`
- opted-in scheduled host-gated workflow when
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1`

Do not add pull request triggers, push triggers, scheduled destructive drills,
host reboot automation, launchd automation, or network expansion from this
tracker. Host reboot and launchd validation are explicit manual/operator-gated
drills; this tracker records whether those drills were run or intentionally
skipped for a prepared-host evidence packet.

## 2026-09-25: Missing-Agent Startup Drill (Accepted)

- Local operator run on Apple Silicon `Mac17,6` (arm64), macOS 26.5.2, branch
  `codex/vz-postmerge-cleanup` at `122519feb236a196d169a62123c67f565deb9954`.
  The working tree was clean when the live workflow began.
- Rebuilt Debian bookworm arm64 source with the repository rootfs, packer,
  kernel/initrd extractor, and bundle scripts inside the existing Debian 13.4
  arm64 Fusion VM. The VM was shut down after the build. Builder logs, read-only
  `e2fsck -fn` output (exit 0), source hashes, and the new bundle are under
  `~/Library/Logs/tldw/vz-bundle-rebuild/20260925-1138/`. The original
  incomplete bundle was not modified.
- Source bundle: `~/Library/Logs/tldw/vz-bundle-rebuild/20260925-1138/bundle`.
  SHA-256: rootfs `b7722641cc1e706fb3c544784c053cd67607f495c310db945605241594d7315a`,
  kernel `84b9c190bb4589c4a9527e3191fec051f9f115e88f0a3e8afae96ba0dfb4dfef`,
  initrd `326f0ef8851ac61e1c837cf833bb116cacd12b9e85f56d9e0ff255c4cb92882d`.
  Builder and host hashes matched; the workflow's before/after source hashes
  also matched for all five fingerprinted source files.
- The signed helper at
  `~/Library/Logs/tldw/vz-workspace-workflow-20260914-r1/helper-used` passed
  strict codesign verification and preflight entitlement checks; SHA-256
  `049ab407cf4e0cb9b858f1b877d9f1df5bdb2c90c048230c2ff16eee9f486638`.
  The manual `vz-failure-drill.py --allow-fault-injection` run used that helper,
  the rebuilt source, and a fresh private evidence directory.
- Invocation from the worktree root, after activating the repository virtual
  environment (with `PYTHONPATH="$PWD"`):

  ```bash
  python tools/macos-vz-helper/scripts/vz-failure-drill.py \
    --allow-fault-injection \
    --source-bundle "$HOME/Library/Logs/tldw/vz-bundle-rebuild/20260925-1138/bundle" \
    --helper "$HOME/Library/Logs/tldw/vz-workspace-workflow-20260914-r1/helper-used" \
    --evidence-dir "$HOME/Library/Logs/tldw/vz-failure-missing-agent-20260925-1140"
  ```

- Full receipt: `~/Library/Logs/tldw/vz-failure-missing-agent-20260925-1140/receipt.json`.
  All ten cases passed: five positive real-VM tests and five exact-failure
  negative controls, each with no skips or errors. The missing-agent positive
  case recorded a fresh guest nonce and the requested fault VM ID, then
  `guest_transport_timeout` after 15.07 seconds with no fault-VM exec. It
  observed zero persisted sessions/live VMs after failure. Healthy recovery
  executed two commands in one different VM; the negative control executed
  the preserved original agent in its fault VM. Each case had empty VM inventory
  and closed disk handles. Final helper stop, runtime removal, unchanged fault
  sources, and unchanged canonical source hashes all passed.
- Portable verification before the live run: 168 passed, five real-host tests
  deselected; Ruff, Black, shell syntax, and Bandit on touched Python code
  passed. No production fault flag or default CI trigger was added.
- Residual limits: this proves a guest whose agent service starts but never
  connects to VSock. It does not prove kernel boot-hang recovery, actual
  workspace mount/path-escape isolation, or host reboot recovery. Retained
  evidence includes disposable disks and logs; review before deleting them.

## 2026-09-25: Missing-Agent PR Review Rerun (Accepted)

- On `codex/vz-missing-agent-startup-drill`, Qodo review found that the portable
  orchestration mock still modeled four profiles. The mock and expectations
  now cover the fifth `missing_agent` profile; the subprocess fixture is an
  integration test with a less brittle startup deadline. No production runtime
  behavior changed in this review pass.
- Reused the rebuilt Debian arm64 source bundle and signed isolated helper
  documented above. The fresh real-VM workflow exited 0 with ten accepted
  cases, no skips or errors, empty VM inventory, closed disposable disks,
  helper stop, removed private runtime, and unchanged canonical/fault-source
  hashes. Its `input_sha256` for the reviewed missing-agent host test is
  `fdc36061bf104274aa37034c58f7f7c3df3a986d19377600cd4a08881f596fd8`.
  Receipt: `~/Library/Logs/tldw/vz-failure-missing-agent-20260925-pr-review-r1/receipt.json`.
- Portable review suite: 219 passed, 11 host-gated skipped. Ruff and Black
  passed. Bandit found no medium/high issues; low findings were pytest asserts
  in test code. GitHub CI was still queued at the time of this local rerun.

## 2026-09-25: Missing-Agent Startup Drill Preflight (Not Accepted)

- TASK-13243.9 adds a test-only guest launcher and a fifth case to the manual
  failure workflow. Focused portable checks: 167 passed, five live tests
  deselected; Black, Ruff, shell syntax, and Bandit passed.
- The previously accepted source bundle at
  `~/Library/Logs/tldw/vz-launchd-recovery/20260913-1905/source-bundle-final`
  currently lacks `rootfs.img`, although its manifest requires that file.
  The signed helper still verifies, but this host cannot boot a VM from the
  incomplete bundle. The earlier accepted workspace evidence remains historical
  evidence and does not establish acceptance for this new drill.
- No real VM run, guest proof, negative control, recovery, cleanup, or source
  integrity result is claimed for TASK-13243.9. Restore or build and validate a
  bootable Debian arm64 bundle, then run the opt-in workflow into a new private
  evidence directory. Record its exact input and helper hashes, ten case results,
  resource cleanup, and canonical source hashes here before calling the task
  complete.

## Evidence Packet

Each prepared-host evidence packet should include these fields.

| Field | Required content |
| --- | --- |
| Evidence date | ISO date and local timezone. |
| Evidence source | `local-operator`, `workflow_dispatch`, or `nightly-host-gated`. |
| Git state | repository, branch, commit SHA, PR number if applicable, and dirty/clean status. |
| Host identity | Apple silicon model or runner label summary, macOS version, architecture, runner name if CI, and whether the host is dedicated or shared. |
| Host prep | Xcode command line tools availability, SwiftPM availability, `xcrun codesign` availability, Virtualization.framework availability, and runner labels for CI. |
| Bundle/template | source bundle path or registered template id, disposable run bundle path when host smoke materialized one, manifest path if registered, source and run artifact hashes when available, build provenance, and whether validation used canonical bundle or compatibility mode. |
| Helper build/signing | helper binary path, helper version, protocol version, signing mode, entitlements path, entitlement validation result, and skip-sign rationale when signing was skipped. |
| Runtime paths | private runtime directory, socket path, serial-log directory, image-store root, disposable run bundle path, log directory, and evidence that runtime/log directories were owner-only. |
| Commands | exact smoke, helperctl, pytest, workflow, restart-drill, and optional launchd-drill commands that were run. |
| Results | pass/fail/skip for daemon smoke, ephemeral command execution, same-session VM reuse, recovery diagnostics, dry-run reconciliation repair, helper shutdown, and artifact upload. |
| Failure drills | pass/fail/skip for drill-owned stale VM replacement and helper restart drill; include skip reason when `include_failure_drills` was not requested. |
| Launchd drill | pass/fail/skip for `launchd-drill`; include skip reason unless a maintainer explicitly requested LaunchAgent validation. |
| Stale socket drill | pass/fail/skip for `stale-socket-drill`; include runtime directory mode, socket path, command output, helper stdout/stderr paths, and skip reason when not requested. |
| Stuck boot/readiness drills | pass/fail/skip for host-independent helper/runner stuck boot/readiness coverage and any later manual prepared-host drill; include stable failure reason or error code, create-path outcome, session-control outcome, helper stdout/stderr paths, serial-log pointers only, and skip reason when no manual drill was requested. |
| Artifacts | workflow run URL or local artifact root, helper stdout/stderr files, serial logs, pytest logs, workflow logs, and checksums or sizes for retained artifacts. |
| Expected skips | explicit non-blocking skips from the acceptance policy, including missing nightly opt-in, no launchd request, no failure-drill request, or local unprepared-host checks. |
| Blocking regressions | any failed guarantee from the acceptance policy and the first failing command/log pointer. |
| Residual gaps | known unrun or uncovered cases such as host reboot when no manual pre/post drill was run, launchd validation when skipped, stale socket validation when skipped, stuck boot/readiness or guest-agent mismatch beyond host-independent coverage, or broader helper crash classes not covered by the selected drills. |
| Follow-up owner | issue, task, or PR that will address each residual gap. |

Do not paste secrets, API keys, raw user data, or full runner logs into the
tracker. Prefer artifact links, file names, byte sizes, checksums, and short
redacted excerpts.

## Acceptance Checklist

Use this checklist for a complete prepared-host acceptance entry.

| Check | Evidence requirement | Required for default smoke |
| --- | --- | --- |
| Prepared Apple silicon host validation | Host facts and helper/template preflight passed or skipped with an operator-setup reason. | Yes |
| Helper build/sign/start | Helper built or existing binary validated, signing/entitlements state recorded, daemon smoke passed, and socket/log paths were private. | Yes |
| Real `vz_linux` ephemeral execution | A command executed inside a real VM and returned expected stdout/stderr/exit status. | Yes |
| Same-session VM reuse | A second command in the same sandbox session reused the same healthy VM or recorded a blocking failure. | Yes |
| Recovery diagnostics | macOS diagnostics and dry-run reconciliation repair planning ran without mutating session-control rows or terminating VMs. | Yes |
| Helper shutdown/cleanup | The helper stopped on exit and did not leave the accepted socket path behind. | Yes |
| Artifact upload or retention | Helper logs, serial logs, and pytest/workflow logs were retained or an early setup skip explains why none exist. | Yes |
| Failure drills | Drill-owned stale VM replacement and helper restart drill results recorded. | Manual opt-in only |
| Launchd drill | LaunchAgent bootstrap/kickstart/status/bootout drill results recorded. | Manual opt-in only |
| Stuck boot/readiness drills | Host-independent helper/runner tests prove registry/session cleanup; manual prepared-host readiness withholding also records a real handshake, stable timeout, recovery, and artifact pointers without exposing raw serial logs. | Portable boot coverage; manual readiness evidence |
| Host reboot drill | Post-reboot helper/session recovery evidence recorded. | Manual operator procedure only |

## Expected Skip Taxonomy

These states are expected skips or setup gaps, not runtime regressions by
themselves:

- ordinary PR checks do not include the host-gated workflow
- scheduled workflow skipped because
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY` is unset or not `1`
- workflow skipped on a ref other than `main` or `dev`
- hosted CI lacks Apple silicon `Virtualization.framework`
- local machine lacks a prepared bundle, helper, Xcode tools, or entitlements
- failure drills skipped because `include_failure_drills=true` was not requested
- managed helper `restart-drill` skipped because the helper was not started by
  `vz-helperctl.py start`
- `launchd-drill` skipped because no maintainer requested LaunchAgent validation
- manual stuck boot/readiness drill skipped because only host-independent
  helper/runner coverage was requested for the current implementation slice
- host reboot validation skipped because no explicit manual pre/post drill was
  requested for the current evidence packet

If a prepared host passes preflight and then fails helper startup, real
ephemeral execution, same-session VM reuse, recovery diagnostics, cleanup, or
artifact retention, record it as a potential blocking regression and link the
triage issue.

## Latest Evidence

### 2026-09-14: Real advertised workspace mismatch and recovery (TASK-13243.7)

- Local operator run on the same Apple Silicon host: macOS `26.5.2` (`25F84`),
  `arm64`. Checkout `codex/vz-workspace-mismatch-drill` at `c18b1e06eb` plus
  the uncommitted workspace-drill changes. Receipt input hashes identify the
  exact sources and were rechecked against the checkout after acceptance.
- Ran the existing `vz-failure-drill.py --allow-fault-injection` with
  `--source-bundle ~/Library/Logs/tldw/vz-launchd-recovery/20260913-1905/source-bundle-final`,
  `--helper /private/tmp/task-13243-6-swift-diagnostic/debug/macos-vz-helper`,
  and `--evidence-dir ~/Library/Logs/tldw/vz-workspace-workflow-20260914-r1`.
  The helper was rebuilt and ad-hoc signed with the checked-in entitlements;
  signature and `com.apple.security.virtualization=true` preflight passed.
  Canonical Debian arm64 images, shared helpers and launchd were untouched.
- All eight cases accepted: four positive tests passed without skips/errors;
  four negative controls reached their intended assertion failures after real,
  completed, exit-zero execution with exact stdout. Capability, readiness and
  protocol cases were rerun rather than inferred from older evidence.
- Workspace-positive VM `1c36b7eb-f7f6-4436-9f3a-d68009b2edf6` advertised
  `/workspace-mismatch/88c2fc7cf92740c69dcd5a9fc2a5d85e`, with supported
  protocol and both `exec` and `output_cap_v1` intact. The runner rejected with
  only `vz_linux_guest_agent_workspace_mismatch`; no exec reached that VM.
  Recovery printed `workspace-drill-first` and `workspace-drill-reuse`, both
  exit zero in replacement VM `087b5173-e345-44a5-93ab-554abd2c5dd0`.
- Workspace-negative VM `071cd606-8c7b-4269-b2b6-515b59e36b5f` advertised
  `/workspace` and executed `workspace-drill-first` successfully. Only the
  advertised root was restored; real admission remained enabled. Both controls
  retain returned helper metadata and dispatch evidence in `guest-workspace.json`.
- Final receipt: `ok=true`, `errors=[]`, canonical and all four fault-source
  hashes unchanged, empty VM inventory, closed disposable disks, helper stopped,
  socket/PID absent, private runtime removed. Public reconciliation was empty
  after workspace rejection and final cleanup. Evidence/clones are retained.
  `receipt.json` SHA-256:
  `9a87be75caab1a23f4d7b057719e50e5adcd59d31625d8520e181e45e09b4261`;
  exact signed `helper-used` SHA-256:
  `049ab407cf4e0cb9b858f1b877d9f1df5bdb2c90c048230c2ff16eee9f486638`.
- Verification: **419 portable Python tests passed, 3 live drills deselected**;
  **95 Swift tests** and the normal Go agent suite passed. Workspace overlay
  compiled; workflow/receipt regressions were observed RED then GREEN. Ruff,
  touched-scope Black and diff checks passed. Bandit found no new issues;
  five existing B108 literals remain in untouched runner tests (B101 test
  assertions excluded). Independent review found no actionable issues.
- This proves advertised-metadata admission, not mount isolation or path-escape
  resistance. Full server tests, reboot, missing-agent and early boot hangs were
  not run. These and broader residual gaps remain tracked by #1442.

### 2026-09-14: PR #2964 review-fix real workflow verification

- Repeated all six cases with the rebuilt, ad-hoc-signed helper after mapping
  bridge-detected exec-response version errors to `guest_protocol_mismatch`.
  The protocol drill now uses public reconciliation instead of a redundant
  private orchestrator assertion; its platform gates reference tracker #1442.
- Accepted packet: `~/Library/Logs/tldw/vz-protocol-workflow-20260914-qodo-r1/`.
  `receipt.json` SHA-256:
  `c8b02771cf643b1dcbd42ca35d58c32a8ba8864067c46be34217759658b1e3a3`.
  The exact signed executable is retained as `helper-used`, SHA-256:
  `049ab407cf4e0cb9b858f1b877d9f1df5bdb2c90c048230c2ff16eee9f486638`.
- All three positive cases passed, without skips/errors; all three negative
  controls produced their intended failure after real exit-zero execution.
  The protocol-positive guest sent version `999`, rejected in 2.184 seconds
  before exec. Both healthy commands completed in replacement VM
  `f49e3d89-78b0-450f-8d5c-94de325690ec`. The protocol-negative guest sent
  version `1` and completed `protocol-drill-first` with validation still enabled.
- Public reconciliation reported zero persisted controls and live VMs after
  rejection and final cleanup. The final receipt reports `ok=true`, no errors,
  empty VM inventory, closed allocated disks, stopped helper, absent socket/PID,
  removed runtime directory, and unchanged canonical/all fault-source hashes.
- Verification: **358 focused Python tests passed, 3 explicit host-gated skips**,
  **95 Swift tests passed**, and the normal Go agent suite passed. The new
  exec-response regression failed before the fix and passed afterward, with
  request-ID/malformed-response controls. Ruff, Black, scoped Bandit (excluding
  test assertions), and diff checks passed; independent review found no issues.
  Later exec-response mismatch classification is covered by Swift server/bridge
  tests, not live fault injection. The full server suite and reboot drill were
  not run; the existing broader residual gaps remain unchanged.

### 2026-09-14: Real guest protocol mismatch and recovery (TASK-13243.6)

- Ran the checked-in `vz-failure-drill.py` on the same Apple Silicon host and
  canonical Debian arm64 bundle, with a rebuilt, ad-hoc-signed Swift helper.
  The helper now reports `guest_protocol_mismatch` for a rejected guest wire
  version; no production fault flags, shared-helper changes or host reboot.
- Accepted packet: `~/Library/Logs/tldw/vz-protocol-workflow-20260914-r1/`.
  `receipt.json` SHA-256:
  `e2bf4d6feda62bc6c10d088411df3e68b67ee4d20d4181dfe855b87779f56486`.
  Receipt input hashes identify the exact scripts, fixtures and agent sources.
  `helper-used` retains the exact signed executable after the run, matching
  receipt helper SHA-256
  `9842e41ac810fa4e72bc6246f9e31c0c455246c9f2d260258d63304e1dcd75e8`.
- All six cases accepted: three positive tests passed without skips/errors;
  three negative controls produced their intended assertion failures backed
  by completed, exit-zero, exact-output guest execution. Existing capability
  and readiness drills were rerun, not inferred from earlier packets.
- The protocol-positive guest proved nonce-correlated version `999` in VM
  `eca2b3d2-e137-48c5-8689-8c5deb7e836b`. Create rejected it in 4.068 seconds
  with `guest_protocol_mismatch: protocolMismatch`; no exec reached that VM
  and no reusable control/VM state remained. Healthy recovery printed
  `protocol-drill-first` and `protocol-drill-reuse`, both exit zero, in the
  same replacement VM `a84cc8e2-16f8-4a0a-9ec7-43f655245fa0`.
- The protocol-negative guest sent supported version `1`, with the real helper
  gate still enabled. VM `04815412-7310-4827-b6fe-4c52432ee5c5` executed
  `protocol-drill-first` successfully before the intended rejection assertion
  failed. `guest-protocol.json` and per-case `result.json` retain that proof.
- Final cleanup: VM inventory empty, all allocated disk handles closed,
  helper stopped, socket/PID absent, short runtime directory removed.
  Canonical and all three fault-source hashes unchanged; receipt `ok=true`
  and `errors=[]`. Evidence and image-store clones are deliberately retained.
- Verification: **358 focused Python tests passed, 3 explicit host-gated skips**;
  all **94 Swift tests** and the normal Go agent suite passed. New workflow,
  diagnostic and proof-validation regressions were observed RED then GREEN.
  Ruff/Black, diff checks and scoped Bandit passed (test assertions excluded).
  An initial restricted test run could not invoke `ps`; the authorized rerun
  passed without weakening cleanup tests. Independent review found no
  actionable issues. The full server suite was not run. Host reboot,
  missing-agent, early boot hangs and other deferred fault classes remain open.

### 2026-09-14: PR #2962 review-fix real workflow verification

- Repeated the checked-in operator command on the same Apple Silicon host,
  using the same canonical Debian arm64 bundle and signed helper as below.
  Includes process-group cancellation, dependency provenance, and contextual
  readiness-overlay errors; no production runtime changes or host reboot.
- Final packet: `~/Library/Logs/tldw/vz-failure-workflow-20260914-final/`.
  `receipt.json` SHA-256:
  `2b8a4d8d5a10642354bb8ff9de0a1d3fb5b77e7c57f034f9df59a54282766456`.
  The receipt includes the exact workflow, helperctl, materializer, guest-source,
  fixture, and helper hashes used for this run. The earlier passing
  `vz-failure-workflow-20260914-review` packet is retained, but final acceptance
  uses the rerun after the independently identified spawn-cancellation fix.
- Both positive cases passed with no skips/errors. Both negative controls
  produced their exact expected assertion failures and verified completed,
  exit-zero guest execution with exact stdout in the fault VM. Healthy-session
  recovery and same-session reuse passed in both positive cases.
- Cleanup: empty VM inventory, no allocated disk handles, helper stopped,
  socket/PID absent, private runtime removed. Canonical and both prepared
  fault-source hashes remained unchanged; `errors` was empty and `ok` true.
- Separate real-process regressions reproduced orphaned descendants on timeout
  and parent-only SIGTERM before the fix; both passed after process-group
  termination and direct-child reaping were added. This is process cancellation
  evidence, not a claim of live-VM SIGTERM or host-reboot acceptance.
- A deterministic signal during process creation also reproduced a child leak.
  Deferring handled termination signals until ownership registration fixed it;
  independent SIGINT/SIGTERM probes confirmed reaping and handler restoration.
- Final focused suite: **295 passed, 2 expected host-gated skips**, four existing
  warnings. Go agent suite, Ruff (including annotations/docstrings), Black,
  diff checks, and scoped Bandit scans passed (test assertions excluded).
  Independent re-review found no remaining issues. Full server/Swift suites
  and additional failure classes remain outside this Python/test-only change.
- Subsequent reviewer-guide follow-up: **298 passed, 2 expected host-gated
  skips**, four existing warnings. Added combined installation/termination
  failure regressions and a startup signal-burst test; documented first-signal
  coalescing and the boot-artifact/metadata fingerprint boundary. This later
  change preserves the primary exception with a cleanup note; it does not
  alter Go overlays or normal VM behavior. No additional real VM run was
  performed for this exception-reporting/test/documentation-only follow-up;
  the packet above records the exact earlier workflow bytes it exercised.

### 2026-09-13: Checked-in real guest-failure workflow (TASK-13243.5)

- Source: `codex/vz-failure-drill-workflow`, based on merged PR #2960
  (`ebdeeac384c58559fa90fd3a5f79f5262ae190d5`), same Apple Silicon host.
  No reboot or changes to the canonical Debian arm64 image.
- Command: `python tools/macos-vz-helper/scripts/vz-failure-drill.py
  --allow-fault-injection --source-bundle "$SOURCE_BUNDLE"
  --helper "$HELPER_BINARY" --evidence-dir "$EVIDENCE_DIR"`, from the project
  environment. The exact operator recipe and prerequisites are in the helper
  README. No local-only driver or fault overlay was needed.
- Final packet: `~/Library/Logs/tldw/vz-failure-workflow-20260913-r2/`.
  `receipt.json` SHA-256:
  `b961d96340c5779de938734737f085e16bfa518b2ff2c2eb146a8e87a442963e`.
  The earlier `vz-failure-workflow-20260913-r1` packet is also retained;
  final acceptance uses r2 after the negative-control review fix.
- Both overlays were built from the checkout and installed through separate
  healthy preparer VMs into offline image-store clones. Installed bytes matched
  the built binaries, and ext4 filesystem checks passed before/after patching.
- Positive capability-mismatch test: **1 passed, 0 skipped/errors**; real missing
  `exec` metadata was rejected without dispatch, then two healthy commands reused
  VM `5845c56e-678c-4780-9d29-0741ad3f811d`.
- Positive readiness test: **1 passed, 0 skipped/errors**; a fresh acknowledged
  handshake proof preceded `guest_transport_timeout` at **15.071 seconds**.
  Two healthy commands then reused VM `d5f5a6c3-075f-4c46-a594-2531bd1a5c42`.
- Both negative controls: exactly **1 expected assertion failure, 0 skips/errors**
  per case. The final wrapper also verified the guest receipts: completed run,
  exit zero, exact expected stdout, and execution in the fault VM. A cancelled
  run or unrelated boot failure cannot count as successful negative evidence.
- Final cleanup: empty VM inventory, no disposable disk handles before helper
  stop, managed helper stopped, socket/PID absent, private runtime removed.
  Canonical and both prepared fault-source hashes remained identical. Canonical
  rootfs SHA-256 remained
  `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`.
- Portable verification: **255 passed, 8 explicitly gated skips** across workflow,
  helperctl, materializer, mismatch, and readiness tests; normal Go agent suite
  passed. Ruff/Black and scoped Bandit validation passed. The eight portable-run
  skips are not substituted for live acceptance: all four explicit live cases
  ran without skips. Full server and Swift suites were not run for this
  Python/test-fixture-only slice.
- Remaining gaps unchanged: host reboot, kernel boot hang, missing agent,
  protocol-version/workspace mismatch injection, and broader crash classes.
  This closes reproducible preparation for the two existing guest drills, not
  every lifecycle failure mode. Follow-up remains tracked by #1442.

### 2026-09-13: PR #2960 review follow-up (portable verification)

- Both drills now share the readiness drill's resilient ownership-scoped cleanup.
  A mismatch session or VM deletion error no longer skips later cleanup attempts;
  accumulated errors and remaining owned VMs are retained before failure.
- Separate session-error and VM-error regressions failed against the old cleanup
  and passed after the fix. The focused suite passed **71 tests**, with two
  intentional host-gated skips; the two drill modules passed **18 unit cases**.
  Unit markers and missing docstrings were added; Black and Bandit were clean.
- No live VM was rerun for this review patch. The accepted runs and source hashes
  below describe their retained historical test versions, not the modified test
  files. No production runtime, helper, or guest-agent source changed.

### 2026-09-13: Real acknowledged-guest readiness timeout and recovery

- TASK-13243.4, branch `codex/vz-readiness-timeout-host-validation`, stacked on
  capability-mismatch checkpoint `f4bab1a456`. No production code changed.
  The test observes the existing service/helper create and exec paths unchanged.
- Evidence root: `$HOME/Library/Logs/tldw/vz-readiness-timeout/20260913-r2`.
  Each of `red`, `green`, and `review` retains `receipt.json`, `host.xml`,
  `host.log`, helper/serial logs, LaunchAgent plist, and
  `pytest/test_vz_linux_readiness_timeou0/guest-readiness.json`.
  Retained source/overlay/binary/driver artifacts describe the test-only guest.
- The offline preparation VM installed a guest that writes a fresh nonce proof
  only after validating the real VSock handshake acknowledgement, then withholds
  readiness with a two-minute watchdog. The host uses a 15-second startup limit.
  Proof reads reject non-regular files and are bounded; missing or stale proof
  cannot turn a failed boot into a passing timeout test.
- First accepted run: **1 passed, 0 skipped/errors**, 19.408 seconds total;
  create returned `guest_transport_timeout` after 15.184 seconds. Final repeat
  after cleanup-review fixes: **1 passed, 0 skipped/errors**, 18.955 seconds
  total; timeout after 15.168 seconds.
- Final helper generation: `C763A2E3-FCB9-4138-9BAB-9FEED7BF3938`.
  Fault VM `da368a46-e204-4e2e-b517-a99ce767cc53` acknowledged the fresh nonce,
  but received no exec. Public reconciliation then showed zero persisted
  controls and zero live VMs. Healthy VM
  `67baf3ba-4c61-4f35-b120-1305f6558776` executed both exact-output commands in
  one new session on the same helper, then session/control cleanup completed.
- Negative control: the same test-only guest was allowed to send `ready`.
  It really executed and returned `readiness-drill-first\n`, exit 0. Pytest
  failed specifically at "Readiness withholding did not fail" (7.240 seconds,
  one failure, no skips/errors). No helper response or guest proof was mocked.
- Both accepted runs verified no open handles on their disposable rootfs disks
  before helper shutdown, in addition to empty registries and control state.
  All r2 attempts ended with absent owned LaunchAgents, unavailable sockets,
  removed runtime directories, and unchanged canonical and fault-source hashes.
  Canonical rootfs: `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`.
  Fault-source rootfs: `87bf764f9e10bc14c605f5fbc83a7c2fca6fe615b8922f67498bbb8900f21432`.
- Initial attempt retained separately in `vz-readiness-timeout/20260913`:
  its test-only nil-channel wait caused an early transport close after 2.894
  seconds, so the test correctly failed rather than accepting it as a timeout.
  The nil-channel pattern reproduced Go's fatal deadlock on the host. A fresh
  timer-backed test image fixed the injection, not production code. A copied
  artifact-kind label was corrected before that first live attempt; the original
  preparation receipt keeps the prior metadata hash.
- Review regressions cover FIFO proof blocking, cleanup continuing after session
  and VM exceptions, and session deletion returning false while the row remains.
  All were observed failing before their test-harness fixes. Final-repeat test
  SHA-256: `464968624f50de905245f12c9ac636ffd35309b0da98cd610b4e3e217f6a463a`.
- Final focused Python/workflow-contract verification: **69 passed, 2 intentional
  host skips**, 1.64 seconds. Black and diff checks passed; Bandit reported zero
  findings. Independent review confirmed the final source hash and evidence.
  A missing lifecycle-spec link exposed by the doc-contract test was restored.
  The repository-wide suite was not run for this test-only slice.
- Scope: acknowledged-handshake readiness withholding only. This does not prove
  kernel boot hangs, a missing guest agent, protocol-version mismatch, host reboot,
  arbitrary stop-error recovery, or escaped guest-process containment. The two
  healthy commands use a new session, not in-place repair of the failed session.

### 2026-09-13: Real missing-exec guest rejection and healthy recovery

- TASK-13243.3, branch `codex/vz-guest-mismatch-host-validation` at `dae1f00974`,
  based on merged PR #2955 (`beac8e9449`). No production runtime or policy code
  changed. The test observes the real helper's VSock-derived metadata and
  delegates create/exec calls unchanged.
- Corrected the earlier storage interpretation: Apple's volume-capacity API
  reported 114,785,116,224 bytes available for important usage while the plain
  available-capacity value was 467,423,232 bytes. A bounded write succeeded.
  `df` alone was not sufficient to declare the test blocked or request manual
  deletion. The earlier `ENOSPC` attempt remains recorded below; the following
  retries succeeded without further storage deletion.
- Evidence root:
  `$HOME/Library/Logs/tldw/vz-guest-mismatch/20260913`.
  `green`, `red`, and `review` each contain `host.xml`, `host.log`,
  `receipt.json`, the generated LaunchAgent plist, helper/serial logs, and
  `pytest/test_vz_linux_rejects_real_gue0/guest-mismatch.json`.
- Normal guard: **1 passed, 0 skipped, 0 errors**, 8.942 seconds. A separate
  fresh-clone repeat after the negative control also passed with **0 skips and
  errors**, 11.232 seconds. The operator-owned LaunchAgent wrapper bootstrapped,
  checked, and stopped a unique helper for each attempt. No default helper or
  host reboot was involved.
- Final-repeat helper generation:
  `235162A9-B6E7-4409-9B23-70D5A80E6B66`. Fault VM
  `9b764241-d778-4e3a-9b2f-47021a13b27e` reported known capabilities containing
  only `output_cap_v1`. Its run failed with
  `vz_linux_guest_agent_required_capability_missing`, no stdout, and no exec
  dispatch. Public reconciliation then reported zero persisted controls and
  zero live VMs.
- A new healthy session on the same helper ran both exact-output commands in VM
  `e8a0f98c-f716-425b-885d-5872a54b785d`, proving same-session reuse after the
  failed session. This is not an in-place repair of the incompatible guest or
  reuse of the failed session.
- Negative control: a test-process-only fixture bypassed the runner's
  create-time compatibility classification. The independent test observer
  still saw the missing `exec` capability. VM
  `1eba8794-237d-45a3-80dd-49175e7e70bf` actually executed the command and
  returned `mismatch-drill-first\n`, exit 0. The test failed specifically at
  "Mismatched guest was not rejected" (5.140 seconds), not on a boot error.
  No production file was edited for the negative control.
- All three attempts finished with empty helper registries, absent LaunchAgents,
  unavailable sockets, and removed private runtime directories. Both accepted
  runs also verified session removal and zero persisted controls. The negative
  control's failure cleanup reported no remaining owned VMs.
- Canonical rootfs stayed
  `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`;
  kernel, initrd, manifest, and build-info hashes matched before/after every
  attempt. The unbooted fault-source rootfs also stayed
  `3406ae92718845dfbf52f8c89628ae237f853d3dd527536dfdc2c9611ab5405c`.
  Source provenance and retained helper signature/hash are recorded below.
- Final focused portable checks: **35 passed, 1 intentional host skip**, 2.06
  seconds. Black and diff checks passed; Bandit reported zero findings.
  Independent review of the green/red evidence found no material issues.
  The repository-wide test suite was not run for this test-only slice.
- Scope: this proves rejection of a known missing required capability and
  subsequent healthy execution/reuse. It does not prove protocol-version
  mismatch, a missing agent, stuck boot/readiness, host reboot, or containment
  of escaped guest descendants. Those remain distinct evidence items.

### 2026-09-13: Post-merge cleanup and guest-mismatch preparation (not acceptance)

- PR #2955 merged as `beac8e9449`. Its clean worktree and local/remote branch
  were removed after ancestry and active-process checks. The main checkout and
  divergent local `dev` were left untouched.
- The exact signed helper and local database artifacts were retained under
  `$HOME/Library/Logs/tldw/vz-launchd-recovery/20260913-pr2955-review/worktree-retained`.
  The helper SHA-256 is
  `a4e988165f18c296dc88e7e80eaddaa0f935bbfb9812463ec0988d2cfa2bff85`;
  the copied signature verified successfully.
- TASK-13243.3 continues the existing guest-agent mismatch contract. The runner
  already rejects explicit mismatches at create time and before session reuse;
  this slice adds a manual real-host test, not another production policy layer.
- Prepared a test-only guest using a Go build overlay that changes advertised
  capabilities from `["exec", "output_cap_v1"]` to `["output_cap_v1"]` while
  leaving the exec handler intact. A separate real VM installed it into an
  offline image-store disposable clone. `e2fsck -fn` and installed-binary `cmp`
  passed; the preparation VM/helper were stopped and their socket removed.
- Preparation receipt and logs:
  `$HOME/Library/Logs/tldw/vz-guest-mismatch/20260913/prepare`.
  The unbooted test-only fault source is
  `../image-store/runs/fault-source/bundle`; its build-info explicitly records
  the overlay and parent hashes. Rootfs SHA-256:
  `3406ae92718845dfbf52f8c89628ae237f853d3dd527536dfdc2c9611ab5405c`.
- Canonical `source-bundle-final` rootfs stayed
  `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`;
  kernel, initrd, manifest, and build-info hashes also stayed unchanged.
- At this preparation checkpoint, live mismatch acceptance and its negative
  control had not run. The host reported `ENOSPC` while creating the next
  evidence directory, before starting a test helper. Subsequent capacity checks
  and successful retries are recorded above; the offline installation alone
  was not counted as a rejection test.
- Space cleanup removed only the inactive `preparer-boot` clone and PR #2955's
  `public-diagnostics` / `linux-regressions` disposable run disks. Their source
  bundles, logs, test binaries, and acceptance receipts remain. The PR #2955
  disk hashes/manifests are retained in its evidence root under `postmerge-gc`;
  the preparer manifest is retained as `prepare/preparer-run-manifest.json`.
- Portable verification: 35 passed, 1 intentional host skip; Bandit zero
  findings. Review added ownership-based cleanup for a lost create reply, with
  a verified failing regression before the fix. Protocol-version mismatch,
  missing-agent, stuck boot/readiness, and reboot remain separate live gaps.

### 2026-09-13: PR #2955 review verification

- Rebased onto `dev` at `b6cf7fd1d5`; the rebase introduced no changes to the
  helper, guest, or sandbox runtime sources. Review fixes change test fixtures
  and documentation, not the production cancellation/drain implementation.
- The revised host test uses public `SandboxService.macos_diagnostics()`
  reconciliation to observe healthy, stale, and removed session controls,
  rather than reading the private orchestrator. Generation persistence remains
  covered by the focused runner tests; the host drill checks helper generation
  change, VM replacement, and replacement reuse through supported observations.
- Fresh disposable clone from the preceding `source-bundle-final`: **1 passed,
  0 skipped, 0 errors**, 14.05 seconds. The first VM was
  `40c207d3-19ef-472d-8bbf-ba8bbc668c47`; replacement and third-command reuse used
  `4c05cb1b-cd1c-416c-8cf7-681ad0d87a2a`. All three commands returned exact stdout
  and exit 0. Cleanup reported zero persisted controls/live VMs, destroyed
  session, absent LaunchAgent, unavailable socket, and removed runtime.
- Evidence root: `$HOME/Library/Logs/tldw/vz-launchd-recovery/20260913-pr2955-review`.
  `host.xml`, `host.log`, and
  `host-pytest/test_vz_linux_session_recovers0/launchd-recovery.json` contain the
  acceptance results. Source rootfs SHA256 remains
  `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`.
- Linux tests now use test-owned release files instead of signaling saved
  numeric PIDs. A fresh liveness acknowledgement proves the escaped child
  still holds the pipes after `Exec` returns; cleanup then releases it.
  Completion and acknowledgements replace five-second elapsed assertions;
  a one-minute watchdog only detects deadlocks. Both waiting-parent and
  exited-parent timeout cases are covered. No production timer injection was
  needed. Ten repetitions of `TestGuestServerExec*` passed in the disposable
  Linux VM (`linux-final/guest-review.test.stdout.log`), including 1,000 fast
  output commands. Test-binary SHA256:
  `85e9162e67d841ae51430753cc54a1fa2ab1da2ab1bc37e594c2311973e8164a`.
- A temporary Go overlay removed only the cancellation pipe-close callback,
  leaving repository production code unchanged. Its escaped-output-limit test
  failed as intended at the deadlock watchdog (60.06 seconds, exit 1), then
  cooperatively released the child. This confirms natural child expiry cannot
  mask a missing drain bound. `linux-final/guest-no-drain.test.stdout.log` and
  `linux-final/linux-review.json` retain the negative result and cleanup receipt.
  Negative-binary SHA256:
  `e44af3a1d601770a5276faa4f332f232e70b221d3456107d51bb20c7b0088d27`.
- Native Go suite/race checks and native/Linux guest vet passed. Focused Python
  suites passed 224 tests with the one manual drill intentionally opted out.
  Bandit passed with the documented B108 exception; no additional suppression
  was added. No host reboot or broader descendant-containment claim is made.

### 2026-09-13: live-session recovery after a launchd restart

- Scope: `TASK-13243.1` and `TASK-13243.2`, on
  `codex/vz-launchd-vm-validation` based on local dev `c70387f496d8`.
  Same Apple Silicon host as the launchd smoke below, macOS 26.5.2 (25F84),
  helper version `0.1.0`, protocol `1`. No host reboot was performed.
- Durable private artifacts:
  `$HOME/Library/Logs/tldw/vz-launchd-recovery/20260913-1905`.
  The final accepted packet is `final.xml`, `final.log`, `final.exit`, and
  `final-pytest/test_vz_linux_session_recovers0/launchd-recovery.json`, with the
  generated plist and helper/serial logs beside the receipt.
- The new manual test is
  `tldw_Server_API/tests/sandbox/test_vz_linux_launchd_recovery_host_gated.py`.
  It uses `TLDW_SANDBOX_VZ_LINUX_E2E=1` and
  `TLDW_SANDBOX_VZ_LINUX_LAUNCHD_RESTART_DRILL=1`, an explicitly selected signed
  helper binary, and a disposable bundle from `prepare-smoke-bundle.py`.
  The repeatable command is in the helper README's
  **Live-Session Launchd Restart** section. Default/scheduled smoke selection
  and helper startup behavior are unchanged.
- Accepted result: **1 passed, 0 skipped, 0 errors**, pytest exit `0`, in
  15.56 seconds. All three `/bin/echo` commands returned their exact stdout
  tokens and exit `0`. Launchd restarted the helper only after the first
  successful session command, not merely before VM creation.
- Helper generation changed from `96271C27-C7AA-48A1-A743-B312E70CC1B7` to
  `AF1778A8-FCFE-454E-8152-A384E3DF9A1B`. The stale session control was still
  present immediately after restart. Normal service execution replaced VM
  `130792b5-7cfd-41fc-8851-fea97b98602b` with
  `ddfbd394-58db-4a5c-b450-79a3c2367f01`; the third command reused the latter.
- Cleanup: session destruction succeeded, session control was removed, the
  helper's VM registry was empty, the unique LaunchAgent was absent, its socket
  was unavailable, and the private runtime directory was removed. Evidence and
  disposable image-store disks were deliberately retained.
- Negative control: temporarily replacing the live-session `kickstart` with
  `status` produced the expected `helper_generation_unchanged` failure after
  successful guest output. `review-negative.xml` records one failure and no
  skips or errors; `review-negative-pytest/` records the original failure,
  bootstrap/kickstart/bootout results, and cleanup. The callback preserves
  exceptions until lifecycle results are returned, then re-raises them. The
  checked-in test restores the real restart operation. The earlier
  `diagnostic*` packet exposed missing lifecycle results on callback failure.
- A real defect was found before acceptance: two attempts completed an echo
  with exit `0` but empty output. Instrumentation showed the guest itself
  reported zero observed stdout bytes, excluding Python stream loss. A
  Linux-arm64 regression against the old code failed in all ten repetitions;
  native macOS repetitions alone had not reproduced it. Those failed runs
  remain in `negative*`, `accepted*`, and `linux-output-red*`; none is counted
  as restart acceptance.
- Fix: `runExecWithOutputLimit` now drains both output-pipe readers before
  `cmd.Wait()` can close their pipes. On timeout or output-limit cancellation,
  a one-second drain grace bounds readers retained by escaped descendants;
  process-group cancellation remains active until draining finishes. The
  corrected Linux regression passed
  1,000 fast commands; ten repetitions also passed output-cap/UTF-8 checks and
  both descendant-pipe timeout cases (parent waiting and parent already exited),
  plus escaped-descendant timeout/output-limit cases. The latter completed in
  approximately three seconds and one second, respectively, rather than the
  eight-second failures recorded in `review-red/` against the ordering-only
  fix. `final-build/linux-regressions.stdout.log` and
  `final-build/offline-refresh.json` contain the final proof. An independent
  second review found no remaining actionable findings after these fixes.
  `linux-output-green*` is an earlier diagnostic run with passing guest tests
  but a Python fixture teardown error, not the final acceptance packet.
- Bundle provenance: a private offline APFS clone was refreshed with the new
  agent, then checked with `e2fsck -fn` and an extracted-binary `cmp`. Kernel,
  initrd, and manifest were unchanged. Original source rootfs SHA-256 remained
  `1083decfb5089e904440d2506e40be78645bdb687e8ce1d220f1b57ba27f7cca`;
  refreshed rootfs SHA-256 is
  `5367aca9725b75bb3fce1465fdc3c3d841a9970126e1e17f4608fb611beb3cc2`.
  Installed agent SHA-256 is
  `4e4d1186f0e4830769999951ba5566815f6d4733f9c7e688474b21086c226a3f`.
  `source-bundle-final/build-info.json` records the dirty worktree build
  provenance; only `final-image-store/runs/launchd-recovery-final/bundle` was
  booted for final acceptance. The source bundle was not booted or mutated.
  The earlier `source-bundle` and `fixed*` packet retain the ordering-only
  build and its ordinary restart pass, not the final bounded-drain build.
  Go 1.26.2 embedded the outer checkout's revision `1600d9b8c8`, while the
  module and Git worktree were verified at `d9d936b612` plus the retained
  `final-build/guest-source.patch`. Use this explicit worktree provenance,
  binary hash, and `final-build/guest-build-info.txt`, not the embedded revision
  alone, to identify the tested build.
- Supporting verification: focused helperctl/runner suites passed 224 tests,
  with only the explicitly disabled real restart test skipped. Native Go
  `go test ./...` and guest race checks passed. Correction during PR preparation:
  the retained Bandit report contains one B108 warning for the short `/tmp`
  parent, not zero findings. `mkdtemp` atomically creates a random, owner-only
  directory (mode `0700`); the test now documents a line-specific B108 exception
  for that reviewed false positive. This evidence does not prove host reboot recovery,
  arbitrary helper/guest crash classes, escaped-descendant containment, or
  broader network policy enforcement.

### 2026-09-13: real VM smoke through a launchd-managed helper

- Evidence source: local operator run, authorized for launchd-managed real VM
  validation after PR `#2628`. Branch `codex/vz-launchd-vm-validation` started
  from local `dev` at `c70387f496d82fcee92926bf3715bf5cd240ba88`;
  this slice changes evidence documentation and Backlog only.
- Host: Apple silicon `arm64`, macOS `26.5.2` build `25F84`, Darwin `25.5.0`;
  shared developer host. Capture time was approximately 11:29 PDT.
- Durable artifact root:
  `$HOME/Library/Logs/tldw/vz-launchd-vm-validation/20260913-1817`.
  The evidence root, helper log directory, serial directory, image store, and
  short runtime directory were private to the operator (`0700`). No raw logs
  or disk images are committed to the repository.
- Helper: freshly built from the worktree with `swift build`; signed using
  `tools/macos-vz-helper/macos-vz-helper.entitlements`. `codesign --verify
  --strict` passed and the signed binary contained
  `com.apple.security.virtualization=true`. Live ping reported helper
  `0.1.0`, protocol `1`.
- Guest provenance: the durable June Debian bookworm arm64 bundle was preserved.
  A separate copy of its rootfs received the current Linux arm64 guest built
  with `CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath`; the guest
  includes buffered-reader fix `dfa67a49927cef63e80bf0903d55109ec283254b`.
  The kernel and initrd were unchanged. This was an offline guest refresh,
  not a fresh Debian distribution build. The copied image's pending ext4
  journal was recovered, its final `e2fsck -fn` passed, and extraction plus
  `cmp` verified the installed executable exactly matched the new binary.
  `source-bundle/build-info.json` records this provenance.
- Image-store preparation: `prepare-smoke-bundle.py` registered the refreshed
  source and materialized run `launchd-13243` beneath
  `<artifact-root>/image-store/runs/launchd-13243/bundle`. Only that disposable
  bundle was passed to VM execution. Relevant SHA-256 values:

  | Artifact | SHA-256 |
  | --- | --- |
  | Original source rootfs, unchanged | `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf` |
  | Refreshed source rootfs, unchanged during smoke | `1083decfb5089e904440d2506e40be78645bdb687e8ce1d220f1b57ba27f7cca` |
  | Disposable rootfs after smoke | `1e31e380439d702b580314eac8da68f23dd917551dc8f795cc08ee8dc5188427` |
  | Installed guest executable | `56e21f6ece89ec94832277bf309dd64d9b21d6dec674a1aa765317117eb61012` |

- Main command, from the isolated worktree, after clone preparation:

  ```bash
  evidence_dir="$HOME/Library/Logs/tldw/vz-launchd-vm-validation/20260913-1817"
  runtime_dir="/private/tmp/tvz-13243.bwB1Gg"
  repo_python="/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python"
  export PYTEST_ADDOPTS="--junitxml=$evidence_dir/pytest.xml --basetemp=$evidence_dir/pytest-data -o junit_logging=all"

  "$repo_python" tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
    --bundle "$evidence_dir/image-store/runs/launchd-13243/bundle" \
    --helper "$PWD/tools/macos-vz-helper/.build/debug/macos-vz-helper" \
    --socket "$runtime_dir/helper.sock" \
    --log-dir "$evidence_dir/helper-logs" \
    --plist-output "$runtime_dir/launchd.plist" \
    --label org.tldw.macos-vz-helper.drill.task13243 \
    --python "$repo_python" \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --write-plist --create-dirs
  ```

  This is the recorded invocation, not a rerun recipe: the temporary runtime
  directory has been removed. A repeat must allocate a new private runtime,
  image-store run ID, pytest output directory, and unused LaunchAgent label.
- Results: drill exit `0`; launchd preflight, signing, bootstrap, status,
  kickstart, helper readiness, VM smoke, and bootout passed. Real pytest
  results were **3 passed, 11 deselected, 0 skipped**, in 5.83 seconds:
  ephemeral command stdout/exit assertions, two successful same-session
  commands with identical VM IDs, and recovery diagnostics/dry-run repair.
  The reuse test also asserted session destruction and removal of session
  control. The managed-socket smoke did not start a second direct helper.
- Cleanup: post-drill `launchctl print` returned `113` (service absent);
  no process held the drill helper executable open. Helper status reported
  `helper_not_running` and failed ping, as expected after shutdown. Bootout
  left an inactive socket (`0755` beneath its `0700` parent); the operator
  explicitly removed it and the temporary runtime directory. Automatic
  socket unlink on launchd termination is not claimed. Generic helper status
  additionally reported the pre-existing default `launchd_plist_mismatch`;
  that default plist was outside this isolated drill and was not changed.
- Preparation incident: the existing Fusion builder had an inactive disk lock
  dated June 15. After checking that no VM process or open disk handle existed,
  the lock was moved into the evidence directory. A persistent launcher
  session kept the builder alive; its verified address was `192.168.241.128`.
  After image preparation, task-owned guest staging files were removed and
  the builder was shut down with `vmrun stop ... soft`; `vmrun list` reported
  zero running VMs.
- Artifacts: `helper-build.log`, `guest-inspection.log`, `guest-refresh.log`,
  `guest-filesystem-recovery.log`, source/run checksums, bundle provenance,
  `prepare-clone.log`, `launchd-drill.log`, `launchd-drill.exit`, `pytest.xml`,
  `pytest-data/`, retained `launchd.plist`, `path-permissions.log`,
  `launchd-after.log`, `helper-after.json`, and helper/serial logs. The private
  artifact root also retains the tested helper and guest binaries, refreshed
  source and disposable run bundle, and verified `artifact-checksums.sha256`.
- Supporting portable verification: 22 launchd/helper-smoke contract tests
  passed, and `go test ./internal/guest` passed. No runtime source changed.
- Residual scope: kickstart preceded VM creation; this packet does not prove
  recovery of a live VM across helper restart. Host reboot, live mismatch
  injection, stuck boot/readiness injection, and scheduled CI remain separate
  evidence items. No new runtime regression was observed in this drill.
- Follow-up owner: `TASK-13243`, under issue `#1442`. Repeat this acceptance
  slice when the helper lifecycle, guest transport, or image preparation changes.

### 2026-07-03: local-operator launchd drill on `codex/vz-launchd-drill-evidence`

- Evidence source: local operator run on the same prepared Apple silicon macOS
  host, using the manual `vz-helperctl.py launchd-drill` lifecycle check from
  `origin/dev` after PR `#2601` merged the stale-socket evidence packet.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
  Git state at capture time was branch `codex/vz-launchd-drill-evidence` at
  `origin/dev` merge commit `f2d9be986499eb1bfda36f566870a98e8dd90d0d` plus
  this evidence/backlog documentation update.
- Host identity: Apple silicon `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: helper build used `vz-helperctl.py build` outside the managed
  filesystem sandbox because Swift/Clang needed access to
  `~/.cache/clang/ModuleCache`. The drill signed the helper with
  `tools/macos-vz-helper/macos-vz-helper.entitlements` before bootstrap.
- Runtime paths: runtime root
  `/private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446`, unique
  LaunchAgent label
  `org.tldw.macos-vz-helper.drill.codex.launchd-drill-20260703-171446`, helper
  socket
  `/private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/helper.sock`,
  log directory
  `/private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/logs`,
  plist
  `/private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/org.tldw.macos-vz-helper.drill.codex.launchd-drill-20260703-171446.plist`,
  and artifact directory
  `/private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/artifacts`.
  Runtime, logs, and artifacts directories were owner-only mode `0700`.
- Commands:

  ```bash
  tools/macos-vz-helper/scripts/vz-helperctl.py build

  tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
    --helper /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-launchd-drill-evidence/tools/macos-vz-helper/.build/debug/macos-vz-helper \
    --socket /private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/helper.sock \
    --log-dir /private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/logs \
    --plist-output /private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446/org.tldw.macos-vz-helper.drill.codex.launchd-drill-20260703-171446.plist \
    --label org.tldw.macos-vz-helper.drill.codex.launchd-drill-20260703-171446 \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --write-plist \
    --create-dirs \
    --skip-smoke \
    --json
  ```

- Results: an initial diagnostic attempt passed a relative `--helper` path; the
  generated LaunchAgent plist preserved that relative `ProgramArguments` value,
  so launchd loaded and kicked the service but helper readiness failed with
  `helper_ping_failed`. The accepted prepared-host evidence reran the drill with
  an absolute helper path and passed with exit code `0`: preflight reported
  `launchd_service_absent`, helper signing passed, `launchd_bootstrap`,
  `launchd_status`, and `launchd_kickstart` passed, helper readiness passed,
  `protocol_version=1`, `helper_version=0.1.0`, and `launchd_bootout` passed.
- Cleanup: after the drill-owned bootout, an explicit follow-up
  `launchd status` returned exit code `1` with `launchd_status_failed=113`, and
  an extra bootout returned `No such process`, confirming the LaunchAgent was
  no longer loaded. Direct helper status showed no pid file and
  `process=helper_not_running` / `ping=helper_ping_failed`; the socket file
  remained as an inactive socket under the private runtime directory. That
  stale socket is isolated by the `0700` parent and is covered by the separate
  stale-socket recovery drill.
- Artifacts: retained under the artifact directory:
  `launchd-drill.json`, `launchd-status-after-drill.json`,
  `helper-status-after-launchd-bootout.json`, `launchd-bootout-after-drill.txt`,
  `runtime-stat.txt`, `paths.txt`, exit code files, and `artifact-list.txt`.
  Helper stdout/stderr were retained under the log directory and were empty,
  both SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, no real `vz_linux` VM smoke because this drill intentionally used
  `--skip-smoke`, no host reboot drill, and no manual boot/readiness fault
  injection.
- Blocking regressions: none observed for the selected manual launchd lifecycle
  drill. The relative-helper diagnostic attempt is recorded as an operator input
  issue for launchd plists, not as the accepted evidence result.
- Residual gaps: launchd-managed real VM smoke, host-reboot, and manual
  boot/readiness fault-injection evidence remain manual/operator-gated items.
  Broader unclassified helper crash recovery and long-term evidence retention
  remain separate follow-ups.
- Follow-up owner: `TASK-12137` records this evidence/update slice; repeat the
  launchd drill when launchd scaffolding, helper signing, or plist generation
  behavior changes.

### 2026-07-03: local-operator stale-socket drill on `codex/vz-stale-socket-evidence`

- Evidence source: local operator run on the same prepared Apple silicon macOS
  host, using the manual `vz-helperctl.py stale-socket-drill` lifecycle check
  from `origin/dev` after PR `#2418` merged.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
  Git state at capture time was branch `codex/vz-stale-socket-evidence` at
  `origin/dev` merge commit `c20013ecce7e3384ec5faa860434d6bdd76d5407` plus
  this evidence/backlog documentation update.
- Host identity: Apple silicon `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: helper build initially failed under the managed filesystem sandbox
  because Swift/Clang could not write `~/.cache/clang/ModuleCache`; the same
  `vz-helperctl.py build` command succeeded outside the sandbox. The helper was
  signed with `tools/macos-vz-helper/macos-vz-helper.entitlements`.
- Runtime paths: runtime root
  `/private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828`, helper
  socket
  `/private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.sock`,
  pid file
  `/private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.pid`,
  log directory
  `/private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/logs`, and
  artifact directory
  `/private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/artifacts`.
  Runtime, logs, and artifacts directories were owner-only mode `0700`.
- Commands:

  ```bash
  tools/macos-vz-helper/scripts/vz-helperctl.py build

  tools/macos-vz-helper/scripts/vz-helperctl.py sign \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements

  tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill \
    --helper tools/macos-vz-helper/.build/debug/macos-vz-helper \
    --socket /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.sock \
    --pid-file /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.pid \
    --log-dir /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/logs \
    --json

  tools/macos-vz-helper/scripts/vz-helperctl.py stop \
    --socket /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.sock \
    --pid-file /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828/helper.pid
  ```

- Results: the first sandbox-managed drill attempt failed before helper start
  with `helper_socket_create_failed` and `Operation not permitted` while
  creating the controlled inactive Unix socket. The accepted prepared-host
  evidence reran the same drill outside the managed sandbox and passed with
  exit code `0`: `stale_socket=ok`, `start=ok`, `after_socket` reported
  `helper_socket_present`, `after_pid_file` and `after_process` reported
  `helper_pid_running`, `after_ping=ok`, `after_protocol_version=1`,
  `after_helper_version=0.1.0`, and `stale_socket_drill=ok`.
- Cleanup: explicit `vz-helperctl.py stop` on the same socket and pid file
  returned exit code `0` with `helper_pid_stale`. Post-stop status reported
  `socket=helper_socket_absent`, `pid_file=ok`, `process=helper_not_running`,
  and `ping=helper_not_running`. The generic status command still exited `1`
  because this host has an unrelated default `launchd_plist_mismatch`; that row
  was not part of the direct-helper stale-socket drill.
- Artifacts: retained under the artifact directory:
  `stale-socket-drill.json`, `status-after-drill.json`,
  `status-after-stop.json`, `runtime-stat.txt`, `paths.txt`, `stop.txt`, exit
  code files, and `artifact-list.txt`. Helper stdout/stderr were retained under
  the log directory and were empty, both SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, no `vz_linux` VM smoke, no launchd validation, no host reboot drill, and
  no manual boot/readiness fault injection.
- Blocking regressions: none observed for the selected manual stale-socket
  lifecycle drill. The only failed attempt was attributable to Codex managed
  sandbox host restrictions around Unix socket creation, not helper behavior.
- Residual gaps: launchd, host-reboot, and manual boot/readiness
  fault-injection evidence remain manual/operator-gated items. Broader
  unclassified helper crash recovery and long-term evidence retention remain
  separate follow-ups.
- Follow-up owner: `TASK-12136` records this evidence/update slice; repeat the
  stale-socket evidence when helper socket-safety behavior or the operator
  drill command changes.

### 2026-06-20: local-operator repeat image-store clone smoke on `codex/vz-image-store-smoke-evidence`

- Evidence source: local operator run on the same prepared Apple silicon macOS
  host, using the image-store-backed smoke wrapper after the host was rebooted
  and the PR `#2414` evidence handoff work had merged.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
  Git state at capture time was branch `codex/vz-image-store-smoke-evidence`
  at `origin/dev` plus this evidence/backlog branch work.
- Host identity: Apple silicon `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: `/usr/bin/swift`, `/usr/bin/codesign`, and `/usr/bin/shasum` were
  available. SwiftPM built the helper in an escalated host command because the
  managed filesystem sandbox blocked the default Clang module cache under
  `~/.cache`; the rerun outside the sandbox completed.
- Source bundle: `/private/tmp/tldw-vz-bundle`, a symlink to
  `$HOME/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle`.
  Source bundle hashes were identical before and after the real smoke:
  `kernel` SHA-256
  `6dc5255afb8c7722896b860e50a892c1a1f0e774a18338dc259e19736f27a3ef`,
  `initrd` SHA-256
  `89ae29154c08e22d09714588bfa94e7ed5894316c89c819b84be62f4e213a054`,
  `manifest.json` SHA-256
  `a7b5dc7d9e4932e5d6c13c287263f6e49dca3e48fa08e191d760f5545f8e3c29`, and
  `rootfs.img` SHA-256
  `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`.
- Image-store disposable run bundle:
  `/private/tmp/tvz-e2e-25415/image-store/runs/host-smoke-25415/bundle`. The
  run bundle rootfs hash after execution was
  `b6809e38b69de1d5c2bf99398ed5eb34ab88e365c2e582cd0a8f06cc605f34f4`, while
  the source rootfs hash remained
  `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`.
  This repeat run proves the default smoke path absorbs VM writes in the
  disposable run bundle rather than mutating the canonical source bundle.
- Helper build/signing: helper binary
  `tools/macos-vz-helper/.build/debug/macos-vz-helper`; ad hoc `codesign`
  completed with `tools/macos-vz-helper/macos-vz-helper.entitlements`.
- Runtime paths: runtime root `/tmp/tvz-e2e-25415`; helper socket
  `/tmp/tvz-e2e-25415/helper.sock`; serial log directory
  `/tmp/tvz-e2e-25415/serial`; image-store root
  `/tmp/tvz-e2e-25415/image-store`; evidence directory
  `/tmp/tvz-e2e-25415/evidence`. Runtime, serial, image-store, run-bundle, and
  evidence directories were owner-only mode `0700`.
- Commands:

  ```bash
  tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
    --bundle /private/tmp/tldw-vz-bundle \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --python <repo>/.venv/bin/python \
    --skip-build
  ```

  The recorded runtime path used the then-current PID-based default directory
  form. PR review follow-up hardened the wrapper default to create a short
  random `mktemp -d /tmp/tvz-e2e.XXXXXX` directory before future captures.

- Results: helper daemon smoke passed `2 passed`; real `vz_linux` host smoke
  passed `3 passed, 11 deselected`. The selected real-host tests covered
  ephemeral execution, same-session VM reuse, and recovery diagnostics plus
  dry-run reconciliation repair planning.
- Failure drills: skipped; this evidence packet did not request
  `--include-failure-drills`.
- Launchd drill: skipped; this evidence packet did not request LaunchAgent
  validation.
- Stale socket drill: skipped; this evidence packet did not request the manual
  `stale-socket-drill`.
- Stuck boot/readiness drills: host-independent coverage remains represented by
  the portable helper/runner test suite; no manual prepared-host boot-fault
  injection was requested for this packet.
- Artifacts: retained under `/tmp/tvz-e2e-25415/evidence`:
  `host-smoke-evidence.json` size `4146`, `source-bundle-hashes-before.txt` size `327`,
  `source-bundle-hashes-after.txt` size `327`, `run-bundle-hashes.txt` size
  `327`, `runtime-paths.txt` size `982`, and `cleanup-status.txt` size `165`.
  Helper stdout/stderr files were retained and empty, both SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  Serial logs were retained as pointers only:
  `b63a1e13-11a8-437e-9229-cc150617bc4b.serial.log` SHA-256
  `8c57397172bf86430518a233174c35b3fd023f157f1df75f5e4c6e3df46a8dd6`,
  `bundle-smoke-vm.serial.log` SHA-256
  `bc1184945c91fdbdeb9bcc0c9781ffb8e4c2afc0853de133ae6d9a124d3819ec`, and
  `vz-linux-real-ephemeral.serial.log` SHA-256
  `f7721cfbd092baed491248531f4fdf5a6ac01903a69cbb2574e9f86b5d4dd2c4`.
- Cleanup: final exit code `0`; cleanup status `0`; helper pid `26434` was not
  running after cleanup; accepted socket `/tmp/tvz-e2e-25415/helper.sock` was
  absent after cleanup.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, no opt-in failure drills, no launchd validation, no stale socket drill,
  no manual host reboot drill, and no manual boot/readiness fault injection.
- Blocking regressions: none observed for the selected image-store clone smoke
  coverage. A preceding setup attempt using a long `${TMPDIR}`-style socket path
  failed before real execution with helper stderr `socketPathTooLong`; the
  wrapper and examples now prefer short `/tmp/tvz-*` runtime paths for helper
  sockets.
- Residual gaps: launchd, stale-socket, host-reboot, and manual boot/readiness
  fault-injection evidence remain manual/operator-gated items. Broader
  unclassified helper crash recovery and long-term evidence retention remain
  separate follow-ups.
- Follow-up owner: `TASK-2394` records this evidence/update slice; future
  focused tasks should cover manual drill evidence only when maintainers
  intentionally request those disruptive checks.

### 2026-06-16: local-operator disposable image-store clone smoke on `codex/vz-smoke-clone-evidence`@`ab1c55c67c`

- Evidence source: local operator run on the same prepared Apple silicon macOS
  host after PR `#2370` merged the disposable image-store smoke-clone path.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
  Git state at capture time was branch `codex/vz-smoke-clone-evidence`,
  commit `ab1c55c67c852040a5162308ef987ea124937baa`, with only the new
  Backlog evidence task untracked before this evidence doc was edited.
- Host identity: Apple M4 Pro, `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: SwiftPM available with Swift `6.1.2`; `xcrun --find codesign`
  returned `/usr/bin/codesign`; `/usr/bin/codesign --version` is not a valid
  version probe on this host, but `codesign` signed and verified the helper;
  Virtualization.framework was exercised by the real helper and `vz_linux`
  smoke.
- Source bundle:
  `/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle`.
  Source bundle hashes and stat output were identical before and after the
  smoke run. The unchanged source hashes were `kernel` SHA-256
  `6dc5255afb8c7722896b860e50a892c1a1f0e774a18338dc259e19736f27a3ef`,
  `initrd` SHA-256
  `89ae29154c08e22d09714588bfa94e7ed5894316c89c819b84be62f4e213a054`,
  `rootfs.img` SHA-256
  `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`,
  and `manifest.json` SHA-256
  `a7b5dc7d9e4932e5d6c13c287263f6e49dca3e48fa08e191d760f5545f8e3c29`.
- Image-store disposable run bundle:
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/image-store/runs/clone-evidence-20260616-130222/bundle`.
  The run manifest used template `vz_linux:host-smoke-source`, run id
  `clone-evidence-20260616-130222`, and `mode=clone` entries for `kernel`,
  `rootfs.img`, and `initrd` from the source bundle into the run bundle.
  The run bundle rootfs hash after execution was
  `ba04818c7f99b8742481b184bcb98eabbcfcdd476760bf13926be82f3cf7bb7c`,
  while the source rootfs hash remained
  `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`.
  This proves the smoke path absorbed VM writes in the disposable run bundle
  instead of mutating the canonical source bundle.
- Helper build/signing: helper built from this worktree at
  `tools/macos-vz-helper/.build/debug/macos-vz-helper`; ad hoc `codesign`
  completed with `tools/macos-vz-helper/macos-vz-helper.entitlements`; signed
  entitlement check showed `com.apple.security.virtualization=true`; helper
  signature CDHash `4e060df093d6f7dd3b5f87a7ee43ad8e81e9ed35`.
- Runtime paths: artifact root
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222`;
  helper socket
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/helper.sock`;
  serial log directory
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/serial`;
  image-store root
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/image-store`.
  The artifact root, image-store root, run directory, run bundle, and serial
  directory were owner-only mode `0700`. The `runs` and `templates` parent
  directories were mode `0755`, but they were nested under the owner-only
  image-store root.
- Commands:

  ```bash
  ./tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
    --dry-run \
    --bundle "/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle" \
    --socket "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/helper.sock" \
    --serial-log-dir "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/serial" \
    --image-store-root "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/image-store" \
    --smoke-run-id clone-evidence-20260616-130222 \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --python /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python

  ./tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
    --bundle "/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle" \
    --socket "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/helper.sock" \
    --serial-log-dir "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/serial" \
    --image-store-root "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/image-store" \
    --smoke-run-id clone-evidence-20260616-130222 \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --python /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python
  ```

- Results: the dry-run expansion used the disposable run bundle for
  `TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH` and
  `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE`. The real run completed
  `swift build`, signed the helper, ran helper daemon smoke `2 passed`, and ran
  real `vz_linux` host smoke `3 passed, 11 deselected`. The selected real-host
  tests included `test_vz_linux_real_ephemeral_run_smoke`,
  `test_vz_linux_real_session_reuse_smoke`, and
  `test_vz_linux_real_recovery_diagnostics_dry_run_smoke`.
- Failure drills: skipped; this evidence packet did not request
  `--include-failure-drills` because those opt-in drills were already captured
  by the preceding failure-drill packet.
- Launchd drill: skipped; this evidence packet did not request LaunchAgent
  validation.
- Stale socket drill: skipped; this evidence packet did not request the manual
  `stale-socket-drill`.
- Stuck boot/readiness drills: host-independent coverage remains represented by
  the portable test suite; no manual prepared-host boot-fault injection was
  requested for this packet.
- Artifacts: `metadata.env`, `smoke-dry-run.log`, `smoke-run.log`,
  `source-hashes-before.txt`, `source-hashes-after.txt`,
  `source-stat-before.txt`, `source-stat-after.txt`,
  `run-bundle-hashes.txt`, `run-bundle-stat.txt`, `image-store-manifests.txt`,
  and `source-hash-diff.txt` retained under the artifact root. Helper
  stdout/stderr were retained and empty, both SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  Serial logs were retained as pointers only:
  `447ffdb6-6ae3-4782-a75e-6fa016f6c819.serial.log` SHA-256
  `48b271960cfbb9611e58c546666c2cfd5280378d8632682f13d63d71323bc85f`,
  `bundle-smoke-vm.serial.log` SHA-256
  `ab44a909ba9c2c6dc0d10e9268d8d145f9981d2c6d26defd747cb548daf6c712`,
  and `vz-linux-real-ephemeral.serial.log` SHA-256
  `6d431955fe9535fe33af62354fdbceae3021fa0b78199d0b24f6e3ccb1031b80`.
  The accepted helper socket was absent after cleanup. The recorded helper pid
  `44279` was no longer running; a separate helper from an earlier worktree was
  still running and was not part of this evidence packet.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, no opt-in failure drills in this packet, launchd validation not
  requested, stale socket drill not requested, no manual host reboot drill, and
  no manual boot/readiness fault injection.
- Blocking regressions: none observed for the selected disposable-clone smoke
  coverage.
- Residual gaps: launchd drill, stale socket drill, host reboot pre/post drill,
  manual stuck boot/readiness fault injection, guest-agent mismatch coverage
  beyond host-independent tests, broader helper crash classes beyond helper
  termination/restart, and automatic long-term retention of local evidence
  artifacts remain separate manual/operator-gated or implementation follow-ups.
- Follow-up owner: issue `#1442` and future focused Backlog tasks for remaining
  manually skipped drills and evidence-retention automation.

### 2026-06-16: local-operator failure drills on `codex/vz-failure-drill-evidence`@`e17d8cbf07`

- Evidence source: local operator run on a prepared Apple silicon macOS host
  with manual `--include-failure-drills` enabled.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
  Git state at capture time was branch `codex/vz-failure-drill-evidence`,
  commit `e17d8cbf07d3f7753713a34bf253d98987757309`, with only the new
  Backlog task untracked before this evidence doc was edited.
- Host identity: Apple M4 Pro, `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: SwiftPM available at `/usr/bin/swift` with Swift `6.1.2`; Xcode
  command line tools at `/Library/Developer/CommandLineTools`; `xcrun` and
  `/usr/bin/codesign` available; Virtualization.framework exercised by the real
  helper and `vz_linux` smoke.
- Bundle/template:
  `/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle`;
  bundle manifest reports `bundle_version=1`, `boot_mode=bundle`,
  `guest_agent_path=/usr/local/bin/tldw-agent-guest`, workspace mount tag
  `workspace`, and vsock port `1024`.
- Bundle hashes recorded after the run:
  `kernel` SHA-256
  `6dc5255afb8c7722896b860e50a892c1a1f0e774a18338dc259e19736f27a3ef`;
  `initrd` SHA-256
  `89ae29154c08e22d09714588bfa94e7ed5894316c89c819b84be62f4e213a054`;
  `rootfs.img` SHA-256
  `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`;
  `manifest.json` SHA-256
  `a7b5dc7d9e4932e5d6c13c287263f6e49dca3e48fa08e191d760f5545f8e3c29`.
  The direct-bundle smoke path updated `rootfs.img` mtime during execution, so
  future evidence should prefer a disposable clone or a reset source bundle when
  immutable-source hashes matter.
- Helper build/signing: helper built from this worktree at
  `tools/macos-vz-helper/.build/debug/macos-vz-helper`; ad hoc `codesign`
  completed with `tools/macos-vz-helper/macos-vz-helper.entitlements`; signed
  entitlement check showed `com.apple.security.virtualization=true`; helper
  signature CDHash `d27ce163c4ed74e65ca888132d71eb2b62f92dfa`.
- Runtime paths: artifact root
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-failure-evidence-20260616-070906`;
  helper socket
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-failure-evidence-20260616-070906/helper.sock`;
  serial log directory
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-failure-evidence-20260616-070906/serial`;
  runtime and serial directories were owner-only mode `0700`.
- Commands:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
    tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
    --bundle "/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle" \
    --socket "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-failure-evidence-20260616-070906/helper.sock" \
    --serial-log-dir "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-failure-evidence-20260616-070906/serial" \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --python /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
    --include-failure-drills
  ```

- Results: dry-run command expansion ended with `smoke: ok`; real run completed
  `swift build`, signed the helper, ran helper daemon smoke `2 passed`, ran
  real `vz_linux` host smoke `3 passed, 11 deselected`, ran manual failure
  drills `2 passed, 12 deselected`, and ended with `smoke: ok`.
- Failure drills: passed. The selected tests were
  `test_vz_linux_real_session_recreates_vm_after_helper_termination` and
  `test_vz_linux_real_session_recreates_vm_after_helper_restart`.
- Launchd drill: skipped; this evidence packet did not request LaunchAgent
  validation.
- Stale socket drill: skipped; this evidence packet did not request the manual
  `stale-socket-drill`.
- Stuck boot/readiness drills: host-independent coverage remains represented by
  the portable test suite; no manual prepared-host boot-fault injection was
  requested for this packet.
- Artifacts: `smoke-failure-dry-run.log` and `smoke-failure-run.log` retained
  under the artifact root; serial logs retained as pointers only:
  `bundle-smoke-vm.serial.log` SHA-256
  `c7b63ac1f061bf6aad164cbc85548326e0e7c67944ffce7eed94d1bb205a6b37`,
  `vz-linux-real-ephemeral.serial.log` SHA-256
  `37e87c4b05a52e74598daefb7256e87948a2eb67ca2f0bffb455a71cafba7374`,
  `49294d34-845b-4cd1-98e0-802b63d0baa3.serial.log` SHA-256
  `f8279c31617c0201720fee9392918ced9a7e4f3762756f87ff67d3eb65e8a12b`,
  `60b5e1c6-2334-4120-8ea3-52843e675399.serial.log` SHA-256
  `776747556169bdfcf9c268e40b2cef094bd60de0b9649c77fb3ef5c5fc1cc207`,
  `754da451-81b6-4219-9b7b-1a996bbdcec7.serial.log` SHA-256
  `0f799172ff2db239114156965faa9b608b156006b1d9fe6868346b072faedc81`,
  `9bce5857-0b03-4441-9bfd-63f0a7788d28.serial.log` SHA-256
  `01214bc50404c8b4cc9ec4794ab0c77ae66960d4a1a32a1fa615306e4081e592`,
  and `da825230-b17f-4915-836d-165349aa05b6.serial.log` SHA-256
  `3ac85a71d07e2dee83ebb126b66101021fa3d5e6f2bb8e3e86f233de56130ad3`.
  Helper stdout/stderr files, including restart helper stdout/stderr files,
  were present and empty. The helper socket was removed after cleanup, and the
  recorded helper pid `66734` was no longer running.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, launchd validation not requested, stale socket drill not requested, no
  manual host reboot drill, and no manual boot/readiness fault injection.
- Blocking regressions: none observed for the selected default smoke and
  failure-drill coverage.
- Residual gaps: launchd drill, stale socket drill, host reboot pre/post drill,
  manual stuck boot/readiness fault injection, guest-agent mismatch coverage
  beyond host-independent tests, broader helper crash classes beyond helper
  termination/restart, and disposable-clone protection for direct-bundle smoke
  evidence remain separate manual/operator-gated or implementation follow-ups.
- Follow-up owner: issue `#1442` and future focused Backlog tasks for remaining
  manually skipped drills and bundle immutability hardening.

### 2026-06-16: local-operator on `codex/vz-prepared-host-evidence-packet`@`ce6276da23`

- Evidence source: local operator run on a prepared Apple silicon macOS host.
- Operator or workflow run: local shell run; no GitHub Actions workflow URL.
- Host identity: Apple M4 Pro, `arm64`, macOS 15.6 build `24G84`, Darwin
  `24.6.0`; local developer machine rather than a dedicated CI runner.
- Host prep: SwiftPM available at `/usr/bin/swift` with Swift `6.1.2`; Xcode
  command line tools at `/Library/Developer/CommandLineTools`; macOS SDK path
  `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; `xcrun` and
  `/usr/bin/codesign` available; Virtualization.framework exercised by the real
  helper and `vz_linux` smoke.
- Bundle/template:
  `/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle`;
  canonical bundle manifest reports `bundle_version=1`, `boot_mode=bundle`,
  `guest_agent_path=/usr/local/bin/tldw-agent-guest`, workspace mount tag
  `workspace`, and vsock port `1024`. Build provenance file reports
  `artifact_kind=canonical_bundle`, Debian `bookworm`, profile `minimal`,
  architecture `arm64`, kernel package `linux-image-arm64`.
- Bundle hashes:
  `kernel` SHA-256
  `6dc5255afb8c7722896b860e50a892c1a1f0e774a18338dc259e19736f27a3ef`;
  `initrd` SHA-256
  `89ae29154c08e22d09714588bfa94e7ed5894316c89c819b84be62f4e213a054`;
  `rootfs.img` SHA-256
  `5cf0e2278e8ec080b46ff496417d2b503ac5c55d1913795633a420b3973ff639`;
  `manifest.json` SHA-256
  `a7b5dc7d9e4932e5d6c13c287263f6e49dca3e48fa08e191d760f5545f8e3c29`.
- Helper build/signing: helper built from this worktree at
  `tools/macos-vz-helper/.build/debug/macos-vz-helper`; ad hoc `codesign`
  completed with `tools/macos-vz-helper/macos-vz-helper.entitlements`; signed
  entitlement check showed `com.apple.security.virtualization=true`; helper
  signature CDHash `80016eb2a537d71efaa51da8a82ee712daae6fa5`.
- Runtime paths: artifact root
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631`;
  helper socket
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631/helper.sock`;
  serial log directory
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631/serial`;
  runtime and serial directories were owner-only mode `0700`.
- Commands:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
    tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
    --bundle "/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle" \
    --socket "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631/helper.sock" \
    --serial-log-dir "/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631/serial" \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
    --python /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python
  ```

- Results: `swift build` completed, helper was signed, helper daemon smoke ran
  `2 passed`, and real `vz_linux` host smoke ran `3 passed, 11 deselected`.
  The selected real-host tests were
  `test_vz_linux_real_ephemeral_run_smoke`,
  `test_vz_linux_real_session_reuse_smoke`, and
  `test_vz_linux_real_recovery_diagnostics_dry_run_smoke`; the wrapper ended
  with `smoke: ok`.
- Failure drills: skipped; default smoke did not pass `--include-failure-drills`.
- Launchd drill: skipped; this evidence packet did not request LaunchAgent
  validation.
- Stale socket drill: skipped; this evidence packet did not request the manual
  `stale-socket-drill`.
- Stuck boot/readiness drills: host-independent coverage remains represented by
  the portable test suite; no manual prepared-host boot-fault injection was
  requested for this packet.
- Artifacts: `smoke-dry-run.log` and `smoke-run.log` retained under the
  artifact root; serial logs retained as pointers only:
  `bundle-smoke-vm.serial.log` SHA-256
  `07ded39bf985377a11776ea903d9f5cdb21f5ca10e9c5fea534f2491e659944d`,
  `vz-linux-real-ephemeral.serial.log` SHA-256
  `1ed98d62d51d61453523c76569dc64395baa031c15b199bcd04bbcbf2b4e27d1`,
  and `dafc6190-c98c-4209-bc37-110958c029cc.serial.log` SHA-256
  `c6e510fc08ce7556ebac4c4c6541e625a240a3945ddc32efbabb75677de4f3fc`.
  Helper stdout/stderr files were present and empty. The helper socket was
  removed after cleanup, and the recorded helper pid was no longer running.
- Expected skips: no PR workflow, no nightly schedule, no self-hosted runner
  URL, failure drills not requested, launchd validation not requested, stale
  socket drill not requested, no manual host reboot drill, and no manual
  boot/readiness fault injection.
- Blocking regressions: none observed for the default prepared-host smoke.
- Residual gaps: failure drills, launchd drill, stale socket drill, host reboot
  pre/post drill, manual stuck boot/readiness fault injection, guest-agent
  mismatch coverage beyond host-independent tests, and broader helper crash
  classes remain separate manual/operator-gated evidence items.
- Follow-up owner: issue `#1442` and future focused Backlog tasks for each
  manually skipped drill when maintainers choose to collect that evidence.

### Template

```markdown
### YYYY-MM-DD: <source> on <branch>@<sha>

- Evidence source:
- Operator or workflow run:
- Host identity:
- Host prep:
- Bundle/template:
- Helper build/signing:
- Runtime paths:
- Commands:
- Results:
- Failure drills:
- Launchd drill:
- Artifacts:
- Expected skips:
- Blocking regressions:
- Residual gaps:
- Follow-up owner:
```

## Current Residual Gaps

| Gap | Current status | Next action |
| --- | --- | --- |
| Prepared-host default smoke evidence | Recorded locally on 2026-06-16 with helper daemon smoke, real ephemeral execution, same-session reuse, and recovery diagnostics/dry-run repair smoke passing. | Repeat periodically through a trusted local or host-gated run and add newer evidence packets as needed. |
| Failure-drill evidence | Recorded locally on 2026-06-16 with drill-owned stale VM replacement and smoke-owned helper restart drill passing. | Repeat when runtime/helper recovery behavior changes; keep manual opt-in only. |
| Launchd-drill evidence | Recorded real VM smoke on 2026-09-13 (3 tests, no skips), plus a separate live-session restart test (1 passed, no skips/errors) proving changed helper generation, stale-VM replacement, replacement reuse, and cleanup. The latter exposed and fixed guest capped-output loss before acceptance. Durable artifacts and failed attempts are recorded above. | Repeat both bounded drills when helper lifecycle, signing, guest transport, or image preparation changes. Keep launchd validation explicitly operator-requested; host reboot remains separate. |
| Host reboot recovery | Manual `host-reboot-drill pre/post` procedure only and out of scheduled CI. | Record results when a maintainer explicitly runs the reboot drill on a prepared host that can tolerate disruptive reboot testing and preserve logs. |
| Stuck boot/readiness | Host-independent helper and runner coverage verifies boot-driver and guest-readiness failure cleanup. Manual real acknowledged-handshake readiness withholding, timeout cleanup, and healthy session reuse passed twice on 2026-09-13 with a normal-ready negative control. A separate no-agent-connection startup timeout and recovery drill passed on 2026-09-25. The default smoke still does not inject faults. | Repeat the opt-in drills when the handshake or lifecycle changes. Kernel boot hangs remain separate; preserve stable reasons and artifact pointers rather than exposing raw serial logs. |
| Guest-agent mismatch | The checked-in eight-case workflow on 2026-09-14 proved missing-`exec`, guest wire protocol and advertised workspace mismatch rejection, cleanup, healthy recovery and session reuse. The ten-case workflow on 2026-09-25 added real missing-agent startup absence with a normal-agent negative control. Not part of default smoke. | Repeat the opted-in workflow when handshake or admission changes. Actual mount/path-escape isolation remains a separate evidence case under the lifecycle-drill contract. |
| Stale socket handling | Manual stale-socket prepared-host evidence was recorded locally on 2026-07-03 with controlled inactive socket recovery, helper start/status verification, and explicit stop cleanup passing. | Repeat when helper socket-safety behavior or the operator drill command changes; keep it manual-only and out of PR/push/scheduled destructive triggers. |
| Direct-bundle smoke mutability | Closed for the default smoke path by the 2026-06-16 disposable-clone evidence and repeated on 2026-06-20 after host reboot: source bundle hashes stayed identical before/after while the disposable run bundle rootfs hash changed after execution. | Repeat periodically when the smoke wrapper, image-store materializer, or helper VM write path changes. |

## Recording Guidance

For a local prepared-host run, prefer the managed helper wrapper:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements
```

For a manual stale-socket check, use an isolated private runtime directory:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-stale-socket.XXXXXX")"
chmod 700 "$runtime_dir"
trap 'rm -rf "$runtime_dir"' EXIT

tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill \
  --helper tools/macos-vz-helper/.build/debug/macos-vz-helper \
  --socket "$runtime_dir/helper.sock" \
  --pid-file "$runtime_dir/helper.pid" \
  --log-dir "$runtime_dir/logs"
```

Record the runtime directory mode, socket path result, command output, helper
stdout/stderr paths, and whether this was skipped because no maintainer
requested the manual drill.

For a lower-level run, use a short private runtime directory and cleanup trap.
Avoid long `${TMPDIR}`-based socket paths on macOS because AF_UNIX sockets are
length-limited:

```bash
runtime_dir="$(mktemp -d "/tmp/tvz-e2e.XXXXXX")"
chmod 700 "$runtime_dir"
trap 'rm -rf "$runtime_dir"' EXIT

tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  --bundle /path/to/canonical/bundle \
  --socket "$runtime_dir/helper.sock" \
  --serial-log-dir "$runtime_dir/serial" \
  --entitlements /path/to/helper.entitlements
```

The lower-level smoke script treats `--bundle` as the source bundle and creates
a disposable bundle under `$runtime_dir/image-store/runs/<run-id>/bundle` by
default. Record source bundle hashes before and after the run, and record run
bundle hashes separately when retaining the image-store artifacts.

For host-gated CI, record the workflow run URL, runner labels, branch/ref, input
values, artifact names, and any expected skips. The workflow must remain
manual/nightly only and must not be promoted into normal PR-triggered CI.
