# VZ Linux Prepared-Host Evidence Tracker

**Status:** Active tracker for prepared Apple silicon `vz_linux` acceptance evidence.
**Scope:** Real `vz_linux` execution evidence from manual operator runs or the host-gated workflow on trusted refs.
**Policy:** `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`.
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
| Stuck boot/readiness drills | Host-independent helper/runner tests prove registry/session cleanup; any manual prepared-host drill records stable reason codes and artifact pointers without exposing raw serial logs. | Portable coverage only |
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
| Launchd-drill evidence | Manual opt-in only. | Record results only when a runner is intentionally configured for LaunchAgent validation. |
| Host reboot recovery | Manual `host-reboot-drill pre/post` procedure only and out of scheduled CI. | Record results when a maintainer explicitly runs the reboot drill on a prepared host that can tolerate disruptive reboot testing and preserve logs. |
| Stuck boot/readiness | Host-independent helper and runner coverage verifies boot-driver failure cleanup, guest-readiness failure cleanup, and no reusable session state after create failure. The default prepared-host smoke still does not inject real boot faults. | Record manual prepared-host evidence only after a separate reviewed fault-injection plan; diagnostics/evidence should report stable reason codes and artifact pointers, not raw serial log contents. |
| Guest-agent mismatch | Not covered by the default smoke. | Use `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md` to guide narrow tests or diagnostics checks before considering automated coverage. |
| Stale socket handling | `tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill` provides a manual operator check for safe inactive socket recovery. | Record prepared-host evidence when a maintainer intentionally runs the drill; keep it manual-only and out of PR/push/scheduled destructive triggers. |
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
