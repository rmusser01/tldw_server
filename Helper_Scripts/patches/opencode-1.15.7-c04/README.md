# OpenCode ACP completion-order repair used by C04

This optional source patch makes the OpenCode dependency used for C04 reproducible. It is not installed automatically and does not change other ACP agents. Ordinary upstream OpenCode 1.15.7 has not passed the C04 orchestration workflow with this repair absent.

Upstream repository: <https://github.com/anomalyco/opencode>. Exact source commit: `5451227deb5a502afe44e1664e81df7ecb208ed1` (v1.15.7). Upstream source is MIT licensed; the notice is included in `LICENSE.opencode`.

The patch waits until the exact current assistant message's final native events have been forwarded before returning the ACP prompt result. It handles event-before-HTTP-result ordering, rejects foreign/stale completion, and cleans up on failure, cancellation and stream termination. It does not synthesize answers, replay late output or extend the server's prompt deadline. Regression tests accompany the implementation. The dependency edit pins `ghostty-web` to its already-locked commit `20bd361` rather than moving `main`.

## Reproduce in a separate checkout

Use Bun **1.3.14**. These commands download source and build dependencies; run them only in an intentionally selected build environment. Keep the server's runtime, global OpenCode and model files separate. Set `C04_PATCH` to the absolute path of this directory's `acp-final-event-order.patch` before changing directories.

```sh
git clone https://github.com/anomalyco/opencode.git opencode-c04
cd opencode-c04
git checkout --detach 5451227deb5a502afe44e1664e81df7ecb208ed1
git apply --check --index --unidiff-zero "$C04_PATCH"
git apply --index --unidiff-zero "$C04_PATCH"
bun install --frozen-lockfile
cd packages/opencode
bun test --timeout 30000 test/acp/event-subscription.test.ts test/acp/agent-interface.test.ts
OPENCODE_VERSION=1.15.7-c04.1 bun run script/build.ts --single --skip-install --skip-embed-web-ui
```

The zero-context patch uses `--index` to require the pinned original index/worktree before applying.

The single-target binary is under `packages/opencode/dist/opencode-<platform>-<architecture>/bin/opencode`. Check its `--version`, then select its absolute path through the normal ACP runner configuration. Do not replace the global binary implicitly. Server-side ownership, permission, cancellation and strict completion validation remain required.

## Qualification limits

The original isolated macOS arm64 build passed the selected C04 live workflows, and its two ACP test files passed 25 tests. Its package typecheck retained the upstream `GlobalBusEmitter.emit` TS2416 error; a complete typecheck pass is not claimed. Those historical results do not qualify another platform or a new build. This PR's publication check applies the patch to the exact pinned four original files and verifies byte equality with the tested source; it does not rerun dependency installation, compilation or live UAT.

Before release, reproduce the build in the target environment and run the listed tests plus the server's affected ACP checks. No prebuilt binary, credentials, model assets or runtime profile is included here. Remove this patch only after a supported upstream version contains and validates the same completion-order contract.
