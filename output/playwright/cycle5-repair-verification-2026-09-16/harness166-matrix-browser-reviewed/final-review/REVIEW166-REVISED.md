# UAT166 revised wrapper — independent review

## Verdict

**CLEAR for the bounded, trusted-operator browser adapter.** The two original command/configuration isolation findings are resolved on the frozen revision. No additional blocking defect was found. This is permission to integrate the reviewed private harness, not authorization or evidence of fresh-matrix/browser execution.

The reviewed author manifest is `.tmp/uat-matrix-browser-20260917/revised-source-manifest.json`, SHA256 `7c53dd12ec1f40a77ec1a429a4dd789ade5d093226bcf39a70bc2daa65a2607c`. All four current files equal their frozen snapshots. Wrapper SHA256: `99be0dfaa78ff38c7ff2366ca9a8516b343b6d8ffdfb225126698c0347d49381`.

## Original findings and correction

1. **Global/external commands:** the original five-counterexample file remains unchanged. Global daemon commands, external attachment, custom profile/config flags, persistent contexts, and session overrides are rejected before spawn. The command-specific option allowlist also rejects malformed short/session-like arguments while allowing ordinary negative numeric mouse movement. The installed CLI source confirms the rejected global and attach commands have broader effects than the selected session. These are actual input-boundary controls, not a reliance on the session prefix alone.
2. **Inherited/configured browser attachment:** the child now receives the launcher's existing small base environment plus `NO_UPDATE_NOTIFIER=1`. Playwright attachment/profile/storage/config overrides and `NODE_OPTIONS` are absent. The installed resolver reads the current working directory's `.playwright/cli.config.json` and the home global file; the wrapper rejects both before spawn. Its working directory is the validated source root. The current installed resolver's read-only positive control selected an isolated context without external endpoint, custom profile, or config. No home/config file was changed.

The existing profile/preparation/initialization identity checks, known-secret redaction, private raw output, snapshot path confinement, exclusive evidence filenames, and failure-code behavior remain covered. The CLI can group daemon registry entries by a discovered workspace, but the wrapper supplies its fixed run/cell session and denies global commands; this does not introduce a separate configuration-search path in the inspected resolver.

## Fresh verification

- `node --test .tmp/uat-matrix-browser-20260917/browser-wrapper.test.mjs .tmp/uat166-browser-independent-20260917/isolation-counterexamples.test.mjs`: **33 passed, 0 failed, 0 skipped**, 579.866291 ms. This includes the original five previously failing controls.
- Node syntax checks: wrapper, author tests, and original reviewer counterexamples all passed.
- All **14** reviewed launcher files and their snapshots remain equal to the role125 manifest, SHA256 `76b49014e0aee3eea3cc6b3c406e7099e4e02a626956384205454be8dffb450a`.
- Installed CLI source was inspected read-only; the bounded excerpts and complete file hashes are recorded in `installed-source-evidence.json`. `source-verification.json` records the exact four-plus-fourteen file/snapshot comparisons.

## Limits

No actual CLI entry point, browser, runtime, application profile, database, archive/dependency preparation, or model was started. Temporary synthetic profiles and the harmless test subprocess are test fixtures only. No independent native acceptance is claimed. The full matrix gate remains external and closed until the parent clears the outstanding repairs.

This is a trusted-operator adapter, not a sandbox for hostile scripts, malicious concurrent filesystem changes, or arbitrary output filenames. The intentionally available `eval`/`run-code` commands and known-secret redaction do not make arbitrary unknown secrets or screenshots publishable. Screenshots and retained evidence still require review. No global host/home isolation or raw operating-system isolation is claimed.

The author's two JavaScript Bandit parse errors are a tool applicability limitation, not JavaScript security coverage. This review uses source inspection and the synthetic boundary tests; it does not claim a clean JavaScript security scan.

Only this private review packet was written. No product, wrapper, test, task, tracker, git, runtime, or browser state was edited.
