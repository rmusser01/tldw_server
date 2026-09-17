# Independent review — TASK13260.166 matrix browser wrapper

**Verdict: changes requested before actual matrix browser use.** This reviewer did not author the wrapper. Review is bound to source manifest `fa62794c6f2f68a52dabc4029d2256441a38c167b3e789b176f90683b8f484a7`, wrapper SHA `334c34e65574bd484ddd9af8ea6424e2b2d921f42e9b28770676d54695e79f1f`. All four source/doc snapshots and all fourteen existing role125 launcher files/snapshots match their respective manifests.

## Findings

### 1. P1 — session argument does not confine accepted commands

`matrix-browser.mjs:47,54` rejects session overrides but forwards all other commands/options. Four review-only counterexamples show `kill-all`, `attach http://synthetic.invalid:9222`, `open --profile=/synthetic/foreign-profile`, and `open --config=/synthetic/foreign-config.json` reach the spawn boundary and report success.

This is more than a hypothetical fake-CLI interpretation. The installed CLI's `tools/cli-client/program.js:124–127,312–345` implements `kill-all` by enumerating matching daemon processes, without consulting the selected session. Its `attach` branch at137–157 accepts external endpoints, and `session.js:141–153` forwards arbitrary profile/config/attachment options to the daemon. A recorded session name alone does not establish a fresh browser context.

**Minimal correction:** a small explicit command/options policy for matrix operations. Permit the actual ordinary navigation, interaction, inspection and evidence commands needed by the protocol, plus scoped open/close. Reject global daemon/install actions, attach, external endpoint/extension selection, arbitrary user profiles/config files and session overrides. Keep trusted `run-code`/evaluation use within the existing authorized test workflow if needed; this is a trusted operator adapter, not an arbitrary-code sandbox. Validate recognized options rather than building a generic command parser/framework.

### 2. P1 — host environment and global configuration can select an existing browser

`matrix-browser.mjs:54` omits `env`, so the real child inherits host settings. A fifth fake-spawn counterexample proves a synthetic `PLAYWRIGHT_MCP_CDP_ENDPOINT` is inherited. Read-only calls to the **actual installed** `tools.resolveCLIConfigForCLI` prove that this setting selects that endpoint with `browser.isolated=false`.

There is an additional default-config seam: installed `coreBundle.js:72774–72820` loads the user's global `.playwright/cli.config.json` independently of an explicit local config and merges both before deciding isolation. A second actual resolver control, using only temporary synthetic files, proves an explicit empty local config still inherits the global CDP endpoint. Therefore environment cleanup or `--config={}` alone is insufficient.

**Minimal correction:** give the child a reviewed explicit environment (the existing launcher's small base-environment helper is a useful starting point), dropping host Playwright connection/profile/config/storage-state/init-script/output overrides and irrelevant Node/test overrides. Establish an owned browser home/config boundary, or fail closed on unowned implicit global configuration; verify the installed resolver sees no foreign global config. Keep browser binary/dependency reuse explicit. Disable CLI update checks for repeatability. No new sandbox framework or browser management service is needed. The final synthetic controls should cover inherited CDP, user-data-dir, storage-state and local/global config, not only a single variable.

## Passing boundaries

- Profile/run/cell identity, expected runtime and credential locations, complete preparation and matching successful initialization hash agree with the current launcher contract. Current launcher prevents preparation from rebinding a source root. Wrapper validates a real owned archive root and runs there; it does not invoke prepare/startup or silently decide the full-matrix gate.
- Known API/JWT/hash/account/provider/PG password strings and their tested JSON/URL encodings are redacted from displayed text. Raw stdout/stderr is exclusive mode0600 under the selected mode0700 evidence directory. Repeated invocations have distinct evidence paths.
- Snapshot copies require realpath containment inside the selected archive's `.playwright-cli`; the tested symlink escape fails closed. Copied text is redacted and mode0600. Private raw output intentionally remains raw; neither raw files nor original CLI artifacts should be published. Screenshots still require visual review. This does not certify redaction of unknown or arbitrary transformed secrets.
- Ordinary CLI nonzero status is retained; subprocess timeout/signal/infrastructure errors suppress partial output. Safe setup errors omit malformed credential contents.
- No other concrete source finding identified within this bounded review. Additional command/config guards need a fresh review before approval.

## Executed checks

1. `node --test .tmp/uat-matrix-browser-20260917/browser-wrapper.test.mjs` — **16 PASS, 0 skipped**. Retained `author-suite-independent.log`.
2. `node --check` wrapper and author test — both exit0.
3. `node --test .tmp/uat166-browser-independent-20260917/isolation-counterexamples.test.mjs` — **5 expected FAIL, 0 skipped**, retained exact source and `isolation-counterexamples-red.log`. It uses temporary synthetic profiles and fake spawn only; the installed CLI is never invoked.
4. `node --test .tmp/uat166-browser-independent-20260917/installed-config-resolution.test.mjs` — **2 PASS, 0 skipped**, retained log. Calls only the installed library's configuration resolver with temporary synthetic configuration and explicit synthetic environment; no CLI program, browser, daemon, network, app, DB or model.
5. `hash-checks.json` — **4/4 new and 14/14 existing source/snapshot hashes match**. `installed-cli-source-excerpts.txt` retains exact relevant installed source lines and hashes.

Bandit is a Python analyzer; the author's two JavaScript parse limitations are correctly disclosed, not security clearance. Independent review does not turn those into passing JS security coverage.

## Limits / unchanged state

No actual browser or CLI program, application runtime, matrix preparation, dependency copy, native profile/database/account, or inference was executed. No wrapper/product/task/git edits were made. Only this separate review packet and disposable synthetic test files were created. No host credential file, session header or token was read. Temporary synthetic files are cleaned by the tests. Full UAT remains gated.
