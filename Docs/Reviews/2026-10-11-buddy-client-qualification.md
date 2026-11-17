# Buddy client qualification — 2026-10-11

Task: TASK-13227. Server source: `39b53a876f2cfefcede02c92ae7dacf88f7a98ec`.

ADR required: no. ADR path: `backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md` and `Docs/ADR/046-persona-live-conversation-and-voice-runtime.md`.
Reason: qualify existing client and ownership contracts, correct test oracles, and repair the missing options-page mount of the existing host.

## Qualification plan

1. Verify the shipped Persona setup labels against their component and English locale, then correct the existing literal E2E expectations. Repeat only the two setup cases against the packaged extension.
2. Exercise independent Buddy selection, explicit conversation/workspace attachment, workspace Persona defaults and Static/Dynamic presentation in disposable browser profiles on the packaged extension and real WebUI. Reuse the current browser harness and synthetic contract fixtures; record fresh versus persisted profile state and the exact storage or migration boundary exercised.
3. Preserve screenshots, source/build hashes, request/state receipts and focused check results. Reconcile AC1 against the actual exercised behavior, preserving accepted earlier UAT and distinguishing browser contract fixtures from real backend schema migration. No provider, physical microphone or user browser profile; no full sweep.

## Reproduced extension defect and approved repair

The packaged options workspace page renders the Buddy & Persona entry, but clicking it opens no management dialog. The native transport fixture reaches the page and its button; the check then fails because the dialog and artwork selection never appear. `OptionsApp` has no independent Buddy host, while `SidepanelApp` owns one. The shared entry publishes only to the page's in-memory management store, so another browser page cannot render its modal.

The approved repair imports the existing `IndependentBuddyHost` in `apps/packages/ui/src/entries/shared/options-app.tsx` and mounts it once in `AppShell` extras. The full options page uses the existing default `web` position bucket and ownership guards. No new host, root, state or storage contract. Rebuild the production extension, then repeat all four client/profile journeys plus the two Persona cases and existing host/component checks. The actual options E2E is the failing regression; no redundant mount-only test is added. Existing ADR-005/046 apply; no new ADR.

## Results

The missing options host is repaired with one import and one `AppShell` extras mount. The native broken-dialog check failed before the repair. The rebuilt production Chrome artifact passed in 71 seconds. The final six-case browser run passed in 67.49 seconds: six passed, zero failed, skipped or flaky; terminal exit 0. The existing independent-host, management-modal and AppShell checks passed 41 cases with zero failures or pending cases; terminal exit 0.

| Client and disposable profile | Actual checked behavior |
| --- | --- |
| Packaged extension, fresh | Select existing independent artwork without a Persona; explicitly attach a conversation; save Static; cancel a staged workspace default; apply workspace attachment and Research Assistant default; save Dynamic and retain it after reload. |
| Packaged extension, upgraded | Preserve position 240,160 through actual Buddy-shell version 0 to 1 rehydration; preserve synthetic existing attachment and default until explicit edits; complete the same selection, attachment, Cancel/Apply and mode/reload journey. |
| Real WebUI, fresh | Complete the same explicit independent Buddy conversation/workspace, default, mode and reload journey. |
| Real WebUI, upgraded | Preserve the saved position through actual version 0 to 1 client rehydration and complete the same explicit journey. |

The two separate packaged Persona cases complete dry-run setup and recover from a controlled starter-command failure. Their corrected oracles use the shipped English step labels and explicit retry/handoff actions. The deleted internal post-target marker is removed; the final Profiles tab and Connections handoff assertions remain. These two cases alone do not qualify independent Buddy setup.

The browser regression asserts real controls and rendered independent artwork, exact attachment IDs, expected mutation versions, Cancel preserving a workspace default, the complete serialized read-only Persona default, saved Static/Dynamic values and Dynamic after a full reload. The existing deterministic one-pixel PNG fixture is used; this is client/transport qualification and does not add real Migu pixel or animation acceptance.

## Runtime and evidence limits

The native cached Chrome for Testing browser loaded a copy of the production extension. Its copied manifest pre-grants the synthetic fixture's loopback origin; the existing launch helper adds its deterministic test key. Original and copied manifest hashes, the actual loaded runtime manifest, browser user agent and all unchanged JavaScript build/copy hashes are retained. `grantHostPermission` verifies that pre-grant here. This does not qualify a human extension installation or permission prompt.

Buddy cases use real extension background HTTP transport with synthetic manual-device authentication metadata. No request API shim is installed. The WebUI ran from this owned worktree on loopback 18180. Each case has disposable browser state and its own synthetic HTTP fixture. Upgraded cases prove actual client-shell version 0 to 1 migration with position retention and synthetic existing server rows; they do not prove a real backend database/profile migration. Earlier accepted backend and human UAT retains its original source attribution.

No provider submission, real credential, physical microphone or human audio playback was used. The two existing Persona setup cases retain their controlled direct-request fallback fixture; their dry-run results are not voice/provider acceptance. The original September 6 trigger remains historically unproven, as reconciled in TASK-13211. No full suite was run. Bandit is inapplicable to this TypeScript/test/documentation scope.

## Retained receipts and checks

- [Verification receipt](artifacts/buddy-client-qualification-20261011/verification.json) records exact source, terminal exits, six browser cases, 41 host/component checks, copied artifact identity, mutation receipts, screenshots and trace filenames.
- [Final Playwright result](artifacts/buddy-client-qualification-20261011/buddy-client-final-green-20261011.json.gz) and [host/component result](artifacts/buddy-client-qualification-20261011/buddy-client-host-components-20261011.json.gz) retain zero failure/skip/pending results.
- [Production Chrome build log](artifacts/buddy-client-qualification-20261011/buddy-extension-host-build-20261011.log.gz) and [source hashes](artifacts/buddy-client-qualification-20261011/buddy-client-final-source-20261011.json) bind the runtime to the approved two-line host repair on base `39b53a876f2cfefcede02c92ae7dacf88f7a98ec`.
- [Original missing-dialog RED](artifacts/buddy-client-qualification-20261011/buddy-independent-clients-workspaces-20261011.json.gz), its screenshot and trace are retained in the same bundle.

The durable trace files retain ordered browser calls, assertions, context metadata and API request/response metadata with raw-trace SHA-256 values. Compiled application assets and DOM snapshots are excluded from that subset. Full raw traces remain in the private test-output directories recorded by the verification receipt; their continued availability is not required to read the retained mutation, artifact and execution evidence.

Run from `apps/extension` after a production Chrome build and isolated WebUI startup:

```sh
TLDW_E2E_SKIP_EXTENSION_BUILD=1 TLDW_BUDDY_WEB_URL=http://127.0.0.1:18180 CI=1 \
node node_modules/@playwright/test/cli.js test \
  tests/e2e/buddy-independent-setup.spec.ts tests/e2e/persona-assistant-setup.spec.ts \
  --workers=1 --retries=0 --reporter=json --trace=on
```

Scoped ESLint reports no findings for the new E2E and production host mount. The existing Persona test retains the same one error/four warnings as the original source under the same filename/configuration. New E2E formatting passes. Full-file options formatting drift also predates this change; the existing surrounding style is preserved. Whitespace checks pass. Independent source review found no actionable host or oracle issue.

## Fixture lesson

This qualification initially failed its native upgrade check because navigation from `options.html` to its workspace hash keeps the same document and does not execute a newly installed `addInitScript`. A real reload is required before claiming version 0 seed/migration evidence. A second fixture assumption toggled already-expanded Buddy options closed after an attachment change; the final check reads the existing details state before opening it. Neither failure justified another production repair. Intermediate failed receipts remain under `/private/tmp/buddy-client-final-six-20261011.json`; only the corrected final six-case run qualifies completion.

Captured logs and raw JSON lacking a canonical final newline are gzip-compressed so Git whitespace checks preserve their exact bytes. The receipt records stored and decoded hashes; decompression was verified against every original capture. Product source and other documents still receive ordinary whitespace checks.
