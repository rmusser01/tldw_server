# UAT093: saved Chat entry follow-up

TASK13260.33 retains the **failed native retest after the first reviewed correction**, the subsequent selection/metadata race repair and its regression evidence. Frozen source/test scope: six paths in uat093-followup-frozen-manifest.json. Exact native saved Robot acceptance and full fresh matrices remain pending. The independent report is retained; use its stated verdict and limitations.

## Native failure

- uat093-native-repaired-entry-timeout-snapshot.txt and uat093-repaired-entry-stalled.png show the Robot title, stale Cedar character and empty/preparing timeline after the initial repair.
- uat093-repaired-stall-normal-reload.txt records another failed normal reload and two real auth/me200 responses for Alice id2. This is failure evidence, not successful recovery.
- The PNG was visually inspected: it shows the described mismatch and no credentials. Empty uat093-native-repaired-robot-entry.txt and uat093-native-stalled-summary.txt were excluded, not rewritten as responses.

## Regression evidence and replay

Final implementer tests pass128/9. Metadata becomes ready only after canonical selection and the existing shared selection commit chain settle; newer queued picker intent wins. Same-turn picker/profile, replacement, principal and unmount controls are permanent actual coordinator tests. The held-storage cases model installed Plasmo await-storage-before-render behavior. No real browser, network or Dexie acceptance is claimed by these tests.

The three original metadata configs and concurrent-metadata config retain their original bytes. Their explicit *-compat.config.mjs runners adapt fixture shape/mock exports while keeping original probe logic. Original timing runners use the one retained prior-coordinator-fixture.tsx, required to replay that old transform. The concurrent runner only restores the Form stub expected by the original transform, which inserts two real loader consumers and holds both responses. Original bodies pass1+1+1+2; these are **adapted harness runs**, not unchanged harness claims.

The held-messages original RED-named log traced clear/repin but eventually recovered; do not count it as an original failed assertion. The shared-storage-prior.config.ts intentionally removes only the final settlement wait/guard at transform time to demonstrate a permanent RED. The current same-turn profile test has a RED log, not a separate original config. The report contains exact commands.

Configs reference the original checkout and /private/tmp paths. For replay after temp cleanup, copy the retained uat093*.config.* and prior-coordinator-fixture.tsx files to their original /private/tmp names, then use the report commands from apps/packages/ui. Run intentional RED transforms separately from the passing suite.

## Validation and provenance

Scoped ESLint has0errors and23 exact existing warnings, with no added/removed warning signatures. JSON copies omit redundant embedded source/output fields; diagnostic results are retained. TypeScript exits2 with exactly90 merged-baseline signatures and no additions/removals; this is not a clean typecheck. The compiler ran22:07:08–22:07:47UTC, after the adjacent source freeze. Bandit is inapplicable to this TypeScript-only scope.

Original config/fixture/JSON/report bytes are preserved except the explicitly stripped ESLint source fields. Log copies normalize trailing whitespace/final newlines. manifest.json records original and retained hashes, sizes, times and transformations. Originals remain untouched. SHA256SUMS indexes every bundle file except itself. Texts were scanned against both private cycle3 runtime credential sets and token/private-key patterns with zero matches; runtime manifests and credentials are never included. scan.json records counts only. The frozen manifest describes its original freeze-time state; the Backlog record may subsequently advance.
