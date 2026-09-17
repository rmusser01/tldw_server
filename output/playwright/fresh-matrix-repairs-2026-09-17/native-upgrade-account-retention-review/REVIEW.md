# Independent retention audit — UAT243 / UAT248 / UAT254

**CLEAR — no actionable defects.** Reviewed `output/playwright/fresh-matrix-repairs-2026-09-17/native-upgrade-account-accepted` offline. This verdict concerns retention integrity and the packet's stated acceptance scope.

## Integrity

- **74 payloads, 77 total files, 76 checkpoint entries, exactly two gzip payloads.** The disk inventory, manifest, checkpoint inventory and independently reconstructed inventory from both native audits plus the specified review/metadata files match exactly. No duplicate destinations, unexpected files or retained symlinks.
- Every stored byte count/hash matches. Every identity payload and decompressed gzip payload exactly matches its original source and reviewed source hash. The two source manifests restore to **8,354,106** and **8,354,105 bytes**; both gzip round trips are exact. Original and stored hashes are recorded in `audit.json`.
- Twenty private inputs from the native provenance audit are excluded from copying. Their safe audit summaries/hashes remain retained. Both native three-file review packets and the earlier implementation-retention three-file review are present and byte-identical to their sources.

| Artifact | SHA256 |
| --- | --- |
| New manifest | `6cb62377b2f6c39218c4c367164540648053dd0722203ddee778e66d07d1ff3f` |
| New CHECKPOINT_SHA256SUMS | `8ccdc5991de8c7c28331b30f35c864471c52c3b61e1741ebb4e820abe089bcdc` |
| Retainer | `aab2aec4d01940739d14e8ae42b42f57eee2395fc527cb06cad2530c0afb252b` |

## Claims and historical evidence

README and manifest correctly limit acceptance to **243, 248 and 254**. The retained native upgrade audit is CLEAR; the account audit retains all **26/26** successful checks and its bounded CLEAR verdicts. The README preserves the meaningful qualifications: contradictory early Character label and stale canonical events are excluded from settled acceptance; failed harness attempts remain retained; blank-tab recovery does not establish same-tab survival; cancellation was observed before body bytes, with no late assistant at later readback, without claiming provider-side cancellation. No new UAT246 attempt or full-matrix acceptance is claimed.

The earlier `targeted-upgrade-harness-reviewed` packet remains unchanged: **66-file inventory, all 65 checkpoint hashes**, manifest `592c3292c98a5a4abfb5c72f0b9ff1e933751d2cb0c6631207c57e35d0e50f70`, and checksums `be750f6be7a0eadff9f8580c78f062d9a152e89b94d8220b2ab0f9a5b6ab0b04` all verify. Its original “actual corrected startup pending” statement is preserved as a historical checkpoint; the later native packet supplies UAT254 closure.

## Credential scan and boundaries

Independently scanned **all 74 original/decompressed payloads and all three metadata files** against credentials collected programmatically from all **seven current matrix profiles**, including `mcp251-fresh-targeted-20260917-pg-single`, their runtime PG configurations, and the local PostgreSQL provisioning record. The scan covered **48 distinct raw/URL/JSON variants**, **74 additional base64/base64url variants**, and JWT patterns: **zero matches**. No private credential values were printed or written.

The retainer and retained audit scripts were inspected, not executed. No tests, runtime/browser/DB actions, source copies, packet mutation, Git, Backlog or tracker changes occurred. Only this separate `REVIEW.md`, `audit.mjs`, and `audit.json` were written. JavaScript security certification is not inferred from Bandit. The machine-readable audit records every original source hash, compressed-file result and check outcome.
