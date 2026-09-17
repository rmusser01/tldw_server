# Paired PostgreSQL Study retention review

**CLEAR — no retention blocker.** The retained packet faithfully preserves the two completed native audits supporting original-scenario acceptance for UAT234, UAT235, UAT240 and UAT242 in PostgreSQL single-user and Alice multi-user modes. This review did not repeat native actions or tests.

## Verified binding

- Packet: `output/playwright/fresh-matrix-repairs-2026-09-17/native-study-accepted`.
- Exact inventory: **68 payloads / 1,565,037 payload bytes / 71 total files**. The only additional files are README, manifest and checksum index. No unexpected file or symlink exists.
- All payload lengths and SHA-256 hashes match both the retained manifest and current original source bytes. All **70 checksum entries** pass.
- Manifest SHA: `1eaa9c16e37f956198d9184f2b120ee287f6cf157dfb31abb7b325a31bb66204`.
- Checksum-index SHA: `b93d291e34d93592b67f7e69db978d5f9a3c6f906141298cc435a249a23f614f`.
- The exact reviewed-input union is complete: PG-single **41 inputs / 838,077 bytes**, PG-multi **17 inputs / 671,467 bytes**, both reports/projections/input manifests/reviewer manifests, original targeted gate and RUN.md. Both reviewer manifests and all referenced review artifacts verify.
- Retainer SHA: `ff396d06e62cc9408d72f8d278df6c4a7371f21a9889dbe436f1e9838f69c1e6`. It checks original input/review hashes, rejects unsafe names/symlinks, and scans selected payloads before creating the destination. The retained inventory contains no private helper or private runtime log.

## Privacy and acceptance limits

An independent in-memory scan of **all 71 retained files** against **41 known credential variants** from the six old/targeted profiles and relevant provisioning/runtime PostgreSQL secrets found **0 matches**; JWT-shape scan found **0 matches**. No secret values were emitted or copied into this review. This establishes known-value/shape checks, not the absence of every possible unknown secret format.

README claims agree with the existing audits: slow successful generation crossing 30 seconds, five distinct grounded drafts saved/reviewed/reloaded, bounded practice-OFF preservation, Good10 followed by authoritative re-rate Hard14, persisted lapses0 and analytics7/100%/0, and separate singular sessions. Re-rate is an additional review; the singular control is the sixth card/eighth review. Original harness selector failures remain retained and qualified.

The gate and RUN.md record revision `86458ab88ce3fa62e6518c9d813c3860254ddb2c` and the targeted PG single/multi run. This retention review does not independently revalidate runtime source or role state. RUN.md's early “no browser acceptance yet” is explicitly identified as historical in README. No clean dependency-installation, raw-SQL/RLS isolation, optional-service, later-runtime or full48-outcome matrix acceptance is claimed.

Only this private review directory was written. Retained payloads, product files, browser/runtime/database state, tests, tasks/tracker and Git were untouched.

## Receipts

`verification.json` records exact counts, input/report bindings, gate provenance and scan results. `verify.mjs` is the independent verification command source; it was run as `node .tmp/uat-repairs-231-246/native-study-retention-review/verify.mjs` and exited0.
