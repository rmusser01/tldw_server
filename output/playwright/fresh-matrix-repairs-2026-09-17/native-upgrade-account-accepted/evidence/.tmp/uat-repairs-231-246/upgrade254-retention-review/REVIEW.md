# UAT254 retention audit

**CLEAR — no actionable findings.** Read-only audit of `output/playwright/fresh-matrix-repairs-2026-09-17/targeted-upgrade-harness-reviewed`; no runtime acceptance claim.

- Exactly **63 payloads / 66 total files**. The manifest inventory matches disk, all 63 payload byte counts/hashes and original source bytes match, and all 65 checkpoint entries match. No extra files, duplicate destinations or retained symlinks.
- Top manifest SHA256: `592c3292c98a5a4abfb5c72f0b9ff1e933751d2cb0c6631207c57e35d0e50f70`.
- CHECKPOINT_SHA256SUMS SHA256: `be750f6be7a0eadff9f8580c78f062d9a152e89b94d8220b2ab0f9a5b6ab0b04`.
- Retainer SHA256: `54e2f5480cbf935a07419e0343d0cf1426604db9a7db5651fc02f6c538686522`.
- All 32 initial evidence entries, original owned-manifest hash and three initial source snapshots verify. The initial test uses the retained baseline and preserves SHA256 `22dc359843445e5464db7b9276483ede17cb95a60e78b7f4267c9e024a7167b3`. The corrected test is separately retained with SHA256 `f6ab8c0e5f03139fd0360272c47453ed48c798130fd30d5884ae188c8c1f950f`; all three corrected source-freeze entries verify.
- README claims match the retained evidence: initial independent 124 checks, native Next-guard failure, causal RED, corrected independent 125 checks with zero skips/failures, successful syntax/parsed ESLint, unchanged product guard, and Bandit's JavaScript parse limitation. The README properly states that corrected startup/readback remain pending and makes no fresh-install/full-matrix acceptance claim. Its historical pending status describes this frozen packet.
- Independent credential scan covered **all 66 files**, including README, manifest and checksums. Collected 41 distinct known credentials from the six named original/fresh profiles, runtime PG configurations, retained fresh PG holders and local PostgreSQL provisioning record. Raw, URL-encoded and JSON-escaped forms yielded 41 distinct variants; an additional base64/base64url scan covered 63 variants. **Zero known credential matches, zero encoded matches, zero JWT-pattern matches.** No secret values were printed or retained in this audit.

The retainer was inspected, not rerun. No tests, source-copy operation, launcher, runtime, browser, database or network action was performed. No root docs/tasks/Git/private records or retained packet files were changed. Only this separate audit packet was written. Exact machine-readable results and the independently implemented audit procedure are in `audit.json` and `audit.mjs`.
