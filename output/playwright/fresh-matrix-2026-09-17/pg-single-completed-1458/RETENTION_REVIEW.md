# PostgreSQL single-user checkpoint retention review

Reviewed 2026-09-17T15:10:40.761658+00:00 under parent TASK13260. **Retention verified: no missing PG-row evidence or byte-fidelity defect.** This review certifies the evidence package, not a clean product or full-matrix pass.

## Integrity

- Main manifest: `5d0f60e143811af5d45cfc575ef6b5753aa6e51d391e6a23d5daaecaa9e09652`; created `2026-09-17T15:05:55.998Z`.
- Source attribution: `8f8774e6c868b304a96d95ab82e28389c129a78b`; run `fresh-final-20260917`. No new source build/check was performed.
- All **125 unique payloads** pass retained and source SHA-256/byte-count checks. Retained total **1,212,934 bytes**; uncompressed source total **1,323,368 bytes**. **125/125** equal their current authorized working sources byte-for-byte; no normalization.
- Exactly one gzip: `native/pg-single/image-final-evidence.txt.gz`, **7,320 bytes**, SHA `9b9259f20aca106248417ccbcb05fb1547a7a35a0685c2d881f89920411e6117`. Decompression gives **117,754 bytes**, SHA `29050c586db9544d7ebdb634e4f653a6aa86294f46b519bf81b6a844e4f24ed8`, identical to the working `.txt`. Logical evidence paths require manifest-aware gzip resolution.
- Retained `retention-source.mjs` equals working `retain-pg-single.mjs`, SHA `b7c468052d6c95b8a62035eaabf0d2f47258db6fafb08a201e3b6bb088de248e`. The script was read, not executed.

## Coverage

The125 payloads comprise **118 native receipts, five audits, matrix-progress and the controller**. All123 selected native/audit inputs present at retention are preserved. The only subsequently eligible file outside the frozen selection is the parent's explicitly excluded administrative follow-up `native/pg-single/backlog-lapse-audit-update.txt`; it is not native workflow evidence and was not added to this checkpoint.

All **50 PG row references / 48 distinct logical paths** in retained progress resolve to retained payloads. All12 rows have bounded outcomes. Five unchanged audits are included: setup/Chat, late-native, ingest RLS, world-book catalogue, and Hard/lapse analytics. The setup audit's **27** input hashes and late-native audit's **49** input hashes also match the retained uncompressed inputs, including controller/progress. Diagnosis reports may reference production source hashes outside this package; this is intentionally not a source distribution.

The referenced base manifest is bound here as `e3403b75043675dff9abc878d594c26c9eecfb6f8125011489b60b7e3d102d6e` at `../sqlite-single-checkpoint-1331/manifest.json`. Its release gate equals the working gate, SHA `c692f5d8bb0f1c653e9172515477d0c90cb46002e4dea68fd916c1fa3ff629d0`. All **15** declared current harness files still match. The entire base payload/archive/dependency review was not repeated. SQLite completion also uses the earlier completed-delta package.

## Privacy and exclusions

The inspected retainer selects direct regular `.txt/.json/.md/.js` files under `native/pg-single` and `pg-single-*` audits, excludes filenames matching `private|credentials`, and adds controller/progress. The actual package contains no symlinks, traversal paths, private/credential filenames, browser profiles or raw `.log` files. Sanitized log excerpts are intentionally included, not the full private logs. Referenced snapshot/log paths inside receipts were not opened or copied by this review.

The parent's manifest records **20 distinct known values/encodings,125 scanned candidates,0 matches,0 JWT-shaped matches**. I inspected that scanner but did **not** access private files or rerun its known-value credential scan. Its code covers API/JWT/hash/account secrets from the three prepared profiles, SQLite provider secrets and provisioning/runtime PostgreSQL passwords, with URI/JSON encodings; it does not enumerate providerSecrets separately for every later profile. No broader provider-secret coverage is asserted here. Independent JWT-shape scanning of all125 uncompressed payloads found **0 matches**. Filename/shape scans cannot prove absence of unknown secrets.

## Qualifications and auxiliary binding

- README/controller preserve findings231–240 and explicit failures/blocks. Pirate-answer reuse is not media-grounded reuse; image guarding is not vision; Character and five-card paths do not pass. Browser closure/recovery, API-key expiry N/A and both multi-user cells pending remain explicit. No main ledger/tracker changes were made.
- Progress snapshot `2026-09-17T15:00:50.020Z` still calls PG row3's image guard **in progress**, while the retained final audit/controller give its completed limitations. This is stale chronological progress wording, not evidence of vision acceptance.
- The main manifest's base link is relative, not an inline base hash; this review binds the exact base above. Private runtime provisioning material, source archives and dependency trees are intentionally omitted.
- README, retainer source and this review are auxiliary, outside the original125-payload manifest. Authorized `CHECKPOINT_SHA256SUMS` now binds **every packet file except itself**:125 payloads, main manifest, README, retainer and this review (**129 entries**). Its own SHA is separately returned to the parent; self-hashing is excluded.
- A first review-only nested-hash parser treated root `matrix-progress.json` as a bare native filename and stopped before writing either output. Correcting that path mapping completed all76 nested checks; no payload, source or product behavior was changed.
- No browser, runtime, API, database, inference, product test, task or git action occurred. Only this review and the authorized checksums file were written. All original payload/manifest bytes were rechecked unchanged.
