# UAT258 native evidence retention review

**CLEAR — bounded UAT258 retention.** Independent verification passes 18 checks. This review makes no additional native acceptance claim.

## Verified packet

`output/playwright/fresh-matrix-repairs-2026-09-17/native-capability258-accepted`

- Manifest SHA-256: `5f6335979143612103e6cd0c5445fb3decf62edd18e9a91ff1e9721df2bb9702`.
- Exactly three expected safe payloads and six total files. Stored bytes, source bytes, manifest hashes and checksum inventory agree. No packet symlinks, unexpected files or escaping paths were found.
- The original native audit remains unchanged: 18 passing checks over 53 hashed inputs on `edfd06ec40a173f2e38ec65af715abb29f3aa002`.
- All 53 input references retain their original hashes, sizes and private flags. Fifty-two current files still match fully. One private live log has appended; its original 733,203-byte prefix still matches `9ab39db347239fc1a6c04713dc487c8bf2ec8dee44d792afe23ad91613e23d5c`. The manifest explicitly discloses this prefix verification. Its later bytes are neither original review evidence nor a copied payload.
- All 13 private inputs remain omitted as raw payloads. Raw native captures and source files also remain local. The original audit script is both a safe retained payload and an input reference; that reference overlap does not claim raw capture retention.
- Independent scanning of all six packet files against credentials from seven local profiles, 122 deduplicated encoding variants and a JWT pattern found zero matches. Credential values were processed in memory and not emitted.

## Scope and limitations

The packet accepts **UAT258 only**: fresh unconfigured contexts in both PostgreSQL modes avoid protected capability requests/errors, and a real authenticated Alice capability request succeeds. It preserves the correction that the old general observer excluded hyphenated `ingestion-sources` paths. Fresh-context observations and the bound authenticated backend access record supply the relevant evidence.

UAT264 remains open: Sources listing loses Authorization across the observed cross-origin redirect and returns 401. The packet does not accept Sources listing/creation, every auth variant, authenticated PostgreSQL single-user capability behavior, or the full matrix. Omitted inputs have hash-only provenance; this is not standalone replay. Credential scanning covers known local values and selected encodings, not every possible secret.

The retainer's leading comment contains stale administrative task IDs. Its manifest, README and acceptance payloads consistently identify UAT258, so this does not affect evidence integrity or acceptance scope.

## Verification

Ran `node .tmp/uat-repairs-231-246/capability258-native-diagnosis/retention-audit.mjs`: **18 checks passed, zero failures**. Details and safe current hash observations are in `retention-audit.json`. No browser, model, runtime, DB, product, test, Git or Backlog changes were performed; only this supplemental retention review was written. Original review and retained packet bytes remain unchanged.
