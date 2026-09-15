# Cached Chat session title and metadata ordering

TASK13260.34 follow-up. This is the reviewed automated checkpoint; native Aster/Robot backlink, reload and rename acceptance, the final combined compiler and full fresh UAT remain parent-owned and pending. No acceptance is inferred from jsdom.

## Change and evidence

The cached session restore previously marked assistant-only metadata complete, allowing a settled saved chat to retain messages but lose its canonical title/version. A title is now restored from local history only when its server_chat_id exactly matches the saved target. Full metadata stays pending for the canonical loader. Existing cached messages remain available offline.

Independent review then found an ordering race: restoration awaited assistant persistence before publishing cached identity and the pending flag. A canonical loader could finish during that await and then be downgraded by cached state. The correction publishes all cached metadata synchronously before either assistant-persistence await, retaining the existing cancellation checks afterward.

- Initial cached-title regression: original3 failures/6 controls, corrected9 passes (overlapping suite).
- Permanent ordering regression: original1 failure/9 controls, corrected10 passes.
- Independent original actual WebUI Storage + session + loader probe:1 RED, then1 GREEN. Final trace preserves canonical metadata readiness/identity/title after the held write.
- Independent actual formatter/session/loader offline control:1 GREEN, cached title/target/answer remain when metadata/messages/profile requests fail.
- Adjacent session/local-load/sidebar tests:24 passes/3 suites. Latest test-only format/type adjustment was followed by10 final session passes. Counts overlap and are not additive.
- Lint comparison:0 errors,1 unchanged warning,0 added. Bandit is not applicable to TypeScript-only changes; combined compiler/native are pending.

## Original inputs and replay qualification

The original race config and its assertions are retained byte-for-byte. The separate current wrapper differs only by bypassing the obsolete .33 loader transformation, because that pre-metadata scaffold has now been removed in production. The wrapper does not change test assertions or the actual storage/session/loader boundary. The offline config builds on that current wrapper.

No exact full session source snapshot was saved immediately before the ordering move. The original RED config/log/review remain historical evidence; the earlier session-title-before.tsx snapshot belongs to the INITIAL cached-title metadata failure, not the later ordering failure. Do not describe it as the ordering preimage. No reconstructed source is represented as an original.

Configs preserve their original absolute repository and /private/tmp references. To rerun in this workspace, restore retained uat034-*.config.ts files to those paths and use the existing frontend Vitest runner. The current race config uses -t 'UAT034 late cached restore'; the offline config uses -t 'UAT034 keeps matching cached title'. Original race RED also required the then-current .33 loader scaffold; it is not a standalone replay against later source. Final source/test snapshots and current-source-freeze.json bind this reviewed checkpoint.

## Retention and limits

Only log/Markdown/text trailing whitespace is normalized. Original source, probe/config and JSON bytes are preserved exactly, with per-file origin hashes in retention-manifest.json. SHA256SUMS binds every retained artifact except itself. Private scans found no known isolated runtime credential, JWT or private-key pattern. Task records may gain notes after the original frozen hashes; production/test hashes are the review boundary. No browser, backend, actual IndexedDB or native geometry was exercised by the reviewer. Parent will add final native/compiler evidence separately.
