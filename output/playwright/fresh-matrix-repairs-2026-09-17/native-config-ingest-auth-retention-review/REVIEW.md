# Independent retained-packet review

**CLEAR for bounded retention of the accepted UAT231/233/237/244/245/247 evidence.**

Packet: `output/playwright/fresh-matrix-repairs-2026-09-17/native-config-ingest-auth-accepted`.

- Final manifest SHA256: `c4f50b39282f2d6f71d8ac82763357608f7f6d9dffd475b3186fe0433fb95cbf`.
- Checkpoint SHA256: `7173b069f0a6401d07892f9ccee03f9edb45208275f10f194b6229cd2f61585e`.

## Verified retained bytes

The packet contains **114 payloads, 117 total files and 8 gzip payloads**. All116 checkpoint entries match their stored files; the checksum file is the sole intentionally unlisted file. Manifest stored sizes/hashes and decoded source sizes/hashes match every payload. All114 decoded payloads equal their current original inputs byte-for-byte. All8 gzip streams decompress exactly and match canonical level9 re-encoding, without extra header/member data.

All payload and source paths are relative and contained within the intended repository/packet. No duplicate paths, unexpected files, symlinks, special files or symlink ancestors are present in the reviewed paths. The current packet is safe on these checks; this is not a certification of the retainer for arbitrary future inputs.

**146 input references** across the five original review audits map to either an exact retained payload or an explicit omission with the original observed hash. Every Markdown, script and JSON artifact in the three original review directories is retained, including initial partial verdicts, later supplements, failed helper evidence and the earlier UAT231 gap. Copied repository source manifests themselves are byte-identical; this retention review does not rehash every file in the larger unretained repository trees.

## Exclusions and review correction

The manifest explicitly records24 omission entries, including repeated references to the same underlying records. Private credentials/helpers, runtime bindings/process receipts, configuration/initialization/holder records and repository archives are excluded. Safe original audits retain the relevant allowlisted identity, timing and hash observations. There is no retained-byte-parity claim for omitted records.

Running tracker/task metadata is also represented by historical observed and current hashes with a stated reason. All four metadata current hashes matched at this review. Task179 is unchanged at that observation point but is still deliberately hash-only under the metadata policy; it is not private data.

The first submitted manifest (`fe005dde4513b7c64521a68ac669e14ac47ece9c7f4a8ed6d7dca9fb4941ce74`) mislabeled public task179 as private because its filename contains “credentials.” The reviewer reported this. The controller corrected the retainer to classify public task metadata before the private-file filter, added the truthful observed/current-hash explanation, and regenerated the final manifest/checkpoint above. Payload bytes and counts remained unchanged. This finding is resolved in the reviewed final packet.

The controller also disclosed that an earlier retention attempt stopped on changed running tracker metadata. The final retention logic allows only the disclosed metadata/private/archive omissions; mismatched product or native-evidence bytes still fail. The omitted historical tracker/task versions are not represented as if their current bytes were the reviewed bytes.

## Private-data check

An independent in-memory scan read the known credentials for7 profiles and the recovery PostgreSQL configuration, derived122 raw/escaped/URL-encoded/base64 variants, and scanned all decoded payloads plus README, manifest, checksum and stored gzip bytes. It found **0 known-secret matches and 0 JWT-pattern matches**. Credential values and private file contents were never serialized or printed. No raw runtime logs, `.private.*` payloads or repository archives are copied.

This is a scoped known-value/pattern scan, not proof against every possible unknown secret format.

## Acceptance claims and limits

The retained assessments support the six named bounded findings. The ingest supplement supplies the warning Results label and literal Open in Media evidence missing from its initial review. The UAT231 fresh-context supplement supplies the corrected generic branch missing from the first capture. The UAT237 packet supplies intentional Disconnect/reconnect and the real owned-API outage/Retry sequence. The README preserves those distinctions.

The README also keeps the substantive limits: analysis was truncated and did not succeed; source persistence and warning reporting succeeded. Native hard-quota/concurrency, all filter permutations and a fresh SQLite matrix are not claimed. UAT238 dependent source QA/reanalysis/Trash and UAT257 retrieval remain open; the capabilities401 is separately tracked as UAT258. Settled snapshots do not prove zero requests or every transient UI event. The packet explicitly does not release or accept the full48-row matrix.

This reviewer changed only the three retention-review files. No product, tests, Git, task, documentation, browser, model, database, runtime or retained-packet mutation was performed by the reviewer. Verification is recorded in `audit.json`; the retained packet's original native acceptance reviews remain the authority for their specific behavior claims.
