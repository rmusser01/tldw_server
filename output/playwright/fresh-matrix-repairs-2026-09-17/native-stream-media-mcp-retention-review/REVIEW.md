# Independent retention review — native stream, Media and MCP

**CLEAR — no actionable defect in the retained packet.** Reviewed `output/playwright/fresh-matrix-repairs-2026-09-17/native-stream-media-mcp-accepted` offline without running its retained audit scripts or retainer.

## Exact inventory and bytes

- **186 payloads, 189 files, 188 checkpoint entries, 12 gzip payloads.** Disk, manifest, checksums and the independently reconstructed union of reviewed inputs plus all nine review files match exactly. No extra files, duplicate destinations, conflicting review hashes or retained symlinks.
- All original bytes/hashes, stored bytes/hashes and original-to-restored comparisons pass. Every gzip payload restores exactly. Original payload bytes total **29,893,958**; stored payload bytes total **8,301,041**.
- All three native reviews and audits retain their CLEAR verdicts: stream **42/42 checks, 53 inputs**; Media **27/27 checks, 78 inputs**; MCP **38/38 checks, 73 inputs**. No native or product tests were rerun here.
- Exactly **17 intentional omissions** are accounted for: **16 private profile/process/initialization/holder/config/log records** and **one repository archive**. Their recorded hashes and safe summaries remain in the retained MCP audit. The raw omitted inputs were not copied. Private profile/config records were read only to assemble the known-credential scan; omitted process/init/holder records, raw logs and the archive were not reopened. Live-log hashes remain historical snapshots, not a promise that current logs cannot append.

| Artifact | SHA256 |
| --- | --- |
| Manifest | `f8d1c2c9287e456fa71b5eb0ff731301bc42e3f8fa755caabec99cb1f0f04902` |
| CHECKPOINT_SHA256SUMS | `84a2297b176074dcab1b1675e009893a16315e13319ee620972dbc819994b984` |
| Retainer | `9b76565ae06a0d6df742b8e3955839d984c9d42a15388796fd62810523fd6b82` |

## Scope and claim checks

The README and manifest preserve bounded acceptance of **UAT241, UAT246, UAT251 and UAT253** on source `a7d3155a567afb25982eb360ea24b973cc3249c9`. They do not accept the full fresh matrix, UAT232, or new upload-isolation findings.

The retained reviews preserve the important qualifications: exact fresh TestBot first turns are distinct from the wrong-prompt handoff and existing-history no-final-answer outcomes; the earlier overlap claim is corrected; the original 45-second failure's cause is not retrospectively established. Media's original single-user item, final-newline normalization and later separate multi-user loading observation remain distinguishable. MCP proves Save packs, the built-in sample and persisted state in a distinct fresh profile; optional audio, first chat and complete-wizard acceptance remain outside scope.

## Paths and credential scan

The corrected retainer normalizes audited absolute paths to repository-relative paths. Its lexical outside-path and **leaf-symlink** checks precede destination creation. The reported initial failed invocation was not replayed; the current ordering is consistent with failure before writes.

Every actual source resolves to a regular file within the repository. Three reviewed dependency package manifests resolve through **internal Bun directory aliases** to the copied source's `apps/node_modules/.bun` tree. Their bytes match the native MCP audit. Thus the accurate claim is no retained symlinks or escaping targets; the source tree itself does contain these legitimate directory links.

Independently scanned **all 186 original/decompressed payloads and all three metadata files** using credentials from all **seven current matrix profiles**, including MCP251, their runtime PG configs and the local PostgreSQL provisioning record. **48 distinct known secrets** produce **122 raw, URL-encoded, JSON-escaped, base64 and base64url variants**. Credential and JWT-pattern matches: **zero**. No values or raw model reasoning were printed or retained in this review.

Two local auditor assumptions were corrected before the final run: select the packet's top-level README explicitly, and allow the three reviewed internal Bun aliases while retaining resolved-target confinement and regular-leaf checks. These were auditor issues, not packet mutations.

Only this separate REVIEW.md, audit.mjs and audit.json were written. No source, packet, Git, Backlog, tracker, runtime, browser or database change was made. Bandit's unsupported JavaScript scope supplies no security certification. Detailed source, gzip, omission and scan results are in audit.json.

Audit SHA256: `7b79c81494a713bc947a446e4396d5ecd50f881ae3594b1329147461ae0fab73`.
