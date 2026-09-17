# Accepted native stream, media and MCP checks

Bounded independent acceptance of UAT246 (exact fresh TestBot first turns in both PostgreSQL auth modes), UAT253 (original persisted Media1 loads after source upgrade), UAT241 (full source handoff plus actual loading guard), and UAT251 (fresh PostgreSQL setup Save packs, sample tool and persisted state). Runtime source a7d3155a567afb25982eb360ea24b973cc3249c9. This does not release or accept the full fresh matrix.

The three independent reviews qualify their scope and preserve failures: the source-handoff mistaken prompt is not an exact TestBot attempt, existing-history runs ended with reasoning but no final answer, and the initial claim of model overlap was corrected. Original frozen source and profiles remain intact. Media1 was not reuploaded. MCP uses a distinct fresh profile because the old wizard was already completed; first chat and optional audio remain outside its acceptance.

Reviewed nonprivate inputs are retained byte-for-byte, with lossless gzip above256KB. Private records, raw runtime logs and large repository archives remain represented by hashes in safe audits; they are not copied. Archive source manifests and selected source snapshots are retained. No product test rerun was needed for retention. JavaScript audit code is outside Bandit's supported language scope. Known credential/JWT scans must pass before writing this packet.

UAT232 remains open: its ordinary chat native check revealed generic error guidance on an actual model_not_available400. This packet does not accept that issue or any additional upload-isolation findings.
