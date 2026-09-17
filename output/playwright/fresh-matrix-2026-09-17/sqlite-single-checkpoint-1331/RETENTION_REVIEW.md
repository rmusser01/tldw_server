# Checkpoint 13:31 — independent retention review

## Verdict

**Clear for faithful retention of this partial matrix checkpoint, with two minor manifest omissions and explicit portability limits below.** This is an evidence-packaging review, not a fresh acceptance run or full-UAT signoff.

Packet: `output/playwright/fresh-matrix-2026-09-17/sqlite-single-checkpoint-1331`.
Reviewed manifest SHA256: `e3403b75043675dff9abc878d594c26c9eecfb6f8125011489b60b7e3d102d6e`.
Frozen revision: `8f8774e6c868b304a96d95ab82e28389c129a78b`; run `fresh-final-20260917`.

## Verification performed

- **201/201 listed payloads** match their exact retained SHA256 and byte count. Decompressed or plain source bytes also match every recorded source hash/size. No duplicate manifest paths, missing listed file, payload symlink or private-named payload was found.
- All 201 source/output pairs were also compared to their working originals, using the retainer's mapping for the controller and compressed manifests: **zero differences** at comparison time. No normalization or silent redaction occurs in this retainer; it copies previously scrubbed inputs and compresses only the four source manifests.
- The four gzip payloads decompress successfully, match recorded original lengths/hashes, parse as JSON and match their current uncompressed working originals. Their complete ordered `files` arrays are equal: **33,433 entries each**, with 33,432 regular-file records and one preserved-symlink record; zero duplicate paths. Revision matches the release. This verifies retained manifest integrity; it does not re-read all archived source file contents or rebuild/re-extract an archive.
- **15/15 release-gated harness hashes** match both the packet and current working files. The retained release gate is byte-identical to the original, SHA256 `c692f5d8bb0f1c653e9172515477d0c90cb46002e4dea68fd916c1fa3ff629d0`.
- Retained `retention-source.mjs` is byte-identical to the actual `.tmp/uat-next-matrix-20260916/retain-checkpoint.mjs`. The script was read, not executed.
- All **27** evidence paths directly named by `matrix-progress.json` exist in this packet. Embedded retained-audit bindings were checked: 18 Chat inputs, 26 source/reuse native inputs, 15 five-card diagnosis bindings and 13 re-rate diagnosis bindings match. Source bindings were checked against the preserved archive manifest; the five separately bound Next dependency files were checked against the still-present copied dependencies.

### Gzip source-byte bindings

| Compressed payload | Uncompressed bytes | Uncompressed SHA256 |
|---|---:|---|
| `copy-preparation/pg-multi-archive-manifest.json.gz` | 7,890,727 | `09120994a7b5346c5372b2954de1e077caa375626cef796a12070a579a666c03` |
| `copy-preparation/pg-single-archive-manifest.json.gz` | 7,890,728 | `f9a6d30e6a8faef5635df40d5ee026e18ebc225d344bf29b62a8bcca7f2b2f4f` |
| `copy-preparation/sqlite-multi-archive-manifest.json.gz` | 7,890,731 | `703f3cd208450440ee4ce31ba51526200257f363fc7fa3fb07c4811d4fa9cb36` |
| `copy-preparation/sqlite-single-archive-manifest.json.gz` | 7,890,732 | `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1` |

## Exclusions and credential check

The retainer selects only the gated files, two named state files, immediate regular files with `.txt/.json/.md/.js` extensions in four named evidence directories, and the controller document. It excludes names matching `private|credentials`; it does not traverse profile/log/browser directories or copy source archives/dependencies. The current native directory has one private-named file; it is excluded and its exact bytes do not occur as a retained payload. No raw `.log`, credential-entry helper, browser profile or private runtime log is copied. The intentionally retained `flashcard-failure-log-excerpt.json` is the sanitized diagnostic excerpt, not the private log.

I independently scanned **all 204 actual packet files**, including README, manifest and retainer source, with gzip contents decompressed. The same seven deduplicated known credential/value encodings were read internally only for this check. Result: **zero known-value matches and zero JWT-shaped matches**. No values were printed or written to this audit. These bounded checks do not assert discovery of every possible unknown secret.

The retainer uses directory/extension selection rather than an immutable per-file candidate allowlist. The resulting 201-file manifest is the exact reviewed snapshot; this verdict does not pre-approve unrelated future files added to those directories.

## Minor omissions and limits

1. **README.md and retention-source.mjs are omitted from `manifest.json.files`.** The script writes/copies them after building the manifest. They are present and inspected; this is not lost evidence, but a supplemental binding or inclusion in a future manifest would close the integrity gap. This audit supplies their exact hashes below. `manifest.json` omitting itself is normal and its externally supplied hash matches.
2. The older matrix-document snapshots referenced by the Chat and source/reuse audits are not separately retained; only the newer controller snapshot is present. Those reports explicitly qualify their earlier matrix hashes and later updates. Native inputs still match their audited hashes, so this does not invalidate their bounded conclusions.
3. Raw application source, source tar and copied dependencies are deliberately not bundled. Historical source claims remain tied to the frozen revision/retained archive manifests; the five installed Next-file bindings depend on external copied dependencies for full offline reproduction. README accurately discloses that source/dependency data is not copied.
4. Controller contains earlier in-progress narrative followed by later outcome sections (for example, the then-pending five-card retry and Study controls). The table and later sections correctly record repeated Fail234 and Partial/Fail235. Treat those final outcomes as current; this audit does not promote earlier progress text into a pass.

## Qualification fidelity

README, controller and progress retain the partial checkpoint status, three unstarted configurations, unresolved 231–235, dependency/browser/model reuse and no clean-machine claim. The backend model-availability400 is explicitly distinguished from an upstream generation outage. Image guard evidence is not labeled vision acceptance, true-hidden visibility is unverified, truncated ingestion analysis is not certified, biology generation/five-card Study are failed/blocked, and re-rate preview235 remains a failure. Row 12 is only not-applicable to the single-user cell. No full matrix or all-green claim is present. The release gate's zero unresolved count is the historical 12:18 release state, not the later checkpoint outcome.

## Supplemental hashes

| File | SHA256 |
|---|---|
| `README.md` | `3bf980bb255cf474170c1d113c75d8d30e00e7e6142fcddd476390b219d802cc` |
| `retention-source.mjs` | `6cc8ac4ee8d0ff427b0c557fea16cbb12999e8828dcd3fdd6a57066597d414e1` |
| `manifest.json` | `e3403b75043675dff9abc878d594c26c9eecfb6f8125011489b60b7e3d102d6e` |
| `controller/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md` | `e8f60c16c02461d825b2b36d0a29d8c6e5c3f12fb8bacc452aa514d3c801c1c0` |
| `matrix-progress.json` | `a3685d71a330f7423129b17aab1cee16a443739501f405ea050de9b59630a451` |

All-204-file binding digest: `079f7272df225c754ce7347502e6c8fc30c1d5994ddbe9882064ad3e44e5ffaa`. Computed as SHA256 of compact, sorted-key JSON for path-sorted objects containing each relative `path`, `sha256`, and `bytes`.

Review completed 2026-09-17T13:35:51.369562+00:00. Only this review file was written. No runtime, browser, inference, database, source, task, tracker or git action was performed.
