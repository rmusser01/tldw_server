# UAT118 independent storage/listing review

## Status

**Final verdict: clear within this read subset.** The three confirmed findings below are resolved in the corrected endpoint, frozen at 2026-09-16T06:26:58.001485+00:00 with SHA256 `a43cd7606709d3ef1cbd886771fe871726551a036938a66182b50325a857850c`. Independent final verification passed24 tests, including all four unchanged private controls. No remaining actionable finding identified in these three paths.

Review began against `/private/tmp/cycle4-uat118-read-subset-freeze.json`, frozen 2026-09-16T06:18:02.285429+00:00. All three original hashes matched. Storage and schema remain byte-identical; the endpoint's authorized narrow correction matches `/private/tmp/cycle4-uat118-production-manifest.json`. Hash evidence is `/private/tmp/cycle4-uat118-read-independent-hashes.json`.

No repository edits, tracking updates, staging/commits, native browser operations, server runtime changes, or live inference were performed. Private probes use existing temporary SQLite/API fixtures and mocked providers; they do not call a model.

## Confirmed P2 findings — now resolved

1. **Empty legacy primary images are silently declared complete/image-free.** `character_messages.py:167` and `:647` use primary-blob truthiness to decide whether a legacy attachment exists. A stored zero-length BLOB with retained `image/png` MIME consequently bypasses validation. Actual authorized opt-in GET returns200 with `images:[]` and `has_image:false`. This violates the approved rule that corrupt/empty stored attachments fail the entire opt-in read instead of dropping them. Detect legacy presence separately from byte validity, including non-None bytes or retained MIME evidence; then reject empty/missing data. Preserve genuinely image-free rows where both columns are None.

2. **Missing MIME can validate as an unsupported MIME.** At `character_messages.py:175`, `_detect_image_mime_type(data) != mime` passes when both values are None. A legacy TIFF BLOB with a missing stored MIME is accepted by Pillow and serialized as `data:None;base64,...`; actual opt-in GET returns200. Require an explicit recognized, nonempty MIME before checking equality. No MIME inference or attachment substitution is needed.

3. **JPEG truncation bypasses `Image.verify()`.** At `character_messages.py:180–181`, Pillow's verification does not inspect JPEG compressed pixel data. A valid2×2 JPEG truncated by10 bytes still passes `verify()` and the actual opt-in endpoint returns200 with its broken bytes. Opening the same bytes and calling `load()` raises `OSError`. Complete attachment validation needs a bounded decode/format check, preserving existing byte and image-size safeguards. Add valid JPEG and truncated JPEG controls; do not relax MIME/byte equality.

The first two were reported to root/author immediately; the third was reported after the additional actual endpoint probe. These are automated corrupt/legacy-data findings, **not new native UAT observations**.

## Independent evidence

- Private probe: `/private/tmp/cycle4-uat118-read-independent-probe.py`.
- Initial endpoint negatives plus serializer compatibility: `/private/tmp/cycle4-uat118-read-independent-probe.log`: **2 failed / 1 passed**. Failures are the expected status>=400 assertions receiving200 for empty-primary and missing-mime.
- Added JPEG endpoint negative: `/private/tmp/cycle4-uat118-read-independent-jpeg.log`: **1 failed**, receiving200 with `data:image/jpeg;base64` for truncated bytes.
- Independent permanent selected tests: `/private/tmp/cycle4-uat118-read-independent-existing.log`: **15 passed / 12 deselected**. Includes both opt-in response branches, unchanged default/completion responses, actual owned paging, foreign/wrong-scope denial, strict read errors, corrupt/truncated/invalid-position/over-budget controls, SQLite snapshot, deletion, and existing failed-turn API controls selected by `opt_in`.
- Serializer private control passed: images absent from default Python/JSON dumps, explicit empty list retained, explicit image list retained. Existing individual/create/update responses therefore preserve omission through the shared response model.

Commands from repository root (project venv activated; Docker startup disabled):

```sh
source .venv/bin/activate && TLDW_TEST_NO_DOCKER=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py \
  -k 'opt_in or sqlite_strict or deleted_attachment' -q

source .venv/bin/activate && TLDW_TEST_NO_DOCKER=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py \
  /private/tmp/cycle4-uat118-read-independent-probe.py -k read_probe -q
```

## Storage/schema review conclusion

The strict query uses a single CTE statement for exact message-page selection, stored byte total, and CASE-gated ordered/legacy blob projection. It retains limit/offset and stable message sort with image position ascending, counts ordered blobs instead of their duplicate primary copy, and does not shorten the page. Above budget it suppresses the ordered join, returns null for both blob projections, and raises before returning a partial list. The independent SQLite cursor test checks actual returned columns and no follow-up image query. Position gaps fail the entire read. Existing DB errors propagate into the opt-in503 response instead of primary fallback. An endpoint decoded-byte recheck occurs before base64 expansion.

Authorization remains ahead of the new read: per-user DB dependency and existing owner/scope/deleted-conversation verification are unchanged. Default calls retain the prior DB path; completion format deliberately does not expand even with include_images=true. Both standard branches attach images only when requested. The shared Pydantic wrap serializer omits None and preserves explicitly requested arrays.

**No other actionable storage or schema finding identified.** A single PostgreSQL statement provides the required per-statement snapshot under READ COMMITTED; however this review did not execute PostgreSQL. The author's existing-fixture run with Docker startup disabled records a PostgreSQL skip. This does not establish whether a Docker-backed fixture could be available; root is separately checking that boundary. No PostgreSQL execution pass is claimed here, and no DB isolation or infrastructure change is requested.

## Final verification and limits

- Reviewed corrected helper: legacy attachment presence now checks non-None bytes/MIME; a detected recognized MIME is required before equality; Pillow verify is followed by actual pixel load, with its decompression-bomb error caught. The existing per-image byte limit is checked before decoder work. Both absent fields still represent a genuinely image-free message. No default/list authorization or cap change accompanied this correction.
- `/private/tmp/cycle4-uat118-read-independent-final.log`: **24 passed / 12 deselected**, exit0, in75.35s. Four unchanged private cases (empty-primary, missing-mime, truncated-JPEG, serializer) all passed alongside20 selected permanent controls, including valid JPEG and null-primary cases. Routine suite warnings and a pre-existing pytest temporary-directory cleanup warning were emitted; there were no test failures.
- Final selection command adds `-k 'read_probe or opt_in or sqlite_strict or deleted_attachment or legacy_image_read or valid_jpeg'` to the combined permanent/private command above.

Parent independently owns the remaining frontend/retry/service review and native verification. This report does not certify those paths, real PostgreSQL execution, or successful vision inference. No new whole-suite/static/security claim is made by this read-only review; the author/root own their full verification.
