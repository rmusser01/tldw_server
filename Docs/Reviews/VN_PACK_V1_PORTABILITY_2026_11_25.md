# VN pack v1 portability recheck

This recheck covers [GitHub #2026](https://github.com/rmusser01/tldw_server/issues/2026) and TASK-15667 on the November 25, 2026 remote `dev` base `1d8589b7c43e5dbb6fca3247e023f562129c70f3`. Both loopback FastAPI instances ran the normal VN Jobs worker and used separate AuthNZ, Jobs, user data and staging directories. The source used a single-user API key. The destination used real JWT accounts, including an unrelated user and a user whose VN pack list started empty. Chromium exercised the rebuilt Next.js workbench at 1280×720.

## Live results

| Case | Observed result |
| --- | --- |
| Export and trusted restore | Strict export contained the approved 32×32 PNG. Its SHA-256 was `9684a9867b49f0f3144f2bf0cdcec2a2cedaf3d12c03861037562bb188ede631`. Destination job 7 created pack 2 for owner 8 from source pack 3 for owner 1. The destination character, item and generated file were resolved locally. The title, scenario, slot labels, exact image bytes, approval and preferred status survived. The destination's existing asset was unchanged. A different account received 404 for the imported pack and its job. |
| Repeated commit and conflicts | Resending the same key and body returned job 7. Reusing the key with a different trust mode returned HTTP 409 and `idempotency_key_conflict`. Reimporting the included character under a new key failed on the duplicate name before creating a pack. Linking the existing character allowed a new import. Its PNG matched the source and the item was draft and unpreferred in untrusted mode. |
| Restore into an empty workbench | Chromium previewed the backup and submitted the offered included-character import. Job 19 completed and the new pack appeared in the previously empty account after reload. Switching to the other test account cleared the previous account’s preview. [Completed preview before commit](assets/2026-11-25-vnpack-portability/empty-destination-preview.png) |
| Corrupt archive | An incomplete download was quarantined with `invalid_archive: bad_zip_file`. The API refused commit with HTTP 409. Deleting the preview removed its staged file. The rebuilt workbench showed a readable explanation and kept Commit disabled. [Quarantined preview](assets/2026-11-25-vnpack-portability/quarantined-archive.png) |
| Missing image | Strict export reported `missing_asset_bytes`. The regular backup recorded a missing-image warning. Import produced one metadata item and zero generated files. The item stayed out of the visible review list. |
| Storage quota | With `USAGE_QUOTAS_ENABLED=1` and a zero-byte destination quota, import failed preflight with `Storage quota exceeded`. The pack list was identical before and after the attempt. |
| Cancelled preview | The API reported the preview cancelled. Deleting it returned HTTP 204 and removed the staged archive. The focused suite separately covers commit refusal for an expired preview. |

## Verify the v1 workflow

1. Start two isolated installations with `VN_ASSET_JOBS_WORKER_ENABLED=1`. Set an independent `DATABASE_URL`, `JOBS_DB_PATH`, `USER_DB_BASE_DIR` and preview/export staging root for each. Set `USAGE_QUOTAS_ENABLED=1` to exercise quota preflight.
2. Create a character and a pack on the source. Create a sprite slot, upload a small PNG through `POST /packs/{pack_id}/items/upload` and approve it. Start `POST /packs/{pack_id}/export` with `strict: true`, `include_character_payload: true` and a fresh idempotency key. Poll the export job and download its completed archive.
3. Upload the archive to `POST /import/previews` on the destination. Poll the preview to completion and use its character actions and update plan. Commit a `create_new` import with `import_included_character` for an empty account. If the character name already exists, use `link_existing_character` with a chosen local character. Use a new idempotency key when changing the requested action. In the workbench, select a pack bound to the local character to link it.
4. Poll the import job, then compare the API pack, slots, items and downloaded PNG with the source. Repeat with `untrusted_import` to check the review reset. In the workbench, reload to see a completed import in Packs.
5. Repeat preview with an incomplete archive to verify quarantine. Use `DELETE /import/previews/{preview_id}` after failure or cancellation to remove staged bytes. Apply a zero-byte quota on the disposable destination to verify that quota failure creates no pack.

The portability archive schema is `tldw.vnpack.v1`. A runtime manifest has its own `vn_asset_manifest.v1` format. The importer accepts the v1 archive and reports `unsupported_vnpack_schema` for a different version. Existing v1 backups can record missing assets, while strict export requires their bytes. Multi-character archive compatibility follows the schema decisions in [#2022](https://github.com/rmusser01/tldw_server/issues/2022) and [#2024](https://github.com/rmusser01/tldw_server/issues/2024). See the [VN Asset Packs API](../API-related/VN_ASSET_PACKS_API.md) for the canonical route prefix and request fields.

## Worker recovery and remaining work

Tests interrupt completion delivery after the domain journal commits. Redelivery returns the original pack and ID maps even after the preview archive has been removed. The tests also cover a transaction readback failure, stale status writes, a previously downgraded create-new receipt, update-existing result replay and mismatched job binding.

TASK-15667.1 tracks recovery before journal completion. The current importer can commit a pack, then register files and items across separate stores. A stop during those writes can leave partial state, and replay can allocate another pack or encounter a changed update plan. The live recheck verified explicit preview deletion. Automatic expiry cleanup and cleanup after mid-write interruption remain part of that follow-up.

TASK-15667.2 tracks terminal import/export status, completed downloads and retry controls in the workbench. The browser test checked job completion through the existing status API and then reloaded the workbench. The issue remains open for these follow-ups and richer-format compatibility.

This work uses existing Jobs and domain publication contracts under [ADR-003](../ADR/003-jobs-vs-scheduler-default.md) and [ADR-094](../ADR/094-jobs-resource-admission-and-domain-publication.md).

## Verification

- Pytest passed all 89 cases in the six VN portability suites. Vitest passed all 12 focused portability panel and API client cases. Scoped ESLint, Python compilation, and Bandit passed. Bandit reported zero findings and errors.
- The broader VN frontend run passed 371 cases and timed out in two workbench storage recovery cases. Both timeouts reproduced with the original components restored temporarily in the same worktree. The two cases passed when run in isolation.
- The broader backend run stopped after setup errors. The replay I/O boundary fixture also hit the same blocked external DNS lookup with the backend files from the base revision.
