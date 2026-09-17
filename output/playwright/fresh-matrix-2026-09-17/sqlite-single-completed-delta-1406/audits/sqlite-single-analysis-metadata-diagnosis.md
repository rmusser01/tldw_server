# SQLite single — analysis save and version metadata

## Disposition

**The captured save does not destroy the original metadata.** The POST 201 and subsequent GET 200 both retain the full original `safe_metadata` under version 1. Only the newly created version 2 has null metadata; `processing.safe_metadata` intentionally projects the latest version according to the documented response schema. This observation alone does not establish a new source-metadata-loss defect.

Frozen source revision: `8f8774e6c868b304a96d95ab82e28389c129a78b`; run `fresh-final-20260917`. Read-only diagnosis under the existing parent UAT task; no task state changed.

## Native evidence (2026-09-17 UTC)

- Last pre-save GET `/api/v1/media/1` at **13:44:41.572**, HTTP 200: latest version 1 UUID `736f4de9-63de-4f39-99a7-611a8317b332`, with `title`, `author`, and `chunking_plan` in both its version metadata and `processing.safe_metadata`.
- POST `/api/v1/media/1/versions` at **13:45:03.290** sends exactly `content`, `analysis_content`, and `prompt`; no `safe_metadata`. Content has 1,914 characters. Persisted analysis is `LIVE_TIER_ANALYSIS_ONE`.
- POST response **13:45:03.323**, HTTP 201, introduces version 2 UUID `a2376f20-9981-4480-bb59-4bfdf32cb9e6` with null metadata. Its `versions` array also includes original version 1 with the full prior metadata.
- Following GET **13:45:03.340**, HTTP 200, has the same two version records: current metadata null, original metadata intact. This is canonical API readback already present in the supplied capture, not a new request.
- Programmatic comparison of supplied response bodies confirms the before/after `source` objects and `content` objects are identical. Source-text SHA-256: `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`.
- Original metadata equals the post-save version-1 metadata structurally. Canonical JSON SHA-256 (sorted keys, compact separators, UTF-8): `6f61e94eae888afc666c70ced3a9a3c23a590a12ae1cccbfea3223ea5ec27f1f`.

## Source explanation

Paths below are relative to the frozen source root.

1. `apps/packages/ui/src/components/Media/AnalysisModal.tsx:377–390` saves analysis by creating a version with source text, analysis and prompt. It omits metadata, consistent with the observed request.
2. `media_request_models.py:82–88` defines `VersionCreateRequest.safe_metadata` as optional, default `None`, described as metadata “to store with this version.” The endpoint `media/versions.py:342–367` serializes it only if supplied and otherwise passes `None` to `create_document_version`.
3. Runtime and package API delegate to `DocumentVersionsRepository.create`. Its lines 59–105 allocate the next version and **insert** a new `DocumentVersions` row with the supplied metadata. There is no update clearing previous version metadata in this path. The captured version history independently confirms preservation.
4. `media_details_service.py:274–293` retrieves the latest version and places only that version's metadata in `processing.safe_metadata`. `media_response_models.py:134` explicitly documents this field as metadata associated with the **latest document version**; line 163 similarly defines per-version metadata. It is not defined as a union or fallback to prior source metadata.

The null projection therefore follows the current version-storage contract. It does not mean `source.title` disappeared, the source text was replaced by analysis, chunking was undone, or historical metadata was deleted. No DB inspection or inference about hidden storage is required to establish the retained version-1 response.

## Existing coverage / historical scope

Existing version tests cover creating and retaining multiple versions, listing and retrieving a specific version, and rollback. The inspected tests do not establish an analysis-specific rule that omitted metadata must be inherited. No such carry-forward promise was found in the bounded source/schema/tracker review.

Prior UAT105/TASK-13260.46 concerns raw provider envelopes being stored as successful analysis; UAT115/TASK-13260.55 concerns selected model persistence. Neither matches this observation. TASK-13139.8 discusses future bounded retrieval provenance through Media/versioning, but does not establish that this analysis-only save must merge the prior chunking metadata into every version.

**Recommendation:** retain this as explained version projection, not proven data loss. If a current-version metadata consumer demonstrably requires original ingestion provenance after analysis, evaluate that precise workflow separately and define inheritance versus explicit replacement/null semantics before changing version creation. The supplied evidence does not show that additional failure and does not justify a blanket merge or fallback.

## Limits and integrity

Only this ignored audit report was written. No tests, provider inference, browser/runtime/configuration/DB operations, product edits or task changes. No raw stream body was reproduced. This is not a broader media-versioning acceptance claim.

All 13 inspected source/history files below match their original frozen archive entries.

| Packet-relative input | SHA-256 |
| --- | --- |
| `copy-preparation/sqlite-single-archive-manifest.json` | `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1` |
| `native/sqlite-single/analysis-one-result.txt` | `88650bee01ed309bfa64ffe850d2d833b3dcd437083f556762dc478230620283` |
| `native/sqlite-single/analysis-one-settled.txt` | `4e1af7e33661d562375acdbd64ea106ea31aa137741125935904d3277d74c34a` |

| Frozen-root-relative source/history | SHA-256 |
| --- | --- |
| `apps/packages/ui/src/components/Media/AnalysisModal.tsx` | `01e61da2ba7ad5df6397d269e5da83d1a559e26c79e44a8c13f05cd2666110ca` |
| `tldw_Server_API/app/api/v1/endpoints/media/versions.py` | `26f8c6330882354e05d75b9bb5dee4839db8b9dbfdac11451000af9080585677` |
| `tldw_Server_API/app/api/v1/schemas/media_request_models.py` | `3ed44eff76417e236c829a45cac7b644d47eafeff3296c89a6c19eb2272d38c5` |
| `tldw_Server_API/app/api/v1/schemas/media_response_models.py` | `cf0b2a6604bc198e136e10fc36b75d390e71f0ae31a7996eb55729df34e37e76` |
| `tldw_Server_API/app/core/DB_Management/media_db/api.py` | `12481e48b89cbf4e1b81b41b5f90fac47d2589f58da69f4ce67339ebae1887e8` |
| `tldw_Server_API/app/core/DB_Management/media_db/runtime/document_keyword_ops.py` | `5c0766b42c1bddb3ab430d788a65069df16e5457b3d88d757250557a7487c21f` |
| `tldw_Server_API/app/core/DB_Management/media_db/repositories/document_versions_repository.py` | `5be4ab101e4236cab9b3790f3914a46c6cd637fbd6511b51ccad612238c3e50e` |
| `tldw_Server_API/app/core/DB_Management/media_db/services/media_details_service.py` | `f72ae82248207b6fbe1a7c10fafacebdc6896e48677b8a210e497c2391bada03` |
| `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_versions_integration.py` | `bedfd0c37c88cedd791fa40395ab0b2aa96f4327ff0fd21a55e0baec56a35172` |
| `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py` | `e2df987d3a9151d98de3e4fc4fa543539f332d3ded641ab38d89d13966a2fda6` |
| `backlog/tasks/task-13260.46 - Reject-raw-provider-envelopes-as-successful-media-analysis.md` | `3a322a9f5a4c978d5915ff203dcfa52a230687268e69d9e1fd4be20f7dbdea46` |
| `backlog/tasks/task-13260.55 - Preserve-the-model-explicitly-selected-for-Media-analysis.md` | `4ef534fce9e6538805108956b05ec473c7431c542cf9d98978f53fc625ac321d` |
| `backlog/tasks/task-13139.8 - Persist-selected-retrieval-provenance-through-Media-and-Chatbooks.md` | `a53118fbe53629681bfd8e26686074f12019897674b84e65c64020c345d0819d` |
