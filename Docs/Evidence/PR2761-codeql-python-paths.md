# PR 2761 Python path-injection evidence

Tracking: TASK-13013.3.1. This records the 152 supplied open Python path alerts individually. No alert was dismissed or suppressed in this work.

SARIF source: `43165c8c82f6df9bab56f425879ac286aecbc5ee`; initially inspected worktree: `28797892b1`. All 152 IDs matched exact SARIF results. Between these commits the only affected sink-file change is Buddy artwork metadata/import additions, not its path checks. New uncommitted repairs are described below.

SARIF SHA-256: `d7d10391add7ed40f58895e31af3222b04ceea8d56204b2e781f71cd568d0bd8`.

## Repairs and validation

Snapshot archive and metadata leaves now use existing `safe_join`; listing and quota enumeration apply the same validation to discovered leaves. Previously a pre-existing symlink could supply an outside archive or disclose outside JSON. Six regression cases failed before repair. Restore/security and quota suites pass **31 tests**. This assumes trusted storage roots and does not eliminate concurrent replacement races.

Research artifacts now confine the fixed `research` child under the supplied output root before creating session directories. Previously a link at that child could make the resolved outside location the effective trusted root. The added root-link regression failed before repair; the Research suite passes **12 tests**, including 10 added tests documenting already-existing filename rejection, session hashing and session/file symlink containment. The public artifact filenames and storage layout are preserved.

Validation results: core DB/storage/RAG **166 passed**; Notes/Chatbooks/FileArtifacts/Buddy **63 passed, 1 skipped**; Skills/sandbox/audio boundary suites **153 passed**; full audio transcription plus Research before the final root-link test **76 passed**. Ruff passes for changed source/tests; Bandit reports zero findings on each changed production file. Logs are named per boundary below. Routine existing pytest/configuration and temporary-directory cleanup warnings occurred. The skipped test is not counted as evidence of passing behavior.

These counts overlap where a suite was rerun. They are not a sum of unique repository tests or a clean full-repository verdict.

## Boundary definitions

### audit: Guarded user directory

All 21 traces carry the media reprocess user identifier through ChromaDB_Library and audit_adapter to Audit_DB_Deps. db_path_utils._normalize_user_id accepts only positive digit strings in production; test-only names are restricted or SHA-256 encoded. _build_user_dir then uses Utils.path_utils.safe_join, whose realpath/commonpath check rejects escapes and whose component walk rejects symlinks. Audit paths append fixed filenames to that user directory. The later fallback-queue relative_to checks do not independently establish root trust: the upstream user-root boundary does. HTTP input cannot choose audit storage or lock filenames.

Coverage: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py; /tmp/pr2761-path-core-tests.log`.

### personalization: Guarded user directory

RAG analytics user_id flows through DatabasePaths.get_user_base_directory: positive numeric production IDs, test-only restricted/hash identifiers, and _build_user_dir/safe_join realpath containment and symlink rejection. UserPersonalizationStore appends only rag_personalization.json and checks the resolved file relative_to its user base before load/save. No request path is accepted as a filename.

Coverage: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py; tldw_Server_API/tests/RAG/test_user_personalization_store.py; /tmp/pr2761-path-core-tests.log`.

### baseline: Guarded baseline identifier

RegressionDetector._get_baseline_path checks _SAFE_ID_RE before joining <configured baseline_dir>/<baseline_id>.json, resolves the candidate, and requires relative_to(baseline_dir). Traversal/absolute IDs fail validation; an existing symlink outside the configured root fails containment. The flagged operation is resolve itself, not an unguarded read.

Coverage: `tldw_Server_API/tests/RAG/test_regression.py::TestPathTraversalAndInvalidIDs; /tmp/pr2761-path-core-tests.log`.

### checkpoint: Guarded checkpoint path

CheckpointManager.load_by_id validates the ID and _resolve_checkpoint_path resolves absolute or relative paths then requires relative_to(checkpoint_dir). The legitimate direct checkpoint API supports absolute paths inside that configured directory. The two flagged resolve calls occur during validation and cannot reach loading with an outside path.

Coverage: `tldw_Server_API/tests/RAG/test_checkpoint.py; /tmp/pr2761-path-core-tests.log`.

### buddy: Owned fixed artwork identity

BuddyService.read_asset first requires the buddy in its per-user repository and finds asset_id only among that buddy assets. _asset_path restricts both IDs to exactly32 lowercase hex characters, chooses an extension from VISUAL_MIME_EXTENSIONS, resolves under the authenticated user root and requires is_relative_to(user_root). Reads are bounded and require stored size plus SHA-256 equality. Neither supplied ID can become a separator or outside path.

Coverage: `tldw_Server_API/tests/Persona/test_independent_buddies.py; /tmp/pr2761-path-apps-tests.log`.

### storage: Guarded media storage root

The SARIF source is the media ingestion authenticated user identifier used to construct the backend base_path, not an arbitrary backend root from the request. DatabasePaths.resolve_user_base_directory normalizes the user ID then uses safe_join under configured USER_DB_BASE_DIR. FileSystemStorage._validate_path resolves every candidate and requires relative_to(self.base_path); delete invokes this before removal, and cleanup independently checks each resolved parent. Read/write filenames are sanitized; base paths intentionally remain configurable by the server.

Coverage: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py; tldw_Server_API/tests/Storage/test_filesystem_storage.py; /tmp/pr2761-path-core-tests.log`.

### notes: Guarded attachment directory

Request user IDs reach DatabasePaths.get_user_base_directory, positive numeric production-ID normalization and safe_join. notes._get_note_attachments_base_dir resolves the fixed notes_attachments child and checks relative_to(user_root) before mkdir; _get_note_attachments_dir sanitizes note IDs through safe_legacy_note_attachment_dirname, resolves and checks relative_to(base_dir). The flagged resolve operations are part of the check. The legacy-helper symlink-escape regression verifies the route fails closed.

Coverage: `tldw_Server_API/tests/Notes/test_legacy_attachment_source.py::test_legacy_route_helper_rejects_attachment_root_symlink_escape; /tmp/pr2761-path-apps-tests.log`.

### legacy: Owned descriptor-relative attachment access

The owner root comes from DatabasePaths.resolve_user_base_directory, including normalized user ID and safe_join; LegacyAttachmentSource requires owner_user_id==note_db.client_id and owns_note_id before listing. Note names are sanitized. _open_directory requires O_NOFOLLOW|O_DIRECTORY|O_NONBLOCK, with child opens relative to already-open directory descriptors. Regular files also use O_NOFOLLOW and fstat identity checks. The preliminary exists probe does not open file contents; linked directories/files fail the subsequent secure open.

Coverage: `tldw_Server_API/tests/Notes/test_legacy_attachment_source.py::test_symlinked_note_directory_and_candidate_fail_closed; /tmp/pr2761-path-apps-tests.log`.

### exports: Guarded generated export filename

ChatbookService._build_export_filename maps every non-alphanumeric/non-underscore/non-hyphen character to underscore, limits length, and appends server-generated timestamp and UUID suffix. _resolve_export_path resolves under the per-user export_dir and requires relative_to(base). Its returned path is used by archive generation, size verification and error cleanup. Continuation exports reuse this boundary. User supplied display names cannot select outside files.

Coverage: `tldw_Server_API/tests/Chatbooks/test_chatbooks_sanitizers.py; tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py; tldw_Server_API/tests/Chatbooks/test_chatbook_service_continuation.py; /tmp/pr2761-path-apps-tests.log`.

### fileartifacts: Guarded generated temporary output

Export filename is file_<database integer ID>.<validated export format>. CollectionsDB.resolve_temp_output_storage_path calls normalize_output_storage_filename with absolute paths forbidden, separators forbidden and resolved containment enabled. _write_export_file then independently resolves against DatabasePaths.get_user_temp_outputs_dir and checks relative_to(outputs_dir) before aiofiles.open. The authenticated user root is normalized separately.

Coverage: `tldw_Server_API/tests/FileArtifacts/test_file_artifacts_service_exports.py; /tmp/pr2761-path-apps-tests.log`.

### research: Research output boundary strengthened

User root originates in DatabasePaths.get_user_temp_outputs_dir: numeric production IDs plus safe_join. Session IDs become fixed SHA-256 directory components; artifact names pass normalize_output_storage_filename; resolved session/file candidates require relative_to their roots. New repair confines the fixed research child with safe_join before using it as a root: previously a pre-existing research-directory symlink could redirect writes outside supplied base_dir. The attack requires filesystem link creation, not just an HTTP string. Base_dir itself remains an intentionally supplied trusted service/configuration root.

Coverage: `tldw_Server_API/tests/Research/test_research_artifact_store.py; /tmp/pr2761-path-research-red.log; /tmp/pr2761-path-research-green.log`.

### skills: Validated Skills names and confined bundle lifecycle

Skills endpoint names pass _normalize_and_validate_skill_name (trim/lowercase plus SKILL_NAME_PATTERN). _get_skill_dir resolves and checks relative_to(skills_dir). Trash archive IDs have a strict alphanumeric/underscore/hyphen allowlist and lstat rejects links; cleanup validates exact parent, regular directory type and no symlink. _move_skill_dir checks both resolved paths under skills root, source lstat and absent destination. Staging names combine validated skill name, fixed operation and generated time. Supporting filenames pass _safe_supporting_path; main files require containment. fd-walk reads use no-follow opens and stat identity checks. The readonly-cleanup callback uses only shutil-provided subtree entries and rejects links before chmod. These are intentional per-user filesystem operations, not arbitrary path acceptance.

Coverage: `tldw_Server_API/tests/Skills/unit/test_skills_service.py (supporting-file traversal, symlinked trash/archive/cleanup, symlinked main file, valid create/delete/restore and integrity cases); /tmp/pr2761-path-skills-tests.log`.

### sandbox: Owned stored workspace confined to sandbox root

The endpoint checks _require_session_owner before service operations. Workspace paths come from stored session rows or validated owner/session path components; _resolve_workspace_path_from_store canonicalizes and requires relative_to(_workspace_root()) plus an exact workspace leaf. _workspace_path and legacy fallback also enforce root containment. SnapshotManager rejects workspace root/ancestor/tree links and archive traversal/link/device members. Backup, restore, clone and cleanup intentionally act on this owned workspace. Session snapshot directories use strict legacy component validation or SHA-256 names and resolved root containment. Metadata/archive leaves additionally have the new safe_join guard.

Coverage: `tldw_Server_API/tests/sandbox/test_orchestrator_artifact_security.py; tldw_Server_API/tests/sandbox/test_snapshot_manager_restore_security.py; tldw_Server_API/tests/sandbox/test_snapshot_quota_enforcement.py; /tmp/pr2761-path-skills-tests.log; /tmp/pr2761-path-snapshot-green.log`.

### audio_cleanup: Confined ingestion input and cleanup

The traced local input passes resolve_safe_local_path against the per-request processing temporary directory before copying/transcribing. That helper compares canonical commonpath to the allowed root. Cleanup may perform exists/is_file/resolve metadata probes before its final check, but unlink executes only when the canonical candidate is_relative_to the canonical processing temporary root. The legacy startswith fallback is unreachable on the supported Python runtime (is_relative_to exists). No file contents are read and no file is removed outside that root by these flagged cleanup operations.

Coverage: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_safe_paths.py::test_process_audio_files_rejects_local_path_outside_temp_dir; /tmp/pr2761-path-skills-tests.log`.

### audio: Guarded transcription input/model

Provider input/conversion helpers use resolve_safe_local_path and/or _resolve_safe_input_path against their allowed base before using output paths; local model identifiers are checked against WHISPER_MODEL_BASE_DIR and no-link policy. validate_whisper_model_identifier checks required artifact files only for validated local directories. Existing CWD-directory detection intentionally performs is_dir before distinguishing Hub IDs/aliases from local directories, then rejects local directories outside the model root. That metadata probe is distinguished below from guarded content access.

Coverage: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_transcription.py (outside-base input/model, absolute/relative symlink, CWD local precedence, remote-ID/alias preservation); /tmp/pr2761-path-audio-research.log`.

## Individual alert dispositions

Each source and sink below is from the exact SARIF analysis, before the new repairs. `guarded` means the traced request value passes the named boundary; it is a false-positive candidate for explicit review, not a claim that every filesystem operation is race-free. `repaired` means this work strengthens a boundary on that path and awaits a new scan/review. `probe review` requires a deliberate metadata-access disposition.

| Alert | Source | Sink | Boundary | Disposition |
|---|---|---|---|---|
| [2104](https://github.com/rmusser01/tldw_server/security/code-scanning/2104) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:502` | audio | guarded |
| [2132](https://github.com/rmusser01/tldw_server/security/code-scanning/2132) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:148` | audit | guarded |
| [2134](https://github.com/rmusser01/tldw_server/security/code-scanning/2134) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:2808` | audit | guarded |
| [2135](https://github.com/rmusser01/tldw_server/security/code-scanning/2135) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3063` | audit | guarded |
| [2136](https://github.com/rmusser01/tldw_server/security/code-scanning/2136) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3064` | audit | guarded |
| [2137](https://github.com/rmusser01/tldw_server/security/code-scanning/2137) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3065` | audit | guarded |
| [2138](https://github.com/rmusser01/tldw_server/security/code-scanning/2138) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:2825` | audit | guarded |
| [2139](https://github.com/rmusser01/tldw_server/security/code-scanning/2139) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3145` | audit | guarded |
| [2140](https://github.com/rmusser01/tldw_server/security/code-scanning/2140) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3146` | audit | guarded |
| [2141](https://github.com/rmusser01/tldw_server/security/code-scanning/2141) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3150` | audit | guarded |
| [2142](https://github.com/rmusser01/tldw_server/security/code-scanning/2142) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3151` | audit | guarded |
| [2143](https://github.com/rmusser01/tldw_server/security/code-scanning/2143) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3161` | audit | guarded |
| [2144](https://github.com/rmusser01/tldw_server/security/code-scanning/2144) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3160` | audit | guarded |
| [2145](https://github.com/rmusser01/tldw_server/security/code-scanning/2145) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3155` | audit | guarded |
| [2146](https://github.com/rmusser01/tldw_server/security/code-scanning/2146) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3157` | audit | guarded |
| [2147](https://github.com/rmusser01/tldw_server/security/code-scanning/2147) | `api/v1/endpoints/rag_unified.py:1572` | `core/RAG/rag_service/user_personalization_store.py:75` | personalization | guarded |
| [2149](https://github.com/rmusser01/tldw_server/security/code-scanning/2149) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:78` | research | repaired |
| [2150](https://github.com/rmusser01/tldw_server/security/code-scanning/2150) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:80` | research | repaired |
| [2151](https://github.com/rmusser01/tldw_server/security/code-scanning/2151) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:81` | research | repaired |
| [2152](https://github.com/rmusser01/tldw_server/security/code-scanning/2152) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:81` | research | repaired |
| [2154](https://github.com/rmusser01/tldw_server/security/code-scanning/2154) | `api/v1/endpoints/chatbooks.py:687` | `core/Chatbooks/chatbook_service.py:2308` | exports | guarded |
| [2155](https://github.com/rmusser01/tldw_server/security/code-scanning/2155) | `api/v1/endpoints/chatbooks.py:687` | `core/Chatbooks/chatbook_service.py:2317` | exports | guarded |
| [2157](https://github.com/rmusser01/tldw_server/security/code-scanning/2157) | `api/v1/endpoints/chatbooks.py:527` | `core/Chatbooks/chatbook_service.py:2796` | exports | guarded |
| [2162](https://github.com/rmusser01/tldw_server/security/code-scanning/2162) | `api/v1/endpoints/files.py:125` | `core/File_Artifacts/file_artifacts_service.py:501` | fileartifacts | guarded |
| [2163](https://github.com/rmusser01/tldw_server/security/code-scanning/2163) | `api/v1/endpoints/media/add.py:63` | `core/Storage/filesystem_storage.py:94` | storage | guarded |
| [2166](https://github.com/rmusser01/tldw_server/security/code-scanning/2166) | `api/v1/endpoints/sandbox.py:717` | `core/Sandbox/orchestrator.py:1570` | sandbox | guarded |
| [2167](https://github.com/rmusser01/tldw_server/security/code-scanning/2167) | `api/v1/endpoints/sandbox.py:717` | `core/Sandbox/orchestrator.py:1579` | sandbox | guarded |
| [2172](https://github.com/rmusser01/tldw_server/security/code-scanning/2172) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1026` | skills | guarded |
| [2173](https://github.com/rmusser01/tldw_server/security/code-scanning/2173) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1029` | skills | guarded |
| [2174](https://github.com/rmusser01/tldw_server/security/code-scanning/2174) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1035` | skills | guarded |
| [2175](https://github.com/rmusser01/tldw_server/security/code-scanning/2175) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1069` | skills | guarded |
| [2176](https://github.com/rmusser01/tldw_server/security/code-scanning/2176) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1338` | skills | guarded |
| [2177](https://github.com/rmusser01/tldw_server/security/code-scanning/2177) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1339` | skills | guarded |
| [2186](https://github.com/rmusser01/tldw_server/security/code-scanning/2186) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:180` | sandbox | guarded |
| [2187](https://github.com/rmusser01/tldw_server/security/code-scanning/2187) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:181` | sandbox | guarded |
| [2188](https://github.com/rmusser01/tldw_server/security/code-scanning/2188) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:191` | sandbox | guarded |
| [2189](https://github.com/rmusser01/tldw_server/security/code-scanning/2189) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:193` | sandbox | guarded |
| [2190](https://github.com/rmusser01/tldw_server/security/code-scanning/2190) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:194` | sandbox | guarded |
| [2191](https://github.com/rmusser01/tldw_server/security/code-scanning/2191) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:196` | sandbox | guarded |
| [2192](https://github.com/rmusser01/tldw_server/security/code-scanning/2192) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:201` | sandbox | guarded |
| [2201](https://github.com/rmusser01/tldw_server/security/code-scanning/2201) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:326` | sandbox | guarded |
| [2202](https://github.com/rmusser01/tldw_server/security/code-scanning/2202) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:327` | sandbox | guarded |
| [2220](https://github.com/rmusser01/tldw_server/security/code-scanning/2220) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:359` | sandbox | guarded |
| [2227](https://github.com/rmusser01/tldw_server/security/code-scanning/2227) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:375` | sandbox | guarded |
| [2234](https://github.com/rmusser01/tldw_server/security/code-scanning/2234) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:384` | sandbox | guarded |
| [2235](https://github.com/rmusser01/tldw_server/security/code-scanning/2235) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:388` | sandbox | guarded |
| [2265](https://github.com/rmusser01/tldw_server/security/code-scanning/2265) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:163` | audit | guarded |
| [2266](https://github.com/rmusser01/tldw_server/security/code-scanning/2266) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:164` | audit | guarded |
| [2267](https://github.com/rmusser01/tldw_server/security/code-scanning/2267) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:167` | audit | guarded |
| [2268](https://github.com/rmusser01/tldw_server/security/code-scanning/2268) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:1022` | audit | guarded |
| [2269](https://github.com/rmusser01/tldw_server/security/code-scanning/2269) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:2633` | audit | guarded |
| [2270](https://github.com/rmusser01/tldw_server/security/code-scanning/2270) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:2634` | audit | guarded |
| [2271](https://github.com/rmusser01/tldw_server/security/code-scanning/2271) | `api/v1/endpoints/media/reprocess.py:204` | `core/Audit/unified_audit_service.py:3157` | audit | guarded |
| [2272](https://github.com/rmusser01/tldw_server/security/code-scanning/2272) | `api/v1/endpoints/rag_unified.py:1572` | `core/RAG/rag_service/user_personalization_store.py:42` | personalization | guarded |
| [2273](https://github.com/rmusser01/tldw_server/security/code-scanning/2273) | `api/v1/endpoints/rag_unified.py:1572` | `core/RAG/rag_service/user_personalization_store.py:43` | personalization | guarded |
| [2274](https://github.com/rmusser01/tldw_server/security/code-scanning/2274) | `api/v1/endpoints/rag_unified.py:1572` | `core/RAG/rag_service/user_personalization_store.py:48` | personalization | guarded |
| [2275](https://github.com/rmusser01/tldw_server/security/code-scanning/2275) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:429` | audio | guarded |
| [2276](https://github.com/rmusser01/tldw_server/security/code-scanning/2276) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:22` | research | guarded |
| [2277](https://github.com/rmusser01/tldw_server/security/code-scanning/2277) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:33` | research | repaired |
| [2278](https://github.com/rmusser01/tldw_server/security/code-scanning/2278) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:34` | research | repaired |
| [2279](https://github.com/rmusser01/tldw_server/security/code-scanning/2279) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:39` | research | repaired |
| [2280](https://github.com/rmusser01/tldw_server/security/code-scanning/2280) | `api/v1/endpoints/research_runs.py:416` | `core/Research/artifact_store.py:40` | research | repaired |
| [2281](https://github.com/rmusser01/tldw_server/security/code-scanning/2281) | `api/v1/endpoints/rag_unified.py:1928` | `core/RAG/rag_service/checkpoint.py:425` | checkpoint | guarded |
| [2282](https://github.com/rmusser01/tldw_server/security/code-scanning/2282) | `api/v1/endpoints/rag_unified.py:1928` | `core/RAG/rag_service/checkpoint.py:427` | checkpoint | guarded |
| [2283](https://github.com/rmusser01/tldw_server/security/code-scanning/2283) | `api/v1/endpoints/chatbooks.py:527` | `core/Chatbooks/chatbook_service.py:611` | exports | guarded |
| [2284](https://github.com/rmusser01/tldw_server/security/code-scanning/2284) | `api/v1/endpoints/chatbooks.py:527` | `core/Chatbooks/chatbook_service.py:2512` | exports | guarded |
| [2285](https://github.com/rmusser01/tldw_server/security/code-scanning/2285) | `api/v1/endpoints/chatbooks.py:527` | `core/Chatbooks/chatbook_service.py:2569` | exports | guarded |
| [2286](https://github.com/rmusser01/tldw_server/security/code-scanning/2286) | `api/v1/endpoints/files.py:125` | `core/File_Artifacts/file_artifacts_service.py:496` | fileartifacts | guarded |
| [2287](https://github.com/rmusser01/tldw_server/security/code-scanning/2287) | `api/v1/endpoints/sandbox.py:717` | `core/Sandbox/orchestrator.py:1534` | sandbox | guarded |
| [2288](https://github.com/rmusser01/tldw_server/security/code-scanning/2288) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/orchestrator.py:1545` | sandbox | guarded |
| [2289](https://github.com/rmusser01/tldw_server/security/code-scanning/2289) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/orchestrator.py:1557` | sandbox | guarded |
| [2290](https://github.com/rmusser01/tldw_server/security/code-scanning/2290) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/orchestrator.py:1611` | sandbox | guarded |
| [2291](https://github.com/rmusser01/tldw_server/security/code-scanning/2291) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/orchestrator.py:1621` | sandbox | guarded |
| [2292](https://github.com/rmusser01/tldw_server/security/code-scanning/2292) | `api/v1/endpoints/rag_health.py:512` | `core/RAG/rag_service/regression.py:397` | baseline | guarded |
| [2293](https://github.com/rmusser01/tldw_server/security/code-scanning/2293) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/service.py:2661` | sandbox | guarded |
| [2294](https://github.com/rmusser01/tldw_server/security/code-scanning/2294) | `api/v1/endpoints/skills.py:438` | `core/Skills/skills_service.py:344` | skills | guarded |
| [2295](https://github.com/rmusser01/tldw_server/security/code-scanning/2295) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1001` | skills | guarded |
| [2296](https://github.com/rmusser01/tldw_server/security/code-scanning/2296) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1006` | skills | guarded |
| [2297](https://github.com/rmusser01/tldw_server/security/code-scanning/2297) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1010` | skills | guarded |
| [2298](https://github.com/rmusser01/tldw_server/security/code-scanning/2298) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:1011` | skills | guarded |
| [2301](https://github.com/rmusser01/tldw_server/security/code-scanning/2301) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/snapshots.py:101` | sandbox | guarded |
| [2302](https://github.com/rmusser01/tldw_server/security/code-scanning/2302) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:142` | sandbox | guarded |
| [2303](https://github.com/rmusser01/tldw_server/security/code-scanning/2303) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:146` | sandbox | guarded |
| [2304](https://github.com/rmusser01/tldw_server/security/code-scanning/2304) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:147` | sandbox | guarded |
| [2305](https://github.com/rmusser01/tldw_server/security/code-scanning/2305) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:149` | sandbox | guarded |
| [2306](https://github.com/rmusser01/tldw_server/security/code-scanning/2306) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:149` | sandbox | guarded |
| [2307](https://github.com/rmusser01/tldw_server/security/code-scanning/2307) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:322` | sandbox | guarded |
| [2308](https://github.com/rmusser01/tldw_server/security/code-scanning/2308) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:334` | sandbox | guarded |
| [2309](https://github.com/rmusser01/tldw_server/security/code-scanning/2309) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:334` | sandbox | guarded |
| [2310](https://github.com/rmusser01/tldw_server/security/code-scanning/2310) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:335` | sandbox | guarded |
| [2311](https://github.com/rmusser01/tldw_server/security/code-scanning/2311) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:335` | sandbox | guarded |
| [2312](https://github.com/rmusser01/tldw_server/security/code-scanning/2312) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:337` | sandbox | guarded |
| [2313](https://github.com/rmusser01/tldw_server/security/code-scanning/2313) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:343` | sandbox | guarded |
| [2314](https://github.com/rmusser01/tldw_server/security/code-scanning/2314) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:351` | sandbox | guarded |
| [2315](https://github.com/rmusser01/tldw_server/security/code-scanning/2315) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:354` | sandbox | guarded |
| [2316](https://github.com/rmusser01/tldw_server/security/code-scanning/2316) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:355` | sandbox | guarded |
| [2317](https://github.com/rmusser01/tldw_server/security/code-scanning/2317) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:356` | sandbox | guarded |
| [2318](https://github.com/rmusser01/tldw_server/security/code-scanning/2318) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:357` | sandbox | guarded |
| [2319](https://github.com/rmusser01/tldw_server/security/code-scanning/2319) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:358` | sandbox | guarded |
| [2320](https://github.com/rmusser01/tldw_server/security/code-scanning/2320) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:358` | sandbox | guarded |
| [2321](https://github.com/rmusser01/tldw_server/security/code-scanning/2321) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:370` | sandbox | guarded |
| [2322](https://github.com/rmusser01/tldw_server/security/code-scanning/2322) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:371` | sandbox | guarded |
| [2323](https://github.com/rmusser01/tldw_server/security/code-scanning/2323) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:372` | sandbox | guarded |
| [2324](https://github.com/rmusser01/tldw_server/security/code-scanning/2324) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:373` | sandbox | guarded |
| [2325](https://github.com/rmusser01/tldw_server/security/code-scanning/2325) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:374` | sandbox | guarded |
| [2326](https://github.com/rmusser01/tldw_server/security/code-scanning/2326) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:374` | sandbox | guarded |
| [2327](https://github.com/rmusser01/tldw_server/security/code-scanning/2327) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:379` | sandbox | guarded |
| [2328](https://github.com/rmusser01/tldw_server/security/code-scanning/2328) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:380` | sandbox | guarded |
| [2329](https://github.com/rmusser01/tldw_server/security/code-scanning/2329) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:381` | sandbox | guarded |
| [2330](https://github.com/rmusser01/tldw_server/security/code-scanning/2330) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:382` | sandbox | guarded |
| [2331](https://github.com/rmusser01/tldw_server/security/code-scanning/2331) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:383` | sandbox | guarded |
| [2332](https://github.com/rmusser01/tldw_server/security/code-scanning/2332) | `api/v1/endpoints/sandbox.py:966` | `core/Sandbox/snapshots.py:383` | sandbox | guarded |
| [2333](https://github.com/rmusser01/tldw_server/security/code-scanning/2333) | `api/v1/endpoints/sandbox.py:1003` | `core/Sandbox/snapshots.py:429` | sandbox | guarded |
| [2334](https://github.com/rmusser01/tldw_server/security/code-scanning/2334) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:462` | sandbox | repaired |
| [2335](https://github.com/rmusser01/tldw_server/security/code-scanning/2335) | `api/v1/endpoints/sandbox.py:929` | `core/Sandbox/snapshots.py:465` | sandbox | repaired |
| [2336](https://github.com/rmusser01/tldw_server/security/code-scanning/2336) | `api/v1/endpoints/sandbox.py:898` | `core/Sandbox/snapshots.py:544` | sandbox | guarded |
| [2339](https://github.com/rmusser01/tldw_server/security/code-scanning/2339) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:463` | skills | guarded |
| [2340](https://github.com/rmusser01/tldw_server/security/code-scanning/2340) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:466` | skills | guarded |
| [2341](https://github.com/rmusser01/tldw_server/security/code-scanning/2341) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:435` | skills | guarded |
| [2342](https://github.com/rmusser01/tldw_server/security/code-scanning/2342) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:436` | skills | guarded |
| [2343](https://github.com/rmusser01/tldw_server/security/code-scanning/2343) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:470` | skills | guarded |
| [2344](https://github.com/rmusser01/tldw_server/security/code-scanning/2344) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:644` | skills | guarded |
| [2345](https://github.com/rmusser01/tldw_server/security/code-scanning/2345) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:645` | skills | guarded |
| [2346](https://github.com/rmusser01/tldw_server/security/code-scanning/2346) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:652` | skills | guarded |
| [2347](https://github.com/rmusser01/tldw_server/security/code-scanning/2347) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:656` | skills | guarded |
| [2348](https://github.com/rmusser01/tldw_server/security/code-scanning/2348) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:662` | skills | guarded |
| [2349](https://github.com/rmusser01/tldw_server/security/code-scanning/2349) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:663` | skills | guarded |
| [2350](https://github.com/rmusser01/tldw_server/security/code-scanning/2350) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:663` | skills | guarded |
| [2351](https://github.com/rmusser01/tldw_server/security/code-scanning/2351) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:682` | skills | guarded |
| [2352](https://github.com/rmusser01/tldw_server/security/code-scanning/2352) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:683` | skills | guarded |
| [2353](https://github.com/rmusser01/tldw_server/security/code-scanning/2353) | `api/v1/endpoints/skills.py:550` | `core/Skills/skills_service.py:686` | skills | guarded |
| [2354](https://github.com/rmusser01/tldw_server/security/code-scanning/2354) | `api/v1/endpoints/skills.py:438` | `core/Skills/skills_service.py:2603` | skills | guarded |
| [2605](https://github.com/rmusser01/tldw_server/security/code-scanning/2605) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:758` | audio | guarded |
| [2606](https://github.com/rmusser01/tldw_server/security/code-scanning/2606) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:756` | audio | guarded |
| [2607](https://github.com/rmusser01/tldw_server/security/code-scanning/2607) | `api/v1/endpoints/sync.py:914` | `core/Notes/legacy_attachment_source.py:160` | legacy | guarded |
| [2608](https://github.com/rmusser01/tldw_server/security/code-scanning/2608) | `api/v1/endpoints/sync.py:914` | `core/Notes/legacy_attachment_source.py:373` | legacy | guarded |
| [2609](https://github.com/rmusser01/tldw_server/security/code-scanning/2609) | `api/v1/endpoints/notes.py:4764` | `api/v1/endpoints/notes.py:1117` | notes | guarded |
| [2610](https://github.com/rmusser01/tldw_server/security/code-scanning/2610) | `api/v1/endpoints/notes.py:4764` | `api/v1/endpoints/notes.py:1118` | notes | guarded |
| [2611](https://github.com/rmusser01/tldw_server/security/code-scanning/2611) | `api/v1/endpoints/notes.py:4764` | `api/v1/endpoints/notes.py:1126` | notes | guarded |
| [2612](https://github.com/rmusser01/tldw_server/security/code-scanning/2612) | `api/v1/endpoints/notes.py:4758` | `api/v1/endpoints/notes.py:1132` | notes | guarded |
| [2634](https://github.com/rmusser01/tldw_server/security/code-scanning/2634) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1417` | audio_cleanup | guarded |
| [2635](https://github.com/rmusser01/tldw_server/security/code-scanning/2635) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1417` | audio_cleanup | guarded |
| [2636](https://github.com/rmusser01/tldw_server/security/code-scanning/2636) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1423` | audio_cleanup | guarded |
| [2637](https://github.com/rmusser01/tldw_server/security/code-scanning/2637) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1425` | audio_cleanup | guarded |
| [2638](https://github.com/rmusser01/tldw_server/security/code-scanning/2638) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1431` | audio_cleanup | guarded |
| [2656](https://github.com/rmusser01/tldw_server/security/code-scanning/2656) | `api/v1/endpoints/buddies.py:212` | `core/Buddy/service.py:85` | buddy | guarded |
| [2657](https://github.com/rmusser01/tldw_server/security/code-scanning/2657) | `api/v1/endpoints/buddies.py:212` | `core/Buddy/service.py:98` | buddy | guarded |
| [2658](https://github.com/rmusser01/tldw_server/security/code-scanning/2658) | `api/v1/endpoints/media/add.py:63` | `core/Storage/filesystem_storage.py:43` | storage | guarded |
| [2659](https://github.com/rmusser01/tldw_server/security/code-scanning/2659) | `api/v1/endpoints/media/add.py:63` | `core/Storage/filesystem_storage.py:242` | storage | guarded |
| [2660](https://github.com/rmusser01/tldw_server/security/code-scanning/2660) | `api/v1/endpoints/media/add.py:63` | `core/Storage/filesystem_storage.py:272` | storage | guarded |
| [2661](https://github.com/rmusser01/tldw_server/security/code-scanning/2661) | `api/v1/endpoints/media/add.py:63` | `core/Storage/filesystem_storage.py:275` | storage | guarded |
| [2672](https://github.com/rmusser01/tldw_server/security/code-scanning/2672) | `api/v1/endpoints/media/add.py:57` | `core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:702` | audio | probe review |

The machine-readable file `/tmp/pr2761-python-path-dispositions.json` includes every exact source/sink, full ordered SARIF trace, boundary proof, regression references, analysis and inspected file hashes, and per-trace SHA-256. The temporary file is local review evidence; this document retains the complete alert mapping.

## Limits requiring reviewer attention

Alert 2672 is the intentional `Path(raw).is_dir()` probe needed to distinguish existing local directories from Hub IDs/Whisper aliases. A valid local directory outside the allowed root is rejected before loading; removing the probe would reintroduce the earlier loader/validator mismatch. Directory-existence classification can be observable, so this is explicitly separated from guarded file reads.

Audio cleanup findings include preliminary exists/is_file/resolve probes. Deletion remains behind canonical temporary-root containment. The Python compatibility `startswith` fallback is not executed on the supported Python runtime.

Snapshot and Research repairs require a pre-existing link in application-managed storage to demonstrate the bypass. No HTTP-only route that creates those links was demonstrated. This distinction prevents claiming a remote arbitrary-file exploit from the scanner trace alone.

No query exclusions, inline suppressions, security thresholds, or alert states were changed. Fresh hosted analysis and individual maintainer review remain necessary to reconcile the open scanner inventory.

## Final focused verification

After the final source and import-cleanup changes, snapshot restore/security,
snapshot quotas and Research artifact tests pass **43 tests** together
(`/tmp/pr2761-path-final-tests.log`). Ruff passes for both changed source files
and both test files. Bandit reports **zero findings** across those four files
with test assertions excluded (`/tmp/pr2761-path-final-bandit.json`); separate
production-only Bandit runs also report zero findings without that exclusion.
`git diff --check` passes. The supported runtime is Python >=3.11, confirming
that Audio cleanup's older-Python compatibility fallback is unreachable.

The exact disposition counts are **141 guarded candidates**, **10 repaired
boundary paths** (2149, 2150, 2151, 2152, 2277, 2278, 2279, 2280, 2334, 2335),
and **one intentional metadata probe requiring review** (2672). These are
source-review dispositions, not a claim that GitHub has closed the alerts.

## Alert 2672 follow-up: loader precedence and observable behavior

A focused, isolated probe (`/tmp/pr2761-model-probe-proof.py`, executed with
`PYTHONPATH=.` against this worktree) confirms the distinction precisely:
`tiny.en`, `organization/remote-model`, and `organization/qwen-model` are accepted
as alias/Hub identifiers when absent from the working directory. Creating a
directory with the same name outside the approved model root makes normalization
reject the identifier before loading. The transcription endpoint maps Whisper
validation failures to HTTP 400 at `audio/audio_transcriptions.py:967–973`; an
accepted identifier proceeds to normal cache/loading behavior. This is a
potential directory-existence distinction for these ambiguous identifier shapes,
not arbitrary outside file reading. The probe also verifies that the Whisper
wrapper never invokes its delegate for the existing outside alias, whereas the
absent alias still reaches normal loading. Output is in
`/tmp/pr2761-model-probe-proof.log`.

Simply moving `is_dir` after root validation or returning remote-shaped strings
without checking their working-directory interpretation would weaken the current
boundary. The installed faster-whisper loader checks
`os.path.isdir(model_size_or_path)` before its remote `download_model` branch;
Transformers likewise treats an existing path as local in `from_pretrained`.
The wrapper currently passes valid Hub IDs/aliases through those APIs. Eliminating
the existence distinction while keeping those identifiers requires a separate
explicit remote-resolution boundary that first downloads/resolves a Hub snapshot
and then supplies an approved absolute snapshot path to the loader, including
cache, revision, authentication, offline and in-memory-model compatibility. It is
not a safe conditional reorder. No such architectural rewrite was introduced to
force analyzer closure.

The existing absolute/relative outside-root, symlink, allowed-root, artifact
completeness, Hub ID and alias regressions remain applicable. Alert 2672 remains
an explicitly identified intentional validation probe for maintainer disposition;
this review does not call it fixed or hide its observable behavior.

## Research read-boundary follow-up

Independent review reproduced an adjacent read leak: a valid recorded artifact
file replaced by a pre-existing symlink was read without checking the recorded
path again. All three readers (`read_text`, `read_json`, `read_jsonl`) now use one
resolver that confines the recorded absolute path to the requested session's
hashed directory and rejects symlinks through the existing `safe_join` helper.
The resolver does not create directories during reads. Valid generated/versioned
paths and missing-file `None` results are preserved.

Nine regressions failed before the change: each reader followed a leaf symlink,
an outside recorded path, or another session's recorded path. They now pass,
along with three missing-file controls and the existing round trips. The broader
artifact/core-hardening/jobs-service/jobs-worker run passes **93 tests**.
Ruff passes and Bandit reports zero findings with test assertions excluded.
Evidence: `/tmp/pr2761-path-research-reads-red.log`,
`/tmp/pr2761-path-research-reads-broader.log`, and
`/tmp/pr2761-path-research-reads-bandit.json`. Independent review also reports
75 passing tests across its combined Research/snapshot/security selection.

## Global alert scope: current main comparison

Global dismissal must account for main, not only this PR. The supplied main
instance snapshot has **137** shared path IDs on
`d9c245ac14c40df855d1ab6cd19b3c137b16b47b`. Comparison covered all **33** files
appearing in those PR traces, including upstream endpoints, database root helpers,
path utilities and service boundaries: 21 files are identical to the Python
analysis source and 12 differ. Changed traced functions and root configuration
selection were reviewed individually.

**Do not globally dismiss 2281 or 2282.** The path-only checkpoint directory
containment is unchanged, but main's resume endpoint lacks the branch-added
checkpoint owner/admin authorization. A probe executing the exact main endpoint
body with only checkpoint loading stubbed returns a different user's completed
checkpoint count (17) to a non-admin principal with `media.read`; the branch body
returns HTTP 403 `checkpoint_owner_forbidden`. This is a concrete authorization
scope difference despite the unchanged sink file. Evidence:
`/tmp/pr2761-main-checkpoint-scope-proof.log`.

The **10 repaired IDs** (2149–2152, 2277–2280, 2334–2335) remain source fixes and
must not be globally dismissed. Of the remaining shared IDs, **125** have the
same relevant path boundary on main. Their exact per-group lists and hashes of
all 33 main source files are in `/tmp/pr2761-main-path-scope.json`; each shared
record in `/tmp/pr2761-python-path-dispositions.json` now has a `main_scope`
section. This supports only the stated path-injection disposition, not a blanket
security verdict on main.

For main audio IDs 2104 and 2275, the supported claim is specifically canonical
allowed-root containment before provider/conversion reads. Main lacks the
branch's pre-resolution no-symlink assertion, so an in-root link may be accepted;
no claim that main rejects every symlink is supported. Audio cleanup still has
its canonical temporary-root check before deletion. Main's database-root helper
diff changes administrator configuration/environment precedence, not user-ID
validation or `safe_join` confinement. Sandbox ownership/workspace guards,
Skills bundle lifecycle guards, export filename containment and temporary file
artifact path checks remain equivalent for the shared findings.
