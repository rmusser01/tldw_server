# PyPI 0.1.42 recovery run final triage

Run35540174556 at cd2dbc792b8888555abac5c5c9eafa7a43d9b0e4 ended cancelled. Test Suite Gate106156383876 started21:57:49UTC and ended22:58:04UTC. GitHub annotation explicitly states: "The job has exceeded the maximum execution time of 1h0m0s". Build Distributions and both publication jobs were skipped. No package was built or published by this run; no rerun or publication mutation performed.

The command was unsharded `python -m pytest -q`. Collection succeeded:63144 items,12 skipped; Python3.12.14/pytest9.1.1, randomized seed9042326. Test execution reached8% before the timeout. It was still making progress up to22:57:23 in MediaIngestion_NEW/integration/test_media_ingest_jobs.py, so the logs do not establish a single hung test as the root cause.

The immediate workflow failure is its one-hour execution limit. Separately, this was not a green suite awaiting completion:128 F/E progress markers occurred across29 files. No terminal assertion tracebacks/node IDs/summary were emitted before cancellation. These counts are observed progress markers, not a complete final failure count or diagnoses of their causes. The first failing file contains only test_artifacts_list_and_download_roundtrip, so that exact test can be identified statically; other randomized node identities require follow-up reproduction with immediate failure output. Increasing timeout alone does not establish correctness.

## Current policy versus prior proposal

Current .github/workflows/publish-pypi.yml requires test-suite before build; current test_pypi_workflow_contracts explicitly requires build.needs=[detect-version,test-suite]. The active 0.1.42 recovery plan Stage3 explicitly requires the full workflow test gate, build, checks and publication to pass and says to keep the task open if later gates fail.

Related TASK13257/commitf5c541db61/PR2956 (open, unmerged, based on dev) proposes replacing that full gate with release-contract tests plus minimal startup smoke, retaining make pypi-check. Its task markedDone describes implemented proposal, not merged release policy. PR body relies on broader normal CI but the patch does not bind publication to green exact-head broad-suite evidence. It is historical evidence of the same capacity problem, not automatic authorization to weaken the existing gate. No changes applied.

## Observed failing/error progress by file

- `tldw_Server_API/tests/sandbox/test_artifacts_api.py`: 1 F/E markers.
- `tldw_Server_API/tests/frontend_e2e/test_knowledge_rag_workflow.py`: 1 F/E markers.
- `tldw_Server_API/tests/sandbox/test_artifacts_perf_large_tree.py`: 2 F/E markers.
- `tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py`: 1 F/E markers.
- `tldw_Server_API/tests/Web_Scraping/test_auto_chunking_web_ingest.py`: 2 F/E markers.
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`: 3 F/E markers.
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`: 43 F/E markers.
- `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`: 1 F/E markers.
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_code_processing.py`: 4 F/E markers.
- `tldw_Server_API/tests/Telegram/test_telegram_jobs_and_delivery.py`: 3 F/E markers.
- `tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py`: 1 F/E markers.
- `tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py`: 4 F/E markers.
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`: 1 F/E markers.
- `tldw_Server_API/tests/AuthNZ_Postgres/test_allowlists_budget_402_pg.py`: 1 F/E markers.
- `tldw_Server_API/tests/sandbox/test_runtime_unavailable.py`: 6 F/E markers.
- `tldw_Server_API/tests/Workspaces/test_workspace_activity_index.py`: 4 F/E markers.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py`: 5 F/E markers.
- `tldw_Server_API/tests/frontend_e2e/test_evaluations_workflow.py`: 1 F/E markers.
- `tldw_Server_API/tests/Admin/test_admin_users_service_sanitizers.py`: 2 F/E markers.
- `tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py`: 1 F/E markers.
- `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`: 5 F/E markers.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`: 2 F/E markers.
- `tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py`: 2 F/E markers.
- `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`: 1 F/E markers.
- `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py`: 3 F/E markers.
- `tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py`: 1 F/E markers.
- `tldw_Server_API/tests/sandbox/test_sandbox_api.py`: 16 F/E markers.
- `tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py`: 3 F/E markers.
- `tldw_Server_API/tests/Integrations/test_integrations_control_plane_endpoints.py`: 8 F/E markers.

## Artifacts

- Raw completed test log: /tmp/pypi0142-recovery-test-gate.log
- Job metadata: /tmp/pypi0142-recovery-job.json
- Explicit timeout annotations: /tmp/pypi0142-recovery-annotations.json
- Final run snapshot: /tmp/pypi0142-recovery-monitor.json
- Progress-marker inventory: /tmp/pypi0142-recovery-failure-markers.json
- Related PR history: /tmp/pypi-timeout-pr2956.json

Read-only monitoring/investigation; repository, gates, package versions and publication state unchanged.
