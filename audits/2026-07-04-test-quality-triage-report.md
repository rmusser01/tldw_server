# Test Quality Triage Report — Initial Run

**Date:** 2026-07-04
**Tool:** `Helper_Scripts/ci/test_quality_triage.py` (see its docstring for detector definitions)
**Command:** `make test-triage` (or `python Helper_Scripts/ci/test_quality_triage.py`)
**Context:** Task 1 of `Docs/superpowers/plans/2026-07-04-test-suite-improvement-implementation-plan.md`, targeting finding RA7 of `audits/2026-07-04-test-suite-audit-round2.md`.

## Summary

- **3,980** test files scanned; **733** flagged; **888** (file, flag) offenses in the ratchet baseline (`Helper_Scripts/ci/test_quality_baseline.txt`).
- Flag totals: ambiguous_accept=112, mock_density=173, skip_stale=440, status_only=696, stub_injection=499, tautology_suspect=477.
- Report-only for now: the script exits 0 by default; `--enforce` (ratchet vs. baseline) is available once the team opts in — the intended promotion path is a CI step after the Task 2 exemplar fixes shrink the baseline.

## Validation

- **Determinism:** two consecutive runs produce byte-identical output.
- **Known offenders:** all 4 audit exemplars flagged (`RAG/test_dual_backend_end_to_end.py` stub_injection, `Media/test_media_navigation.py` stub_injection, `DB_Management/test_media_db_schema_bootstrap.py` tautology_suspect, `Character_Chat/test_complete_v2_streaming_with_mock_openai.py` ambiguous_accept).
- **Precision:** 10 flagged files sampled across the score range, 14 (file, flag) pairs hand-verified — **14/14 definitional true-positives**. Two definition-level tunings were applied as a result: tautology_suspect's collector shape only fires when the collector comparisons are the test's *only* asserts (legitimate interaction tests also assert real outputs), and skip_stale excludes dependency/env availability gates.

## Reading the flags

- `ambiguous_accept` and `tautology_suspect` are the highest-signal flags (a test that cannot fail, or that verifies its own stub).
- `stub_injection`'s `dependency_overrides` arm fires on the standard FastAPI TestClient wiring idiom — treat as a ranking input for over-mocked endpoint tests, not a per-instance verdict.
- `skip_stale` is informational (round-1 F9 already enforces reasons exist).

## Top 50 by score

```
[test-triage] top 50 by score:
   276  tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py  [status_onlyx4, stub_injectionx1, tautology_suspectx53]
   205  tldw_Server_API/tests/test_authnz_backends.py  [tautology_suspectx41]
   180  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py  [tautology_suspectx36]
   126  tldw_Server_API/tests/Workspaces/test_workspaces_api.py  [status_onlyx3, stub_injectionx40]
    74  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_external_file_sync_integration.py  [mock_densityx3, tautology_suspectx13]
    65  tldw_Server_API/tests/Audit/test_audit_export_endpoint.py  [status_onlyx4, stub_injectionx19]
    59  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_integration.py  [ambiguous_acceptx10, skip_stalex8, status_onlyx3, stub_injectionx1]
    56  tldw_Server_API/tests/External_Sources/test_connectors_endpoints_api.py  [mock_densityx14, status_onlyx7]
    55  tldw_Server_API/tests/RAG_NEW/unit/test_chromadb_adapter_sanitizers.py  [tautology_suspectx11]
    51  tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py  [skip_stalex3, stub_injectionx17]
    50  tldw_Server_API/tests/Embeddings/test_embeddings_v5_property.py  [ambiguous_acceptx2, tautology_suspectx8]
    50  tldw_Server_API/tests/Evaluations/unit/test_evals_cli_recipe_commands.py  [tautology_suspectx10]
    46  tldw_Server_API/tests/Research/test_research_runs_endpoint.py  [status_onlyx2, stub_injectionx14]
    45  tldw_Server_API/tests/Evaluations/test_evaluations_crud_endpoint_sanitization.py  [tautology_suspectx9]
    45  tldw_Server_API/tests/Media/test_document_references.py  [status_onlyx3, stub_injectionx13]
    45  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_external_reference_import_integration.py  [mock_densityx5, tautology_suspectx6]
    40  tldw_Server_API/tests/AuthNZ/unit/test_key_resolution.py  [tautology_suspectx8]
    40  tldw_Server_API/tests/Character_Chat_NEW/unit/test_stream_persist_lookup.py  [tautology_suspectx8]
    40  tldw_Server_API/tests/test_contextual_properties.py  [tautology_suspectx8]
    39  tldw_Server_API/tests/PaperSearch/integration/test_biorxiv_reports_external.py  [ambiguous_acceptx7, status_onlyx2]
    39  tldw_Server_API/tests/Sharing/test_sharing_endpoints.py  [mock_densityx5, status_onlyx12]
    37  tldw_Server_API/tests/MediaFiles/test_file_endpoint.py  [status_onlyx2, stub_injectionx11]
    36  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py  [stub_injectionx12]
    36  tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_mlx.py  [mock_densityx12]
    36  tldw_Server_API/tests/Workflows/test_workflows_api.py  [status_onlyx12, stub_injectionx4]
    35  tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs_worker.py  [tautology_suspectx7]
    35  tldw_Server_API/tests/Integrations/test_integrations_control_plane_service.py  [tautology_suspectx7]
    34  tldw_Server_API/tests/Skills/integration/test_skills_api.py  [status_onlyx17]
    32  tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py  [status_onlyx1, stub_injectionx10]
    32  tldw_Server_API/tests/PaperSearch/integration/test_hal_external.py  [ambiguous_acceptx6, status_onlyx1]
    31  tldw_Server_API/tests/PaperSearch/integration/test_paper_search_external.py  [ambiguous_acceptx5, status_onlyx3]
    30  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_runners_llm.py  [tautology_suspectx6]
    30  tldw_Server_API/tests/Billing/test_billing_enforcer_org_usage.py  [tautology_suspectx6]
    30  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py  [tautology_suspectx6]
    30  tldw_Server_API/tests/Evaluations/property/test_evaluation_invariants.py  [tautology_suspectx6]
    30  tldw_Server_API/tests/Guardian/test_self_monitoring_alerts_pagination.py  [tautology_suspectx6]
    30  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py  [stub_injectionx10]
    30  tldw_Server_API/tests/Media/test_media_navigation.py  [stub_injectionx10]
    29  tldw_Server_API/tests/Media/test_document_outline.py  [status_onlyx1, stub_injectionx9]
    29  tldw_Server_API/tests/PaperSearch/integration/test_figshare_external.py  [ambiguous_acceptx5, status_onlyx2]
    28  tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py  [stub_injectionx1, tautology_suspectx5]
    27  tldw_Server_API/tests/Media/test_document_insights.py  [stub_injectionx9]
    26  tldw_Server_API/tests/Watchlists/test_operational_limits.py  [status_onlyx13]
    26  tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py  [status_onlyx13]
    25  tldw_Server_API/tests/AuthNZ_Federation/test_oidc_service.py  [tautology_suspectx5]
    25  tldw_Server_API/tests/Notes_NEW/unit/test_notes_keyword_link_endpoint_pagination.py  [tautology_suspectx5]
    25  tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_service.py  [tautology_suspectx5]
    24  tldw_Server_API/tests/Media/test_media_usage_events.py  [status_onlyx12]
    24  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py  [mock_densityx8]
    24  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py  [stub_injectionx3, tautology_suspectx3]
```
