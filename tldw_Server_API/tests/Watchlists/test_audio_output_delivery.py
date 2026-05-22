"""Tests for audio output delivery endpoint.

Tests the GET /watchlists/runs/{run_id}/audio endpoint and
audio artifact lookup behavior.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestGetRunAudioEndpoint:
    """Tests for the /runs/{run_id}/audio endpoint."""

    @pytest.mark.asyncio
    async def test_returns_404_when_run_not_found(self):
        """Test 404 when run doesn't exist."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        db = MagicMock()
        db.get_run.side_effect = KeyError("not found")

        user = MagicMock()
        user.role = "admin"

        with pytest.raises(Exception) as exc_info:
            await get_run_audio(run_id=999, target_user_id=None, current_user=user, db=db)
        assert "404" in str(exc_info.value.status_code) or exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_when_no_audio_task(self):
        """Test 404 when run has no audio briefing task."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=1,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"items_fetched": 10}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        with pytest.raises(Exception) as exc_info:
            await get_run_audio(run_id=1, target_user_id=None, current_user=user, db=db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_workflows_db_returns_queued_scheduler_task(self):
        """Missing Workflows DB should still report queued scheduler task status."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=1,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_queued"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        scheduler_task = SimpleNamespace(
            id="task_queued",
            status="queued",
            queue_name="workflows",
            error=None,
        )
        scheduler = MagicMock()
        scheduler.get_task = AsyncMock(return_value=scheduler_task)

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.get_global_scheduler",
                new=AsyncMock(return_value=scheduler),
                create=True,
            ),
        ):
            result = await get_run_audio(run_id=1, target_user_id=None, current_user=user, db=db)

        assert result["run_id"] == 1
        assert result["task_id"] == "task_queued"
        assert result["status"] == "queued"
        assert result["queue_name"] == "workflows"
        assert result["audio_uri"] is None
        assert result["download_url"] is None

    @pytest.mark.asyncio
    async def test_missing_workflows_db_returns_pending_when_scheduler_unavailable(self):
        """Missing Workflows DB should return safe pending fallback when scheduler lookup fails."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=1,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_pending"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        sensitive_error = "scheduler failed at /tmp/secret/path with bearer token abc123"
        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.get_global_scheduler",
                new=AsyncMock(side_effect=RuntimeError(sensitive_error)),
                create=True,
            ),
        ):
            result = await get_run_audio(run_id=1, target_user_id=None, current_user=user, db=db)

        assert result["run_id"] == 1
        assert result["task_id"] == "task_pending"
        assert result["status"] == "pending"
        assert result["queue_name"] == "workflows"
        assert result["audio_uri"] is None
        assert result["download_url"] is None
        assert result["fallback_reason"] == "workflow_run_not_started"
        assert sensitive_error not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_returns_pending_when_workflow_not_found(self):
        """Test returns pending status when workflow run not found yet."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=1,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_abc"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        # Mock WorkflowsDB to return no matching runs
        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = []

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=1, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "pending"
        assert result["task_id"] == "task_abc"
        assert result["audio_uri"] is None

    @pytest.mark.asyncio
    async def test_returns_audio_when_artifact_found(self):
        """Test returns audio info when artifact is found."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=7,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_xyz"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        # Mock workflow run with matching metadata
        wf_run = SimpleNamespace(
            id="wf_run_1",
            status="completed",
            metadata_json=json.dumps({"watchlist_run_id": 7}),
        )

        # Mock audio artifact
        audio_art = SimpleNamespace(
            id="art_audio_1",
            type="tts_audio",
            uri="file:///tmp/briefing.mp3",
            size_bytes=1024000,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"multi_voice": True}),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = [wf_run]
        mock_wf_db.list_artifacts.return_value = [audio_art]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=7, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "completed"
        assert result["audio_uri"] == "file:///tmp/briefing.mp3"
        assert result["artifact_id"] == "art_audio_1"
        assert result["download_url"] == "/api/v1/workflows/artifacts/art_audio_1/download"
        assert result["size_bytes"] == 1024000

    @pytest.mark.asyncio
    async def test_handles_db_errors_gracefully(self):
        """Test graceful error handling when workflow DB lookup fails."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=1,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_fail"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                side_effect=RuntimeError("db path error /private/secret/workflows.db"),
            ),
        ):
            result = await get_run_audio(run_id=1, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "unknown"
        assert result["task_id"] == "task_fail"
        assert result["error"] == "artifact_lookup_failed"
        assert result["fallback_reason"] == "artifact_lookup_failed"
        assert "db path error" not in json.dumps(result)
        assert "/private/secret" not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_paginated_scan_returns_pending_when_no_matching_run(self):
        """Scans beyond first page and returns pending when no metadata match exists."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=77,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_paged_pending"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "default"

        page1 = [
            SimpleNamespace(
                run_id=f"wf_{idx}",
                status="running",
                metadata_json=json.dumps({"watchlist_run_id": 99999}),
            )
            for idx in range(50)
        ]
        page2 = [
            SimpleNamespace(
                run_id=f"wf_tail_{idx}",
                status="running",
                metadata_json=json.dumps({"watchlist_run_id": 88888}),
            )
            for idx in range(25)
        ]

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.side_effect = [page1, page2]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=77, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "pending"
        assert result["task_id"] == "task_paged_pending"
        assert mock_wf_db.list_runs.call_count == 2
        first_call = mock_wf_db.list_runs.call_args_list[0].kwargs
        second_call = mock_wf_db.list_runs.call_args_list[1].kwargs
        assert first_call["offset"] == 0
        assert second_call["offset"] == 50

    @pytest.mark.asyncio
    async def test_paginated_scan_finds_matching_run_and_audio_later_page(self):
        """Finds match after first page and returns artifact metadata."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=42,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_paged_hit"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "default"

        page1 = [
            SimpleNamespace(
                run_id=f"wf_old_{idx}",
                status="completed",
                metadata_json=json.dumps({"watchlist_run_id": 123456}),
            )
            for idx in range(50)
        ]
        matching_run = SimpleNamespace(
            run_id="wf_target_42",
            status="completed",
            metadata_json=json.dumps({"watchlist_run_id": 42}),
        )
        page2 = [matching_run]

        audio_art = SimpleNamespace(
            id="art_audio_paged",
            type="tts_audio",
            uri="file:///tmp/paged-briefing.mp3",
            size_bytes=777,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"multi_voice": True}),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.side_effect = [page1, page2]
        mock_wf_db.list_artifacts.return_value = [audio_art]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=42, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "completed"
        assert result["task_id"] == "task_paged_hit"
        assert result["artifact_id"] == "art_audio_paged"
        assert result["audio_uri"] == "file:///tmp/paged-briefing.mp3"
        assert mock_wf_db.list_runs.call_count == 2
        first_call = mock_wf_db.list_runs.call_args_list[0].kwargs
        second_call = mock_wf_db.list_runs.call_args_list[1].kwargs
        assert first_call["offset"] == 0
        assert second_call["offset"] == 50

    @pytest.mark.asyncio
    async def test_paginated_scan_finds_matching_run_after_twenty_full_pages(self):
        """Does not mark old audio runs pending just because they are beyond 1,000 rows."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=43,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_deep_hit"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "default"

        non_matching_pages = [
            [
                SimpleNamespace(
                    run_id=f"wf_old_{page_idx}_{row_idx}",
                    status="completed",
                    metadata_json=json.dumps({"watchlist_run_id": 123456}),
                )
                for row_idx in range(50)
            ]
            for page_idx in range(20)
        ]
        matching_run = SimpleNamespace(
            run_id="wf_target_43",
            status="completed",
            metadata_json=json.dumps({"watchlist_run_id": 43}),
        )
        audio_art = SimpleNamespace(
            id="art_audio_deep_paged",
            type="tts_audio",
            uri="file:///tmp/deep-paged-briefing.mp3",
            size_bytes=999,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"multi_voice": True}),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.side_effect = [*non_matching_pages, [matching_run]]
        mock_wf_db.list_artifacts.return_value = [audio_art]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=43, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "completed"
        assert result["task_id"] == "task_deep_hit"
        assert result["artifact_id"] == "art_audio_deep_paged"
        assert mock_wf_db.list_runs.call_count == 21
        assert mock_wf_db.list_runs.call_args_list[-1].kwargs["offset"] == 1000

    @pytest.mark.asyncio
    async def test_cross_user_audio_lookup_uses_resolved_workflow_tenant(self):
        """Admin cross-user audio lookup should not filter workflow runs by the admin tenant."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=44,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_cross_user"}),
            error_msg=None,
        )
        target_db = MagicMock()
        target_db.get_run.return_value = run

        current_db = MagicMock()
        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "admin-tenant"

        matching_run = SimpleNamespace(
            run_id="wf_target_44",
            status="completed",
            metadata_json=json.dumps({"watchlist_run_id": 44}),
        )
        audio_art = SimpleNamespace(
            id="art_cross_user",
            type="tts_audio",
            uri="file:///tmp/cross-user-briefing.mp3",
            size_bytes=444,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"multi_voice": True}),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = [matching_run]
        mock_wf_db.list_artifacts.return_value = [audio_art]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
                new=AsyncMock(return_value=(2, target_db)),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_watchlist_workflow_tenant_id",
                new=AsyncMock(return_value="target-tenant"),
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/target_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=44, target_user_id=2, current_user=user, db=current_db)

        assert result["status"] == "completed"
        assert result["artifact_id"] == "art_cross_user"
        assert mock_wf_db.list_runs.call_args.kwargs["tenant_id"] == "target-tenant"
        assert mock_wf_db.list_runs.call_args.kwargs["tenant_id"] != "admin-tenant"

    @pytest.mark.asyncio
    async def test_prefers_final_or_mixed_artifact_when_multiple_candidates(self):
        """Returns final-tagged/mixed artifact over earlier intermediate artifacts."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=88,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_prefer_final"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "default"

        wf_run = SimpleNamespace(
            run_id="wf_run_88",
            status="completed",
            metadata_json=json.dumps({"watchlist_run_id": 88}),
        )
        intermediate = SimpleNamespace(
            id="art_raw",
            type="tts_audio",
            uri="file:///tmp/briefing_raw.mp3",
            size_bytes=120,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"multi_voice": True}),
        )
        final_mixed = SimpleNamespace(
            id="art_final",
            type="tts_audio",
            uri="file:///tmp/briefing_mixed.mp3",
            size_bytes=240,
            mime_type="audio/mpeg",
            metadata_json=json.dumps(
                {
                    "multi_voice": True,
                    "background_mixed": True,
                    "final_artifact": True,
                }
            ),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = [wf_run]
        mock_wf_db.list_artifacts.return_value = [intermediate, final_mixed]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=88, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "completed"
        assert result["artifact_id"] == "art_final"
        assert result["audio_uri"] == "file:///tmp/briefing_mixed.mp3"

    @pytest.mark.asyncio
    async def test_returns_structured_audio_artifact_graph(self):
        """Returns script, speaker, final, and fallback details for audio briefings."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=91,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_graph"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"
        user.id = 1
        user.tenant_id = "default"

        wf_run = SimpleNamespace(
            run_id="wf_run_91",
            status="completed",
            metadata_json=json.dumps(
                {
                    "watchlist_run_id": 91,
                    "fallback_reason": "multi-voice generation failed; single voice fallback used",
                }
            ),
        )
        script_art = SimpleNamespace(
            id="art_script",
            type="audio_script",
            uri="file:///tmp/briefing-script.md",
            size_bytes=512,
            mime_type="text/markdown",
            metadata_json=json.dumps({"script_artifact": True, "title": "Briefing script"}),
        )
        host_art = SimpleNamespace(
            id="art_host",
            type="tts_audio",
            uri="file:///tmp/host.mp3",
            size_bytes=1024,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"speaker_artifact": True, "speaker_id": "host", "label": "Host"}),
        )
        analyst_art = SimpleNamespace(
            id="art_analyst",
            type="tts_audio",
            uri="file:///tmp/analyst.mp3",
            size_bytes=2048,
            mime_type="audio/mpeg",
            metadata_json=json.dumps({"speaker_artifact": True, "speaker_id": "analyst", "label": "Analyst"}),
        )
        final_art = SimpleNamespace(
            id="art_final_graph",
            type="tts_audio",
            uri="file:///tmp/final.mp3",
            size_bytes=4096,
            mime_type="audio/mpeg",
            metadata_json=json.dumps(
                {
                    "multi_voice": True,
                    "background_mixed": True,
                    "final_artifact": True,
                    "title": "Final mix",
                }
            ),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = [wf_run]
        mock_wf_db.list_artifacts.return_value = [
            script_art,
            host_art,
            analyst_art,
            final_art,
        ]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=91, target_user_id=None, current_user=user, db=db)

        assert result["status"] == "completed"
        assert result["artifact_id"] == "art_final_graph"
        assert result["audio_uri"] == "file:///tmp/final.mp3"
        assert result["download_url"] == "/api/v1/workflows/artifacts/art_final_graph/download"
        assert result["script_artifact"]["artifact_id"] == "art_script"
        assert result["script_artifact"]["title"] == "Briefing script"
        assert [entry["speaker_id"] for entry in result["speaker_artifacts"]] == [
            "host",
            "analyst",
        ]
        assert result["final_artifact"]["artifact_id"] == "art_final_graph"
        assert result["final_artifact"]["title"] == "Final mix"
        assert result["fallback_reason"] == "multi-voice generation failed; single voice fallback used"

    @pytest.mark.asyncio
    async def test_speaker_artifacts_are_not_final_audio_candidates(self):
        """Speaker clips should not masquerade as the final podcast artifact."""
        from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_audio

        run = SimpleNamespace(
            id=92,
            job_id=1,
            status="completed",
            started_at=None,
            finished_at=None,
            stats_json=json.dumps({"audio_briefing_task_id": "task_speaker_only"}),
            error_msg=None,
        )
        db = MagicMock()
        db.get_run.return_value = run

        user = MagicMock()
        user.role = "admin"

        wf_run = SimpleNamespace(
            id="wf_speaker_only",
            status="running",
            metadata_json=json.dumps({"watchlist_run_id": 92}),
        )
        speaker_art = SimpleNamespace(
            id="art_speaker_only",
            type="tts_audio",
            uri="file:///tmp/host.mp3",
            size_bytes=1024,
            mime_type="audio/mpeg",
            metadata_json=json.dumps(
                {
                    "speaker_artifact": True,
                    "speaker_id": "HOST",
                    "voice": "af_bella",
                }
            ),
        )

        mock_wf_db = MagicMock()
        mock_wf_db.list_runs.return_value = [wf_run]
        mock_wf_db.list_artifacts.return_value = [speaker_art]

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
                return_value=1,
            ),
            patch(
                "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_user_base_directory",
                return_value="/tmp/test_user",  # nosec B108
            ),
            patch("os.path.exists", return_value=True),
            patch(
                "tldw_Server_API.app.core.DB_Management.Workflows_DB.WorkflowsDatabase",
                return_value=mock_wf_db,
            ),
        ):
            result = await get_run_audio(run_id=92, target_user_id=None, current_user=user, db=db)

        assert result["audio_uri"] is None
        assert result["download_url"] is None
        assert result["final_artifact"] is None
        assert result["speaker_artifacts"][0]["artifact_id"] == "art_speaker_only"
