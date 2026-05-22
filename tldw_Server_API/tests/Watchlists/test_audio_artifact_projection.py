"""Tests for Watchlists durable audio artifact projection helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _workflow_run(
    *,
    run_id: str = "wf_run_91",
    status: str = "succeeded",
    metadata: dict[str, Any] | None = None,
    definition_metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        status=status,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
        definition_snapshot_json=json.dumps({"metadata": definition_metadata or {}}),
        inputs_json=json.dumps(inputs or {}),
    )


def _artifact(
    artifact_id: str,
    *,
    type_: str = "tts_audio",
    metadata: dict[str, Any] | None = None,
    uri: str | None = None,
    size_bytes: int = 100,
    created_at: str = "2026-05-22T10:00:00Z",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=artifact_id,
        type=type_,
        uri=uri or f"file:///tmp/{artifact_id}.mp3",
        size_bytes=size_bytes,
        mime_type="text/markdown" if type_ == "audio_script" else "audio/mpeg",
        metadata_json=json.dumps(metadata or {}),
        created_at=created_at,
    )


def _watchlist_meta(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "source": "watchlist_audio_briefing",
        "watchlist_job_id": 42,
        "watchlist_run_id": 91,
        "audio_request_id": "wla_current",
    }
    metadata.update(overrides)
    return metadata


def test_build_audio_projection_graph_sanitizes_artifact_summaries():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import build_audio_projection

    projection = build_audio_projection(
        run_id=91,
        task_id="task_graph",
        audio_request_id="wla_current",
        workflow_run=_workflow_run(
            status="succeeded",
            metadata=_watchlist_meta(fallback_reason="single voice fallback used"),
        ),
        artifacts=[
            _artifact(
                "art_script",
                type_="audio_script",
                metadata=_watchlist_meta(script_artifact=True, title="Briefing script"),
                uri="file:///tmp/script.md",
                size_bytes=512,
            ),
            _artifact("art_host", metadata=_watchlist_meta(speaker_artifact=True, speaker_id="HOST", voice="af_bella")),
            _artifact(
                "art_analyst",
                metadata=_watchlist_meta(speaker_artifact=True, speaker_id="ANALYST", voice="am_adam"),
            ),
            _artifact(
                "art_final",
                metadata=_watchlist_meta(final_artifact=True, background_mixed=True, title="Final mix"),
                size_bytes=4096,
            ),
        ],
    )

    assert projection["status"] == "completed"
    assert projection["run_id"] == 91
    assert projection["task_id"] == "task_graph"
    assert projection["workflow_run_id"] == "wf_run_91"
    assert projection["audio_request_id"] == "wla_current"
    assert projection["artifact_id"] == "art_final"
    assert projection["download_url"] == "/api/v1/workflows/artifacts/art_final/download"
    assert projection["final_artifact"]["artifact_id"] == "art_final"
    assert projection["final_artifact"]["title"] == "Final mix"
    assert "uri" not in projection["final_artifact"]
    assert projection["script_artifact"]["artifact_id"] == "art_script"
    assert "uri" not in projection["script_artifact"]
    assert [entry["speaker_id"] for entry in projection["speaker_artifacts"]] == ["HOST", "ANALYST"]
    assert all("uri" not in entry for entry in projection["speaker_artifacts"])
    assert projection["fallback_reason"] == "single voice fallback used"


def test_build_audio_projection_script_only_partial_graph():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import build_audio_projection

    projection = build_audio_projection(
        run_id=91,
        task_id="task_script",
        audio_request_id="wla_current",
        workflow_run=_workflow_run(status="running", metadata=_watchlist_meta()),
        artifacts=[
            _artifact(
                "art_script",
                type_="audio_script",
                metadata=_watchlist_meta(script_artifact=True, title="Briefing script"),
                uri="file:///tmp/script.md",
            )
        ],
    )

    assert projection["status"] == "running"
    assert projection["script_artifact"]["artifact_id"] == "art_script"
    assert projection["speaker_artifacts"] == []
    assert projection["final_artifact"] is None
    assert projection.get("artifact_id") is None
    assert projection.get("download_url") is None


def test_speaker_artifacts_do_not_become_final_audio():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import build_audio_projection

    projection = build_audio_projection(
        run_id=91,
        task_id="task_speaker",
        audio_request_id="wla_current",
        workflow_run=_workflow_run(status="running", metadata=_watchlist_meta()),
        artifacts=[
            _artifact("art_host", metadata=_watchlist_meta(speaker_artifact=True, speaker_id="HOST", voice="af_bella"))
        ],
    )

    assert projection["final_artifact"] is None
    assert projection.get("artifact_id") is None
    assert projection["speaker_artifacts"][0]["artifact_id"] == "art_host"


def test_background_mixed_and_current_request_take_precedence():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import build_audio_projection

    projection = build_audio_projection(
        run_id=91,
        task_id="task_retry",
        audio_request_id="wla_current",
        workflow_run=_workflow_run(status="completed", metadata=_watchlist_meta()),
        artifacts=[
            _artifact(
                "art_old_later",
                metadata=_watchlist_meta(audio_request_id="wla_old", final_artifact=True, title="Old final"),
                created_at="2026-05-22T12:00:00Z",
            ),
            _artifact(
                "art_raw_current",
                metadata=_watchlist_meta(final_artifact=True, title="Raw final"),
                created_at="2026-05-22T10:00:00Z",
            ),
            _artifact(
                "art_mixed_current",
                metadata=_watchlist_meta(background_mixed=True, final_artifact=True, title="Mixed final"),
                created_at="2026-05-22T10:01:00Z",
            ),
        ],
    )

    assert projection["final_artifact"]["artifact_id"] == "art_mixed_current"
    assert projection["final_artifact"]["title"] == "Mixed final"


def test_extract_workflow_run_metadata_uses_compatible_sources_in_order():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import extract_workflow_run_metadata

    assert extract_workflow_run_metadata(
        _workflow_run(
            metadata={"audio_request_id": "wla_metadata"},
            definition_metadata={"audio_request_id": "wla_definition"},
            inputs={"audio_request_id": "wla_inputs"},
        )
    )["audio_request_id"] == "wla_metadata"
    assert extract_workflow_run_metadata(
        _workflow_run(
            metadata=None,
            definition_metadata={"audio_request_id": "wla_definition"},
            inputs={"audio_request_id": "wla_inputs"},
        )
    )["audio_request_id"] == "wla_definition"
    assert extract_workflow_run_metadata(
        _workflow_run(metadata=None, definition_metadata=None, inputs={"audio_request_id": "wla_inputs"})
    )["audio_request_id"] == "wla_inputs"


def test_status_and_download_url_normalization():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import (
        artifact_download_url,
        normalize_audio_status,
    )

    assert normalize_audio_status("succeeded") == "completed"
    assert normalize_audio_status("in_progress") == "running"
    assert normalize_audio_status(None, task_id="task_pending") == "pending"
    assert artifact_download_url("art id/1", target_user_id=2) == "/api/v1/workflows/artifacts/art%20id%2F1/download"


def test_merge_and_stale_helpers_preserve_unrelated_metadata():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import (
        mark_audio_projection_stale,
        merge_audio_projection_metadata,
    )

    existing = {
        "delivery_status": "sent",
        "template_name": "daily_digest",
        "chatbook_path": "/tmp/digest.chatbook",
        "audio": {"status": "completed", "audio_request_id": "wla_old", "final_artifact": {"artifact_id": "old"}},
    }
    projection = {
        "status": "completed",
        "audio_request_id": "wla_current",
        "artifact_id": "art_final",
        "final_artifact": {"artifact_id": "art_final"},
    }

    merged = merge_audio_projection_metadata(existing, projection)

    assert merged["delivery_status"] == "sent"
    assert merged["template_name"] == "daily_digest"
    assert merged["chatbook_path"] == "/tmp/digest.chatbook"
    assert merged["audio"] == projection
    assert merged["audio_briefing_status"] == "completed"
    assert merged["audio_request_id"] == "wla_current"

    stale = mark_audio_projection_stale(merged, superseded_by="wla_next")

    assert "audio" not in stale
    assert stale["previous_audio"]["stale"] is True
    assert stale["previous_audio"]["superseded_by"] == "wla_next"
    assert stale["previous_audio"]["final_artifact"]["artifact_id"] == "art_final"


def test_mirror_audio_projection_updates_run_and_canonical_output_metadata():
    from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import mirror_audio_projection

    run = SimpleNamespace(id=91, stats_json=json.dumps({"items": 3}))
    output = SimpleNamespace(
        id=7,
        metadata_json=json.dumps({"template_name": "daily_digest", "delivery_status": "sent"}),
    )
    run_db = SimpleNamespace(update_run=lambda run_id, **kwargs: kwargs)
    collections_db = SimpleNamespace(
        list_output_artifacts=lambda **kwargs: ([output], 1),
        update_output_artifact_metadata=lambda output_id, **kwargs: kwargs,
    )
    run_updates: list[dict[str, Any]] = []
    output_updates: list[dict[str, Any]] = []
    run_db.update_run = lambda run_id, **kwargs: run_updates.append({"run_id": run_id, **kwargs})
    collections_db.update_output_artifact_metadata = (
        lambda output_id, **kwargs: output_updates.append({"output_id": output_id, **kwargs})
    )

    projection = {
        "status": "completed",
        "audio_request_id": "wla_current",
        "artifact_id": "art_final",
        "final_artifact": {"artifact_id": "art_final"},
    }

    assert mirror_audio_projection(run_db, collections_db, run, projection, user_id=1) is True
    persisted_stats = json.loads(run_updates[0]["stats_json"])
    persisted_output = json.loads(output_updates[0]["metadata_json"])

    assert persisted_stats["items"] == 3
    assert persisted_stats["audio"]["artifact_id"] == "art_final"
    assert persisted_output["template_name"] == "daily_digest"
    assert persisted_output["delivery_status"] == "sent"
    assert persisted_output["audio"]["artifact_id"] == "art_final"
