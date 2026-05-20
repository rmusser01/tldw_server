"""Tests for audio briefing workflow bridge.

Tests the trigger function, workflow input construction, and workflow definition.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestAudioBriefingWorkflowDefinition:
    """Tests for the built-in workflow definition."""

    def test_workflow_def_has_required_steps(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            AUDIO_BRIEFING_WORKFLOW_DEF,
        )

        step_ids = [s["id"] for s in AUDIO_BRIEFING_WORKFLOW_DEF["steps"]]
        assert "compose_script" in step_ids
        assert "clean_script" in step_ids
        assert "generate_audio" in step_ids

    def test_workflow_def_step_types(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            AUDIO_BRIEFING_WORKFLOW_DEF,
        )

        steps = AUDIO_BRIEFING_WORKFLOW_DEF["steps"]
        step_types = {s["id"]: s["type"] for s in steps}
        assert step_types["compose_script"] == "audio_briefing_compose"
        assert step_types["clean_script"] == "text_clean"
        assert step_types["generate_audio"] == "multi_voice_tts"

    def test_workflow_def_has_timeouts(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            AUDIO_BRIEFING_WORKFLOW_DEF,
        )

        for step in AUDIO_BRIEFING_WORKFLOW_DEF["steps"]:
            assert "timeout_seconds" in step, f"Step {step['id']} missing timeout"

    def test_workflow_def_passes_persona_and_background_inputs(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            AUDIO_BRIEFING_WORKFLOW_DEF,
        )

        compose_cfg = next(
            step["config"] for step in AUDIO_BRIEFING_WORKFLOW_DEF["steps"] if step["id"] == "compose_script"
        )
        assert compose_cfg["persona_summarize"] == "{{ inputs.persona_summarize }}"
        assert compose_cfg["persona_id"] == "{{ inputs.persona_id }}"
        assert compose_cfg["persona_provider"] == "{{ inputs.persona_provider }}"
        assert compose_cfg["persona_model"] == "{{ inputs.persona_model }}"

        audio_cfg = next(
            step["config"] for step in AUDIO_BRIEFING_WORKFLOW_DEF["steps"] if step["id"] == "generate_audio"
        )
        assert audio_cfg["background_audio_uri"] == "{{ inputs.background_audio_uri }}"
        assert audio_cfg["background_volume"] == "{{ inputs.background_volume }}"


class TestBuildWorkflowInputs:
    """Tests for _build_workflow_inputs."""

    def test_default_inputs(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            _build_workflow_inputs,
        )

        items = [{"title": "Test", "summary": "Summary"}]
        output_prefs = {"generate_audio": True}

        inputs = _build_workflow_inputs(items, output_prefs)

        assert inputs["items"] == items
        assert inputs["target_audio_minutes"] == 10
        assert inputs["audio_language"] == "en"
        assert inputs["tts_model"] == "kokoro"
        assert inputs["tts_voice"] == "af_heart"
        assert inputs["tts_speed"] == 1.0
        assert inputs["llm_provider"] is None
        assert inputs["llm_model"] is None
        assert inputs["voice_map"] is None
        assert inputs["persona_summarize"] is False
        assert inputs["persona_id"] is None
        assert inputs["persona_provider"] is None
        assert inputs["persona_model"] is None
        assert inputs["background_audio_uri"] is None
        assert inputs["background_volume"] == 0.15
        assert inputs["background_delay_ms"] == 0
        assert inputs["background_fade_seconds"] == 2.0

    def test_custom_inputs(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            _build_workflow_inputs,
        )

        items = [{"title": "News", "summary": "Story"}]
        output_prefs = {
            "generate_audio": True,
            "target_audio_minutes": 5,
            "audio_model": "tts-1",
            "audio_voice": "nova",
            "audio_speed": 1.2,
            "audio_language": "es",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "persona_summarize": True,
            "persona_id": "analyst",
            "persona_provider": "openai",
            "persona_model": "gpt-4o-mini",
            "voice_map": {"HOST": "af_bella"},
            "background_audio_uri": "file:///tmp/bed.mp3",
            "background_volume": 0.2,
            "background_delay_ms": 500,
            "background_fade_seconds": 3.0,
        }

        inputs = _build_workflow_inputs(items, output_prefs)

        assert inputs["target_audio_minutes"] == 5
        assert inputs["audio_language"] == "es"
        assert inputs["tts_model"] == "tts-1"
        assert inputs["tts_voice"] == "nova"
        assert inputs["tts_speed"] == 1.2
        assert inputs["llm_provider"] == "openai"
        assert inputs["llm_model"] == "gpt-4o"
        assert inputs["persona_summarize"] is True
        assert inputs["persona_id"] == "analyst"
        assert inputs["persona_provider"] == "openai"
        assert inputs["persona_model"] == "gpt-4o-mini"
        assert inputs["voice_map"] == {"HOST": "af_bella"}
        assert inputs["background_audio_uri"] == "file:///tmp/bed.mp3"
        assert inputs["background_volume"] == 0.2
        assert inputs["background_delay_ms"] == 500
        assert inputs["background_fade_seconds"] == 3.0

    def test_structured_audio_cast_inputs_preserve_voice_map_compatibility(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            _build_workflow_inputs,
        )

        items = [{"title": "News", "summary": "Story"}]
        audio_cast = {
            "speaker_count": 2,
            "speakers": [
                {
                    "id": "host",
                    "label": "Host",
                    "role": "anchor",
                    "voice": "af_bella",
                    "persona": "calm",
                },
                {
                    "id": "analyst",
                    "label": "Analyst",
                    "voice": "am_adam",
                },
            ],
        }

        inputs = _build_workflow_inputs(
            items,
            {
                "generate_audio": True,
                "audio_cast": audio_cast,
                "voice_map": {"HOST": "af_heart"},
            },
        )

        assert inputs["audio_cast"] == audio_cast
        assert inputs["voice_map"] == {"HOST": "af_heart"}

        derived_inputs = _build_workflow_inputs(
            items,
            {
                "generate_audio": True,
                "audio_cast": audio_cast,
            },
        )

        assert derived_inputs["audio_cast"] == audio_cast
        assert derived_inputs["voice_map"] == {
            "HOST": "af_bella",
            "ANALYST": "am_adam",
        }

    def test_audio_cast_requires_matching_speaker_count(self):
        from pydantic import ValidationError

        from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import WatchlistAudioCast

        with pytest.raises(ValidationError, match="speaker_count_must_match_speakers_length"):
            WatchlistAudioCast(
                speaker_count=2,
                speakers=[
                    {
                        "id": "host",
                        "label": "Host",
                        "voice": "af_bella",
                    }
                ],
            )

    def test_audio_cast_requires_unique_speaker_ids(self):
        from pydantic import ValidationError

        from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import WatchlistAudioCast

        with pytest.raises(ValidationError, match="speaker_ids_must_be_unique"):
            WatchlistAudioCast(
                speaker_count=2,
                speakers=[
                    {"id": "host", "label": "Host", "voice": "af_bella"},
                    {"id": "host", "label": "Second host", "voice": "am_adam"},
                ],
            )


class TestTriggerAudioBriefing:
    """Tests for trigger_audio_briefing."""

    class SubmitOnlyScheduler:
        """Scheduler double that exposes only the real submit API."""

        def __init__(self, return_value: str) -> None:
            self.return_value = return_value
            self.submit = AsyncMock(side_effect=self._submit)

        async def _submit(self, *args: Any, **kwargs: Any) -> str:
            metadata = kwargs.get("metadata")
            user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("Task metadata must include a non-empty 'user_id'")
            return self.return_value

    @pytest.mark.asyncio
    async def test_trigger_skips_when_generate_audio_false(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            trigger_audio_briefing,
        )

        result = await trigger_audio_briefing(
            user_id=1,
            job_id=1,
            run_id=1,
            output_prefs={"generate_audio": False},
            db=MagicMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_skips_when_no_items(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            trigger_audio_briefing,
        )

        db = MagicMock()
        db.list_items.return_value = ([], 0)

        result = await trigger_audio_briefing(
            user_id=1,
            job_id=1,
            run_id=1,
            output_prefs={"generate_audio": True},
            db=db,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_submits_workflow(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            AUDIO_BRIEFING_WORKFLOW_DEF,
            trigger_audio_briefing,
        )

        db = MagicMock()
        db.list_items.return_value = (
            [
                {"title": "Story 1", "summary": "Summary 1", "url": "https://example.com/1"},
                {"title": "Story 2", "summary": "Summary 2", "url": "https://example.com/2"},
            ],
            2,
        )

        mock_scheduler = self.SubmitOnlyScheduler("task_abc123")

        async def run_sync_in_test(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch(
                "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.run_in_threadpool",
                new_callable=AsyncMock,
                side_effect=run_sync_in_test,
                create=True,
            ) as mock_threadpool,
            patch(
                "tldw_Server_API.app.core.Scheduler.get_global_scheduler",
                new_callable=AsyncMock,
                return_value=mock_scheduler,
            ),
        ):
            result = await trigger_audio_briefing(
                user_id=1,
                job_id=42,
                run_id=7,
                output_prefs={
                    "generate_audio": True,
                    "target_audio_minutes": 5,
                    "voice_map": {"HOST": "af_bella"},
                    "background_audio_uri": "file:///tmp/bed.mp3",
                    "background_volume": 0.22,
                    "persona_summarize": True,
                    "persona_id": "host_style",
                },
                db=db,
            )

        assert result == "task_abc123"
        mock_scheduler.submit.assert_awaited_once()

        # Verify the workflow submission payload
        args = mock_scheduler.submit.call_args.args
        kwargs = mock_scheduler.submit.call_args.kwargs
        assert args == ("workflow_run",)
        assert kwargs["queue_name"] == "workflows"
        assert kwargs["idempotency_key"] == "watchlist-audio-briefing:1:42:7"
        assert kwargs["metadata"] == {
            "source": "watchlist_audio_briefing",
            "watchlist_job_id": 42,
            "watchlist_run_id": 7,
            "user_id": "1",
        }
        payload = kwargs["payload"]
        assert payload["user_id"] == 1
        assert payload["definition_snapshot"] == AUDIO_BRIEFING_WORKFLOW_DEF
        assert payload["mode"] == "async"
        assert payload["inputs"]["target_audio_minutes"] == 5
        assert payload["inputs"]["voice_map"] == {"HOST": "af_bella"}
        assert payload["inputs"]["background_audio_uri"] == "file:///tmp/bed.mp3"
        assert payload["inputs"]["background_volume"] == 0.22
        assert payload["inputs"]["persona_summarize"] is True
        assert payload["inputs"]["persona_id"] == "host_style"
        assert len(payload["inputs"]["items"]) == 2
        assert payload["metadata"]["watchlist_job_id"] == 42
        assert payload["metadata"]["watchlist_run_id"] == 7
        mock_threadpool.assert_awaited_once()
        db.list_items.assert_called_once_with(run_id=7, status="ingested", limit=100, offset=0)

    @pytest.mark.asyncio
    async def test_trigger_handles_scheduler_failure(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            trigger_audio_briefing,
        )

        db = MagicMock()
        db.list_items.return_value = (
            [
                {"title": "Story", "summary": "S", "url": "https://x.com"},
            ],
            1,
        )

        with patch(
            "tldw_Server_API.app.core.Scheduler.get_global_scheduler",
            new_callable=AsyncMock,
            side_effect=RuntimeError("scheduler not available"),
        ):
            result = await trigger_audio_briefing(
                user_id=1,
                job_id=1,
                run_id=1,
                output_prefs={"generate_audio": True},
                db=db,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_handles_db_error(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            trigger_audio_briefing,
        )

        db = MagicMock()
        db.list_items.side_effect = RuntimeError("db error")

        result = await trigger_audio_briefing(
            user_id=1,
            job_id=1,
            run_id=1,
            output_prefs={"generate_audio": True},
            db=db,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_submits_with_object_rows(self):
        from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
            trigger_audio_briefing,
        )

        db = MagicMock()
        db.list_items.return_value = (
            [SimpleNamespace(title="Story Obj", summary="Summary Obj", url="https://example.com/obj")],
            1,
        )

        mock_scheduler = self.SubmitOnlyScheduler("task_object_row")

        with patch(
            "tldw_Server_API.app.core.Scheduler.get_global_scheduler",
            new_callable=AsyncMock,
            return_value=mock_scheduler,
        ):
            result = await trigger_audio_briefing(
                user_id=1,
                job_id=99,
                run_id=123,
                output_prefs={"generate_audio": True},
                db=db,
            )

        assert result == "task_object_row"
        payload = mock_scheduler.submit.call_args.kwargs["payload"]
        assert payload["inputs"]["items"] == [
            {"title": "Story Obj", "summary": "Summary Obj", "url": "https://example.com/obj"}
        ]
