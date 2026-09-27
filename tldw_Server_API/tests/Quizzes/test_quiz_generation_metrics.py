"""Exercise generation observability through the public service and real registry."""

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import ArtifactVerificationResult
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import (
    BadRequestError,
    OsceCitationError,
    OsceProviderError,
    OsceVerificationError,
    QuizMalformedOutputError,
)
from tldw_Server_API.app.core.Metrics import metrics_manager
from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry
from tldw_Server_API.app.services import osce_generator, quiz_generation_metrics, quiz_generator

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> MetricsRegistry:
    """Use an isolated real registry without changing other tests' singleton."""
    value = MetricsRegistry()
    monkeypatch.setattr(metrics_manager, "_metrics_registry", value)
    return value


@pytest.fixture
def generation_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """Provide real SQLite persistence and a deterministic source-backed generator."""
    monkeypatch.setenv("TEST_MODE", "1")
    db = CharactersRAGDB(str(tmp_path / "quizzes.db"), client_id="metrics-test")
    media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="metrics-test")
    note_id = db.add_note(
        title="Private source title",
        content="Warfarin requires INR monitoring and review of important bleeding warning signs.",
    )
    try:
        yield {
            "db": db,
            "media_db": media_db,
            "sources": [{"source_type": "note", "source_id": str(note_id)}],
            "num_questions": 2,
        }
    finally:
        db.close_connection()
        media_db.close_connection()


@pytest.mark.parametrize(
    "profile",
    ["standard_recall", "mixed_assessment", "best_of_five", "emq", "assertion_reasoning", "osce_scenario"],
)
async def test_success_records_one_generation_for_every_profile(
    profile: str,
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Count one request, one successful outcome, and one duration across all profiles."""
    result = await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile=profile)

    assert result["quiz"]["id"]
    labels = {"profile": profile, "source_type": "note"}
    assert registry.get_cumulative_counter("quiz_generation_requests_total", labels) == 1
    outcome = {**labels, "outcome": "success"}
    assert registry.get_cumulative_counter("quiz_generation_outcomes_total", outcome) == 1
    stats = registry.get_metric_stats("quiz_generation_duration_seconds", labels=outcome)
    assert stats["count"] == 1
    assert stats["latest"] >= 0


async def test_request_is_counted_while_generation_is_in_flight(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose started work before a provider has produced its final outcome."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def provider(**kwargs: object) -> dict:
        """Hold generation open until the test releases the provider."""
        entered.set()
        await release.wait()
        return {"questions": []}

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    task = asyncio.create_task(quiz_generator.generate_quiz_from_sources(**generation_args))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert (
            registry.get_cumulative_counter(
                "quiz_generation_requests_total",
                {"profile": "standard_recall", "source_type": "note"},
            )
            == 1
        )
        assert registry.get_cumulative_counter_total("quiz_generation_outcomes_total") == 0
    finally:
        release.set()
        with pytest.raises(ValueError, match="No valid questions"):
            await task


@pytest.mark.parametrize("duplicate", [False, True])
async def test_source_labels_collapse_mixed_types_but_not_duplicate_notes(
    duplicate: bool,
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Count a mixed-source invocation once, independent of source count or identifiers."""
    if duplicate:
        generation_args["sources"].append(dict(generation_args["sources"][0]))
    else:
        card_id = generation_args["db"].add_flashcard({"front": "Warfarin", "back": "Regular INR monitoring."})
        generation_args["sources"].append({"source_type": "flashcard_card", "source_id": str(card_id)})

    await quiz_generator.generate_quiz_from_sources(**generation_args)

    assert (
        registry.get_cumulative_counter(
            "quiz_generation_requests_total",
            {
                "profile": "standard_recall",
                "source_type": "note" if duplicate else "mixed",
            },
        )
        == 1
    )
    assert registry.get_cumulative_counter_total("quiz_generation_requests_total") == 1


@pytest.mark.parametrize("raw_response", ["not valid JSON", {"questions": []}])
async def test_malformed_output_is_validation_failure(
    raw_response: object,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differentiate rejected generated output from a failed provider invocation."""
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=raw_response))
    with pytest.raises(ValueError):
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": "note",
                "outcome": "validation_error",
            },
        )
        == 1
    )
    assert generation_args["db"].list_quizzes()["count"] == 0


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, TimeoutError])
async def test_provider_errors_preserve_exception_and_are_not_validation_failures(
    error_type: type[Exception],
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify by the failing phase, not just a provider's exception base class."""
    error = error_type("private provider response")
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(side_effect=error))
    with pytest.raises(error_type) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert caught.value is error
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": "note",
                "outcome": "provider_error",
            },
        )
        == 1
    )


@pytest.mark.parametrize("profile", ["standard_recall", "osce_scenario"])
async def test_persistence_error_is_runtime_failure_even_when_wrapped_by_osce(
    profile: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not mislabel OSCE persistence's existing verification error as invalid output."""
    method = "create_quiz_with_osce_stations_atomic" if profile == "osce_scenario" else "create_quiz"
    error = ValueError("private database detail")
    monkeypatch.setattr(generation_args["db"], method, Mock(side_effect=error))
    with pytest.raises(OsceVerificationError if profile == "osce_scenario" else ValueError) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile=profile)
    assert (caught.value.__cause__ if profile == "osce_scenario" else caught.value) is error
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": profile,
                "source_type": "note",
                "outcome": "runtime_error",
            },
        )
        == 1
    )


@pytest.mark.parametrize(
    ("error_type", "outcome"),
    [
        (OsceCitationError, "validation_error"),
        (OsceVerificationError, "validation_error"),
        (OsceProviderError, "provider_error"),
    ],
)
async def test_osce_generation_errors_keep_their_public_identity(
    error_type: type[Exception],
    outcome: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use OSCE's typed pre-persistence errors without exposing private details."""
    error = error_type("private detail")
    monkeypatch.setattr(quiz_generator, "generate_osce_stations_from_sources", AsyncMock(side_effect=error))
    with pytest.raises(error_type) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile="osce_scenario")
    assert caught.value is error
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "osce_scenario",
                "source_type": "note",
                "outcome": outcome,
            },
        )
        == 1
    )


async def test_cancellation_is_preserved_and_observed(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record cancellation once without swallowing or replacing CancelledError."""
    error = asyncio.CancelledError()
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(side_effect=error))
    with pytest.raises(asyncio.CancelledError) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert caught.value is error
    labels = {"profile": "standard_recall", "source_type": "note", "outcome": "cancelled"}
    assert registry.get_cumulative_counter("quiz_generation_outcomes_total", labels) == 1
    assert registry.get_metric_stats("quiz_generation_duration_seconds", labels=labels)["count"] == 1


async def test_unknown_profile_is_bounded_and_still_counted(
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Never put an arbitrary rejected profile name into metric labels."""
    with pytest.raises(BadRequestError):
        await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile="private-user-profile")
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "unknown",
                "source_type": "unknown",
                "outcome": "validation_error",
            },
        )
        == 1
    )
    assert registry.get_cumulative_counter_total("quiz_generation_requests_total") == 1
    assert "private-user-profile" not in registry.export_prometheus_format()


@pytest.mark.parametrize("fail_generation", [False, True])
@pytest.mark.parametrize("failure_point", ["lookup", "registration", "increment", "observe"])
async def test_metrics_failures_do_not_change_generation_results_or_errors(
    fail_generation: bool,
    failure_point: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry outages at each write boundary must not affect the underlying operation."""
    error = QuizMalformedOutputError("original private error")
    if fail_generation:
        monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(side_effect=error))

    def unavailable(*args: object, **kwargs: object) -> None:
        """Simulate an unavailable metrics registry without changing application dependencies."""
        raise RuntimeError("private metrics backend detail")

    if failure_point == "lookup":
        monkeypatch.setattr(metrics_manager, "get_metrics_registry", unavailable)
    elif failure_point == "registration":
        monkeypatch.setattr(metrics_manager, "_metrics_registry", None)
        monkeypatch.setattr(MetricsRegistry, "_register_standard_metrics", unavailable)
    else:
        monkeypatch.setattr(registry, failure_point, unavailable)
    if fail_generation:
        with pytest.raises(QuizMalformedOutputError) as caught:
            await quiz_generator.generate_quiz_from_sources(**generation_args)
        assert caught.value is error
    else:
        result = await quiz_generator.generate_quiz_from_sources(**generation_args)
        assert len(result["questions"]) == 2
    if failure_point == "increment":
        assert registry.get_metric_stats("quiz_generation_duration_seconds")["count"] == 1
    elif failure_point == "observe":
        assert registry.get_cumulative_counter_total("quiz_generation_outcomes_total") == 1


async def test_metric_definitions_survive_registry_reset_and_export_histogram(
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Export a registered histogram with explicit buckets after a registry reset."""
    registry.reset()
    await quiz_generator.generate_quiz_from_sources(**generation_args)
    exported = registry.export_prometheus_format()
    assert "# TYPE quiz_generation_requests_total counter" in exported
    assert "# TYPE quiz_generation_outcomes_total counter" in exported
    assert "# TYPE quiz_generation_duration_seconds histogram" in exported
    assert any(
        line.startswith("quiz_generation_duration_seconds_bucket{") and 'le="60"' in line
        for line in exported.splitlines()
    )
    assert "Private source title" not in exported
    assert generation_args["sources"][0]["source_id"] not in exported


async def test_duration_uses_monotonic_elapsed_time_even_for_failed_generation(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe the whole failed attempt rather than wall-clock timestamps or success only."""
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(time, "perf_counter", clock)

    async def provider(**kwargs: object) -> dict:
        """Advance a deterministic elapsed clock before returning invalid output."""
        clock.return_value = 102.5
        return {"questions": []}

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    with pytest.raises(ValueError, match="No valid questions"):
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    stats = registry.get_metric_stats(
        "quiz_generation_duration_seconds",
        labels={
            "profile": "standard_recall",
            "source_type": "note",
            "outcome": "validation_error",
        },
    )
    assert stats["latest"] == 2.5


@pytest.mark.parametrize(
    ("source_type", "label"),
    [
        ("media", "media"),
        ("note", "note"),
        ("flashcard_deck", "flashcard_deck"),
        ("flashcard_card", "flashcard_card"),
        ("quiz_attempt", "quiz_attempt"),
        ("quiz_attempt_question", "quiz_attempt_question"),
        ("private-source-type", "unknown"),
    ],
)
async def test_source_resolution_failures_keep_only_allowlisted_type_labels(
    source_type: str,
    label: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep source labels bounded even when resolution fails before generation begins."""
    generation_args["sources"] = [{"source_type": source_type, "source_id": "private-source-id"}]
    monkeypatch.setattr(
        quiz_generator, "resolve_quiz_sources", Mock(side_effect=RuntimeError("private source content"))
    )
    with pytest.raises(RuntimeError):
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": label,
                "outcome": "runtime_error",
            },
        )
        == 1
    )
    assert "private" not in registry.export_prometheus_format()


async def test_invalid_sources_record_known_profile_without_resolving_sources(
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Count attempts rejected before source resolution without losing the normalized profile."""
    generation_args["sources"] = []
    with pytest.raises(ValueError, match="At least one source"):
        await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile="bof")
    labels = {"profile": "best_of_five", "source_type": "unknown"}
    assert registry.get_cumulative_counter("quiz_generation_requests_total", labels) == 1
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                **labels,
                "outcome": "validation_error",
            },
        )
        == 1
    )


@pytest.mark.parametrize("failure", ["failed", "needs_revision", "provider"])
async def test_verification_rejection_is_distinct_from_verifier_failure(
    failure: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed source-grounding verdict is validation, while a verifier outage is operational."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setattr(
        quiz_generator,
        "_call_quiz_generation_llm",
        AsyncMock(
            return_value={
                "questions": [
                    {
                        "question_type": "multiple_choice",
                        "question_text": "Which test monitors warfarin?",
                        "options": ["INR", "HbA1c", "CRP", "Sodium"],
                        "correct_answer": 0,
                        "source_citations": [generation_args["sources"][0]],
                    }
                ]
            }
        ),
    )
    error = ValueError("private verifier response")
    verifier = (
        AsyncMock(side_effect=error)
        if failure == "provider"
        else AsyncMock(
            return_value=ArtifactVerificationResult(
                verdict=failure,
                report={},
                unit_results=[],
                metadata={},
            )
        )
    )
    monkeypatch.setattr(quiz_generator, "verify_generated_artifact_against_sources", verifier)
    with pytest.raises(ValueError) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    if failure == "provider":
        assert caught.value is error
    else:
        assert isinstance(caught.value, quiz_generator.QuizClaimVerificationError)
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": "note",
                "outcome": "provider_error" if failure == "provider" else "validation_error",
            },
        )
        == 1
    )


async def test_legacy_media_wrapper_does_not_double_count(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy entry point delegates to the same single observation boundary."""
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        Mock(
            return_value=[
                {
                    "source_type": "media",
                    "source_id": "42",
                    "text": "Warfarin requires INR monitoring.",
                }
            ]
        ),
    )
    await quiz_generator.generate_quiz_from_media(
        db=generation_args["db"],
        media_db=generation_args["media_db"],
        media_id=42,
        num_questions=1,
    )
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_requests_total",
            {
                "profile": "standard_recall",
                "source_type": "media",
            },
        )
        == 1
    )
    assert registry.get_cumulative_counter_total("quiz_generation_outcomes_total") == 1


async def test_concurrent_attempts_do_not_share_phase_or_profile_labels(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleaved provider and validation failures retain independent per-request state."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def provider(**kwargs: object) -> dict:
        """Interleave two different profiles at the provider boundary."""
        if kwargs.get("model") == "blocked-provider":
            entered.set()
            await release.wait()
            raise ValueError("provider failure")
        return {"questions": []}

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    blocked = asyncio.create_task(
        quiz_generator.generate_quiz_from_sources(
            **generation_args,
            generation_profile="best_of_five",
            model="blocked-provider",
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(ValueError, match="No valid questions"):
            await quiz_generator.generate_quiz_from_sources(**generation_args)
    finally:
        release.set()
        with pytest.raises(ValueError, match="provider failure"):
            await blocked
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "best_of_five",
                "source_type": "note",
                "outcome": "provider_error",
            },
        )
        == 1
    )
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": "note",
                "outcome": "validation_error",
            },
        )
        == 1
    )


@pytest.mark.parametrize("failure_point", ["clock", "logging"])
async def test_secondary_observability_failures_cannot_break_generation(
    failure_point: str,
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clock or fallback-logger failures must not turn successful generation into an error."""
    broken = Mock(side_effect=RuntimeError("private metrics failure"))
    if failure_point == "clock":
        monkeypatch.setattr(quiz_generation_metrics, "time", SimpleNamespace(perf_counter=broken))
    else:
        monkeypatch.setattr(metrics_manager, "get_metrics_registry", broken)
        monkeypatch.setattr(quiz_generation_metrics, "logger", SimpleNamespace(debug=broken))
    result = await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert len(result["questions"]) == 2
    if failure_point == "clock":
        assert registry.get_cumulative_counter_total("quiz_generation_outcomes_total") == 1


@pytest.mark.parametrize("error_type", [TimeoutError, ValueError])
async def test_osce_verifier_outage_is_operational_after_wrapping(
    error_type: type[Exception],
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real OSCE verifier boundary, including its public error wrapper."""
    error = error_type("private verification service failure")
    monkeypatch.setattr(osce_generator, "verify_generated_artifact_against_sources", AsyncMock(side_effect=error))
    with pytest.raises(OsceVerificationError) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args, generation_profile="osce_scenario")
    assert caught.value.__cause__ is error
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {"profile": "osce_scenario", "source_type": "note", "outcome": "provider_error"},
        )
        == 1
    )


@pytest.mark.parametrize("source_case", ["unsupported", "malformed_id", "missing", "empty"])
async def test_real_source_rejections_are_validation_failures(
    source_case: str,
    generation_args: dict,
    registry: MetricsRegistry,
) -> None:
    """Reject invalid selections without reporting a database or provider outage."""
    source = {"source_type": "note", "source_id": "missing-private-note"}
    if source_case == "unsupported":
        source["source_type"] = "untrusted-private-type"
    elif source_case == "malformed_id":
        source = {"source_type": "media", "source_id": "not-an-integer"}
    elif source_case == "empty":
        source["source_id"] = generation_args["db"].add_note(title="Empty", content="")
    generation_args["sources"] = [source]
    with pytest.raises(ValueError):
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {
                "profile": "standard_recall",
                "source_type": "unknown" if source_case == "unsupported" else source["source_type"],
                "outcome": "validation_error",
            },
        )
        == 1
    )


async def test_source_database_outage_remains_runtime_failure(
    generation_args: dict,
    registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver's database exception must not become a source validation rejection."""
    error = RuntimeError("private database connection failure")
    monkeypatch.setattr(generation_args["db"], "get_note_by_id", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await quiz_generator.generate_quiz_from_sources(**generation_args)
    assert caught.value is error
    assert (
        registry.get_cumulative_counter(
            "quiz_generation_outcomes_total",
            {"profile": "standard_recall", "source_type": "note", "outcome": "runtime_error"},
        )
        == 1
    )
