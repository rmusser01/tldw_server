"""The comparison harness must distinguish measured behavior from model quality."""

from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import PreparedResult, ToolResultContext

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_comparison_preserves_baseline_and_recovers_expected_evidence() -> None:
    """All comparison modes must retain recoverable evidence and label oracle measurements."""
    from Helper_Scripts.benchmarks.acp_tool_result_experiment import compare_case

    text = "Background detail. " * 700 + "The launch code is amber."
    reports = await compare_case(
        {
            "id": "late-evidence",
            "question": "What is the launch code?",
            "text": text,
            "expected_quote": "The launch code is amber.",
        }
    )
    assert [r["mode"] for r in reports] == ["off", "excerpt", "worker"]
    assert reports[0]["output_bytes"] == len(text.encode())
    assert all(r["evidence_recovered"] for r in reports)
    assert reports[2]["worker_kind"] == "fixture_oracle"
    assert reports[2]["worker_usage"] is None
    assert reports[2]["output_bytes"] < reports[0]["output_bytes"]
    assert all(r["main_model_usage"] is None for r in reports)


@pytest.mark.asyncio
async def test_invalid_ground_truth_is_rejected() -> None:
    """A comparison must reject expected evidence that is absent from its source."""
    from Helper_Scripts.benchmarks.acp_tool_result_experiment import compare_case

    with pytest.raises(ValueError, match="expected_quote"):
        await compare_case({"id": "invalid", "question": "q", "text": "source", "expected_quote": "invented"})


def test_cli_emits_parseable_report_and_labels_simulated_worker(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI report must identify fixture-oracle scope and remain valid JSON."""
    from Helper_Scripts.benchmarks.acp_tool_result_experiment import main

    main([])
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == "tool_result_policy_replay"
    assert report["worker_kind"] == "fixture_oracle"
    assert len(report["results"]) >= 6
    assert report["measures_task_success"] is False


@pytest.mark.asyncio
async def test_recovery_metric_requires_actual_returned_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery credit must depend on delivered source text, including rereads."""
    from Helper_Scripts.benchmarks.acp_tool_result_experiment import compare_case

    original_read = ToolResultContext.read

    def broken_read(self: ToolResultContext, source_id: str, offset: int = 0, limit: int = 512) -> PreparedResult:
        """Preserve range metadata while simulating a source-delivery failure."""
        result = original_read(self, source_id, offset, limit)
        return PreparedResult("No source evidence was returned.", result.metadata)

    monkeypatch.setattr(ToolResultContext, "read", broken_read)
    reports = await compare_case(
        {
            "id": "recovery-check",
            "question": "Describe zephyr.",
            "text": "unrelated " * 1500 + "Launch code is amber.",
            "expected_quote": "Launch code is amber.",
        }
    )
    excerpt = next(row for row in reports if row["mode"] == "excerpt")
    assert excerpt["evidence_present"] is False
    assert excerpt["scripted_reread_count"] > 0
    assert excerpt["evidence_recovered"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_quote", ["Excerpts only.", "characters."])
async def test_generated_header_cannot_satisfy_expected_source_evidence(expected_quote: str) -> None:
    """Generated wrapper text must never substitute for the expected source quote."""
    from Helper_Scripts.benchmarks.acp_tool_result_experiment import compare_case

    reports = await compare_case(
        {
            "id": "header-collision",
            "question": "Describe zephyr.",
            "text": "Unrelated background. " * 700 + expected_quote,
            "expected_quote": expected_quote,
        }
    )
    excerpt = next(row for row in reports if row["mode"] == "excerpt")
    assert excerpt["evidence_present"] is False
    assert excerpt["scripted_reread_count"] == 1
    assert excerpt["evidence_recovered"] is True
