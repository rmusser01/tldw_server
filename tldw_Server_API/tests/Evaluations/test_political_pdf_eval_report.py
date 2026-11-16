"""Offline evidence reports retain failures and never manufacture semantic scores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    _json(tmp_path / "manifest.json", {
        "schema_version": 1, "source_sha256": "a" * 64, "fixture_sha256": "b" * 64,
        "git_sha": "revision", "config": {"source": {"media_id": 42, "pdf_pages": 1000},
            "models": [{"key": "luna", "provider": "openai", "model": "gpt-6-luna"}]},
        "cases": [{"id": "H05", "question": "What is the proposal and its condition?",
            "source_pages": [{"pdf_page": 871}, {"pdf_page": 872}],
            "expected_source_facts": ["A proposal with a condition, not enacted law."],
            "human_review": "pending", "label_provenance": "tuned regression"}],
    })
    (tmp_path / "attempts.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def _event(run_dir: Path, *, event: str = "finished", attempt: int = 1, stage: str = "qa",
           operation_id: str = "qa-luna-H05", **fields: object) -> None:
    record = {"schema_version": 1, "operation_id": operation_id, "attempt": attempt,
        "event": event, "stage": stage, "case_id": "H05", "model_key": "luna",
        "context": "reference", "utc_time": "2026-11-15T20:00:00Z", **fields}
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _qa(run_dir: Path, *, attempt: int = 1, finish_reason: str = "stop",
        content: object = None, usage: object = None, **fields: object) -> None:
    request_path, response_path = f"requests/qa-{attempt}.json", f"responses/qa-{attempt}.json"
    _json(run_dir / request_path, {"method": "POST", "path": "/api/v1/chat/completions",
        "body": {"model": "gpt-6-luna", "messages": []},
        "context_evidence": {"passages": [{"id": "P1", "text": "source text",
            "start_char": 0, "end_char": 11}], "pages": [871, 872], "media_id": 42}})
    if content is None:
        content = json.dumps({"summary": "A proposed change.", "evidence": [
            {"claim": "A proposed change.", "source_span_ids": ["P1"]}]})
    body = {"model": "gpt-6-luna-2026-11-01", "choices": [{"finish_reason": finish_reason,
        "message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    _json(run_dir / response_path, {"status_code": 200, "body": body})
    _event(run_dir, event="started", attempt=attempt, request_path=request_path)
    _event(run_dir, attempt=attempt, outcome="success", response_path=response_path,
        elapsed_seconds=2.5, http_status=200, checks={"structured_output_valid": True}, **fields)


def _report(run_dir: Path) -> dict:
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report

    return write_report(run_dir)


def test_report_preserves_first_failure_latest_success_and_unknown_attempt(run_dir: Path) -> None:
    _event(run_dir, event="started")
    _event(run_dir, outcome="failed", http_status=400, elapsed_seconds=1,
        checks={"completion_not_truncated": False})
    _qa(run_dir, attempt=2, usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    _event(run_dir, event="started", operation_id="interrupted", stage="summary")

    report = _report(run_dir)

    operation = report["operations"][0]
    assert (operation["first_outcome"], operation["latest_outcome"]) == ("failed", "success")
    assert [row["attempt"] for row in operation["attempts"]] == [1, 2]
    assert operation["attempts"][1]["reported_usage"]["total_tokens"] == 120
    assert operation["attempts"][1]["returned_model"] == "gpt-6-luna-2026-11-01"
    assert report["operations"][1]["latest_outcome"] == "unknown"
    assert report["models"][0]["first_attempt_failures"] == 1
    assert report["models"][0]["unknown_operations"] == 1
    assert json.loads((run_dir / "report.json").read_text()) == report
    markdown = (run_dir / "report.md").read_text()
    assert "failed" in markdown and "unknown" in markdown


def test_semantic_review_keeps_expected_facts_pending_and_marks_tuned_case(run_dir: Path) -> None:
    _qa(run_dir)

    report = _report(run_dir)
    packet = json.loads((run_dir / "semantic-review.json").read_text())

    review = packet["items"][0]
    assert review["expected_source_facts"] == ["A proposal with a condition, not enacted law."]
    assert review["status"] == "pending"
    assert review["labels"] == {"faithfulness": None, "author_attribution": None,
        "disagreement": None, "completeness": None}
    assert review["response_path"] == "responses/qa-1.json"
    assert "tuned regression" in review["case_note"]
    assert report["operations"][0]["attempts"][0]["usage_status"] == "unavailable"
    assert "semantic" in (run_dir / "report.md").read_text().lower()


def test_retrieval_uses_frozen_expected_pages_and_keeps_overbroad_ranges(run_dir: Path) -> None:
    _event(run_dir, event="started", stage="retrieval")
    _event(run_dir, stage="retrieval", outcome="success", checks={
        "expected_pages": [871], "covered_pages": [871, 999], "source_scope_valid": False,
        "returned_page_ranges": [[870, 999]], "overbroad_ranges": [[870, 999]], "pipeline_errors": []})

    metrics = _report(run_dir)["operations"][0]["attempts"][0]["metrics"]

    assert metrics["expected_pages"] == [871, 872]
    assert metrics["covered_expected_pages"] == [871]
    assert metrics["expected_page_coverage"] == 0.5
    assert metrics["source_scope_valid"] is False
    assert metrics["overbroad_ranges"] == [[870, 999]]


@pytest.mark.parametrize("content,reason,valid,citations", [
    ("not json", "length", False, False),
    (json.dumps({"summary": "Claim", "evidence": [{"claim": "Claim", "source_span_ids": ["P9"]}]}),
        "stop", True, False),
    (json.dumps({"summary": "Claim", "evidence": []}), "stop", False, False),
])
def test_qa_checks_saved_response_instead_of_trusting_success_flags(
    run_dir: Path, content: str, reason: str, valid: bool, citations: bool,
) -> None:
    _qa(run_dir, content=content, finish_reason=reason)

    metrics = _report(run_dir)["operations"][0]["attempts"][0]["metrics"]

    assert metrics["structured_output_valid"] is valid
    assert metrics["citation_ids_valid"] is citations
    assert metrics["completion_not_truncated"] is (reason == "stop")


def test_summary_retains_partial_counts_and_never_estimates_job_usage(run_dir: Path) -> None:
    _event(run_dir, event="started", stage="summary")
    _json(run_dir / "responses/job.json", {"status_code": 200, "body": {"usage": {"total_tokens": 99}}})
    _event(run_dir, stage="summary", outcome="partial", elapsed_seconds=30,
        response_path="responses/job.json", checks={"job_status": "completed",
            "completed_sections": 2, "failed_sections": 1, "total_sections": 3,
            "completed_portions": 5, "failed_portions": 1, "artifact_readback_matches": True})

    attempt = _report(run_dir)["operations"][0]["attempts"][0]

    assert attempt["metrics"]["completed_sections"] == 2
    assert attempt["metrics"]["failed_sections"] == 1
    assert attempt["elapsed_seconds"] == 30
    assert attempt["usage_status"] == "unavailable"
    assert attempt["reported_usage"] is None


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json", "responses/link.json"])
def test_evidence_paths_cannot_escape_run_directory(run_dir: Path, path: str) -> None:
    outside = run_dir.parent / "outside.json"
    _json(outside, {"secret": "never read"})
    (run_dir / "responses").mkdir()
    (run_dir / "responses/link.json").symlink_to(outside)
    _event(run_dir, event="started", request_path=path)

    with pytest.raises(ValueError, match="path"):
        _report(run_dir)
    assert not (run_dir / "report.json").exists()


@pytest.mark.parametrize("malformation", ["duplicate", "orphan", "metadata", "invalid-json", "unknown-case"])
def test_malformed_ledger_fails_without_publishing_report(run_dir: Path, malformation: str) -> None:
    if malformation == "orphan":
        _event(run_dir, outcome="success")
    elif malformation == "invalid-json":
        (run_dir / "attempts.jsonl").write_text('{"partial":', encoding="utf-8")
    else:
        _event(run_dir, event="started")
        if malformation == "duplicate":
            _event(run_dir, event="started")
        else:
            _event(run_dir, outcome="success", **({"model_key": "another"} if malformation == "metadata"
                else {"case_id": "unknown"}))

    with pytest.raises(ValueError):
        _report(run_dir)
    assert not (run_dir / "report.json").exists()


def test_report_does_not_copy_request_credentials_or_raw_config(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"]["unrelated_private_field"] = "do-not-copy-config"
    _json(run_dir / "manifest.json", manifest)
    _qa(run_dir)
    request = json.loads((run_dir / "requests/qa-1.json").read_text())
    request["headers"] = {"Authorization": "Bearer do-not-copy-request"}
    _json(run_dir / "requests/qa-1.json", request)

    _report(run_dir)

    outputs = "".join((run_dir / name).read_text() for name in (
        "report.json", "report.md", "semantic-review.json"))
    assert "do-not-copy" not in outputs


def test_regenerating_report_preserves_human_labels_and_notes_for_same_evidence(run_dir: Path) -> None:
    _qa(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    packet["items"][0].update({"status": "reviewed", "reviewer": "reader",
        "notes": "The condition was omitted."})
    packet["items"][0]["labels"]["completeness"] = "fail"
    _json(path, packet)

    _report(run_dir)

    reviewed = json.loads(path.read_text())["items"][0]
    assert reviewed["status"] == "reviewed"
    assert reviewed["labels"]["completeness"] == "fail"
    assert reviewed["notes"] == "The condition was omitted."
    assert reviewed["reviewer"] == "reader"


def test_changed_evidence_does_not_reuse_or_destroy_previous_human_labels(run_dir: Path) -> None:
    _qa(run_dir)
    _report(run_dir)
    review_before = (run_dir / "semantic-review.json").read_bytes()
    response = json.loads((run_dir / "responses/qa-1.json").read_text())
    response["body"]["choices"][0]["message"]["content"] = "different output"
    _json(run_dir / "responses/qa-1.json", response)

    with pytest.raises(ValueError, match="changed"):
        _report(run_dir)
    assert (run_dir / "semantic-review.json").read_bytes() == review_before


@pytest.mark.parametrize("usage", [{}, {"total_tokens": -1}, {"total_tokens": float("nan")},
    {"total_tokens": "120"}, {"arbitrary_number": 120}])
def test_missing_or_invalid_usage_is_unavailable_not_zero(run_dir: Path, usage: dict) -> None:
    _qa(run_dir, usage=usage)

    attempt = _report(run_dir)["operations"][0]["attempts"][0]

    assert attempt["usage_status"] == "unavailable"
    assert attempt["reported_usage"] is None


def test_unstarted_model_has_no_fabricated_latency_or_success(run_dir: Path) -> None:
    model = _report(run_dir)["models"][0]

    assert model["requested_model"] == "gpt-6-luna"
    assert model["operation_count"] == 0
    assert model["reported_elapsed_seconds"] is None
    assert model["latest_outcomes"] == {}


def test_report_cannot_overwrite_external_file_through_output_symlink(run_dir: Path) -> None:
    outside = run_dir.parent / "existing-review.txt"
    outside.write_text("preserve external data", encoding="utf-8")
    (run_dir / "report.md").symlink_to(outside)

    with pytest.raises(ValueError, match="path"):
        _report(run_dir)
    assert outside.read_text() == "preserve external data"
    assert not (run_dir / "report.json").exists()


@pytest.mark.parametrize("fields", [{"utc_time": "yesterday"}, {"model_key": []},
    {"schema_version": True}, {"elapsed_seconds": float("inf")}])
def test_invalid_event_fields_are_rejected_before_any_report(run_dir: Path, fields: dict) -> None:
    _event(run_dir, event="started", **fields)

    with pytest.raises(ValueError):
        _report(run_dir)
    assert not (run_dir / "report.json").exists()


def test_completed_outcome_and_mechanical_details_are_visible_in_markdown(run_dir: Path) -> None:
    _qa(run_dir)
    text = (run_dir / "attempts.jsonl").read_text().replace('"outcome": "success"', '"outcome": "completed"')
    (run_dir / "attempts.jsonl").write_text(text, encoding="utf-8")
    _event(run_dir, operation_id="retrieval", event="started", stage="retrieval")
    _event(run_dir, operation_id="retrieval", stage="retrieval", outcome="completed",
        checks={"covered_pages": [871], "source_scope_valid": True, "overbroad_ranges": []})

    report = _report(run_dir)
    markdown = (run_dir / "report.md").read_text()

    assert report["models"][0]["first_attempt_failures"] == 0
    assert "gpt-6-luna-2026-11-01" in markdown
    assert "span IDs=True" in markdown
    assert "pages=1/2" in markdown


@pytest.mark.parametrize("output", ["report.json", "report.md", "semantic-review.json"])
def test_output_symlinks_inside_run_cannot_overwrite_immutable_inputs(run_dir: Path, output: str) -> None:
    _qa(run_dir)
    original = (run_dir / "manifest.json").read_bytes()
    (run_dir / output).symlink_to(run_dir / "manifest.json")

    with pytest.raises(ValueError, match="path"):
        _report(run_dir)
    assert (run_dir / "manifest.json").read_bytes() == original
    assert not any((run_dir / name).exists() for name in (
        "report.json", "report.md", "semantic-review.json") if name != output)


def test_atomic_report_replacement_preserves_hardlinked_input(run_dir: Path) -> None:
    original = (run_dir / "manifest.json").read_bytes()
    (run_dir / "report.json").hardlink_to(run_dir / "manifest.json")

    _report(run_dir)

    assert (run_dir / "manifest.json").read_bytes() == original


def test_summary_reports_job_elapsed_basis_separately_from_local_record_time(run_dir: Path) -> None:
    _event(run_dir, event="started", stage="summary")
    _event(run_dir, stage="summary", outcome="completed", elapsed_seconds=0.001,
        checks={"job_elapsed_seconds": 40, "job_latency_basis": "including resume downtime"})

    report = _report(run_dir)
    markdown = (run_dir / "report.md").read_text()

    assert report["operations"][0]["attempts"][0]["metrics"]["job_elapsed_seconds"] == 40
    assert "job seconds=40" in markdown
    assert "including resume downtime" in markdown


def test_semantic_packet_includes_frozen_physical_pages_for_source_lookup(run_dir: Path) -> None:
    _qa(run_dir)
    _report(run_dir)

    packet = json.loads((run_dir / "semantic-review.json").read_text())

    assert packet["source_sha256"] == "a" * 64
    assert packet["items"][0]["source_pages"] == [{"pdf_page": 871}, {"pdf_page": 872}]


def test_pending_job_poll_is_visible_without_counting_it_as_failed(run_dir: Path) -> None:
    _event(run_dir, event="started", stage="summary_poll")
    _event(run_dir, stage="summary_poll", outcome="pending")

    report = _report(run_dir)

    assert report["models"][0]["first_attempt_failures"] == 0
    assert report["models"][0]["latest_outcomes"] == {"pending": 1}


def test_derived_report_name_cannot_be_used_as_immutable_evidence(run_dir: Path) -> None:
    _qa(run_dir)
    source = run_dir / "requests/qa-1.json"
    original = source.read_bytes()
    source.rename(run_dir / "report.json")
    ledger = (run_dir / "attempts.jsonl").read_text().replace("requests/qa-1.json", "report.json")
    (run_dir / "attempts.jsonl").write_text(ledger, encoding="utf-8")

    with pytest.raises(ValueError, match="path"):
        _report(run_dir)
    assert (run_dir / "report.json").read_bytes() == original
