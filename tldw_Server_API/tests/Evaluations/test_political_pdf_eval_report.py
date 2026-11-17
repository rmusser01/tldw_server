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


@pytest.mark.parametrize("cite_second", [False, True])
def test_summary_report_recomputes_overview_coverage_from_frozen_evidence(run_dir: Path, cite_second: bool) -> None:
    sections = [
        {"title": title, "status": "completed", "batches": [{"status": "completed",
            "evidence": [{"id": evidence_id, "quote": "Exact source."}]}]}
        for title, evidence_id in [("First chapter", "E1.1.1"), ("Financial agencies", "E2.1.1")]
    ]
    second = " Agencies propose new rules [E2.1.1]."
    overview = "Offices should stay open [E1.1.1]."
    analysis = {"sections": sections, "synthesis": overview + (second if cite_second else "")}
    checks = {"passed": True, "structural_passed": True, "completed_sections": 2, "total_sections": 2}
    if not cite_second:
        analysis["overview_coverage"] = checks["overview_coverage"] = {"complete": True, "cited_sections": 2}
    response = run_dir / "responses/job.json"
    _json(response, {"status_code": 200, "body": {"status": "completed", "artifact": {
        "status": "complete", "content": f"{analysis['synthesis']}\n\n## Financial agencies\n{second.strip()}",
        "version_metadata": {"workspaceClient": {"data": {"sectionAnalysis": analysis}}},
    }}})
    _event(run_dir, event="started", stage="summary")
    _event(run_dir, stage="summary", outcome="completed", response_path="responses/job.json", checks=checks)
    frozen = response.read_bytes(), (run_dir / "attempts.jsonl").read_bytes()

    attempt = _report(run_dir)["operations"][0]["attempts"][0]

    assert attempt["metrics"]["overview_coverage"] == {
        "selected_sections": 2, "completed_sections": 2, "cited_sections": 2 if cite_second else 1,
        "missing_sections": [] if cite_second else [{"index": 1, "title": "Financial agencies"}], "complete": cite_second,
    }
    assert attempt["metrics"]["structural_passed"] is cite_second
    assert attempt["metrics"]["passed"] is cite_second
    assert attempt["checks"] == checks
    assert attempt["outcome"] == "completed"
    assert attempt["metrics"]["strict_verification"]["human_review"] == "pending"
    markdown = (run_dir / "report.md").read_text()
    assert f"Overview cites {2 if cite_second else 1}/2 completed chapters" in markdown
    assert ("Financial agencies" in markdown) is (not cite_second)
    packet = json.loads((run_dir / "semantic-review.json").read_text())
    assert packet["items"][0]["status"] == "pending"
    assert all(label is None for label in packet["items"][0]["labels"].values())
    assert (response.read_bytes(), (run_dir / "attempts.jsonl").read_bytes()) == frozen


@pytest.mark.parametrize("response", [{"status_code": 200}, {"status_code": 200, "body": []}])
def test_summary_report_rejects_stale_acceptance_without_readable_overview(run_dir: Path, response: dict) -> None:
    _json(run_dir / "responses/job.json", response)
    _event(run_dir, event="started", stage="summary")
    checks = {"passed": True, "structural_passed": True, "overview_coverage": {"complete": True},
              "strict_verification": {"all_claims_supported": True}}
    _event(run_dir, stage="summary", outcome="completed", response_path="responses/job.json", checks=checks)

    attempt = _report(run_dir)["operations"][0]["attempts"][0]

    assert attempt["metrics"]["passed"] is False
    assert attempt["metrics"]["structural_passed"] is False
    assert "overview_coverage" not in attempt["metrics"]
    assert "strict_verification" not in attempt["metrics"]
    assert attempt["checks"] == checks
    assert attempt["outcome"] == "completed"
    item = json.loads((run_dir / "semantic-review.json").read_text())["items"][0]
    assert item["status"] == "pending"
    assert all(label is None for label in item["labels"].values())


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


def _strict_report(text: str, status: str = "supported") -> dict:
    import hashlib
    evaluated = status in {"supported", "unsupported", "contradicted"}
    return {"version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verdict": "supported" if status == "supported" else "needs_revision" if status in {"unsupported", "contradicted", "uncited"} else "unverified",
        "provider": "openai", "model": "test-verifier", "context": {"scope": "test"}, "limits": {"max_units": 256},
        "coverage": {"total": 1, "evaluated": int(evaluated), "supported": int(status == "supported"), "uncovered": int(not evaluated), "complete": evaluated},
        "findings": [{"id": "C1", "text": text, "start": 0, "end": len(text), "evidence_ids": ["P1"], "status": status, "reason": "Recorded cited-source verdict"}],
        "evidence": {"P1": {"text_sha256": hashlib.sha256(b"Exact source.").hexdigest(), "media_id": 42}}}


@pytest.mark.parametrize(("status", "expected"), [
    ("supported", "1/1"), ("unsupported", "0/1"), ("stale", "unavailable"), ("missing", "unavailable"),
    ("retained", "unscored (synthesis failed)"),
])
def test_summary_markdown_shows_verified_overview_counts_separately_from_chapters(run_dir, status, expected):
    text = "Claim [P1]"
    section = {"status": "completed", "content": text,
               "batches": [{"evidence": [{"id": "P1", "quote": "Exact source."}]}]}
    analysis = {"sections": [section], "synthesis": text}
    if status != "missing":
        receipt = _strict_report(text, "supported" if status in ("stale", "retained") else status)
        receipt["model"] = "other-model" if status == "stale" else "gpt-6-luna"
        analysis["synthesis_verification"] = receipt
    if status == "retained":
        analysis["synthesis_error_code"] = "unknown_evidence_reference"
    _json(run_dir / "response.json", {"body": {"artifact": {
        "version_metadata": {"workspaceClient": {"data": {"sectionAnalysis": analysis}}}}}})
    _event(run_dir, event="started", stage="summary", operation_id="summary-luna")
    _event(run_dir, stage="summary", operation_id="summary-luna", outcome="partial", response_path="response.json")

    _report(run_dir)
    markdown = (run_dir / "report.md").read_text()
    table = markdown.split("## Saved summary verification coverage")[1]
    assert "Overview units supported by verifier / total" in table
    assert next(line for line in table.splitlines() if line.startswith("| summary-luna ")).endswith(f"| {expected} |")
    assert "## Strict verification" not in markdown
    if status == "retained":
        assert "Saved draft cites" in markdown


@pytest.mark.parametrize("failure,status", [
    ("wrong citation", "unsupported"), ("wrong attribution", "contradicted"),
    ("dropped qualifier", "unsupported"), ("uncited assertion", "uncited"), ("verifier failure", "error")])
def test_strict_receipt_preserves_failures_separately_from_structural_success(run_dir: Path, failure: str, status: str) -> None:
    _qa(run_dir)
    text = f"{failure} [P1]"
    _json(run_dir / "requests/verify.json", {"body": {"text": text, "api_provider": "openai", "model": "test-verifier", "evidence": [{"id": "P1", "quote": "Exact source."}]}})
    _json(run_dir / "responses/verify.json", {"status_code": 200, "body": _strict_report(text, status)})
    _event(run_dir, event="started", stage="qa_verification", operation_id="verify", request_path="requests/verify.json")
    _event(run_dir, stage="qa_verification", operation_id="verify", outcome="completed", response_path="responses/verify.json", checks={"passed": True})

    report = _report(run_dir)

    assert report["operations"][0]["attempts"][0]["metrics"]["structured_output_valid"] is True
    metrics = report["operations"][1]["attempts"][0]["metrics"]["strict_verification"]
    assert metrics["integrity_valid"] is True
    assert metrics["all_claims_supported"] is False
    assert metrics["finding_status_counts"] == {status: 1}
    assert metrics["coverage"]["evaluated"] == int(status not in {"uncited", "error"})
    packet = json.loads((run_dir / "semantic-review.json").read_text())
    assert all(item["status"] == "pending" and all(label is None for label in item["labels"].values()) for item in packet["items"])
    assert "Strict verification" in (run_dir / "report.md").read_text()


@pytest.mark.parametrize("mutation", ["text_hash", "evidence_hash", "coverage", "range", "unsupported_claimed_supported", "missing"])
def test_strict_report_cannot_claim_support_with_stale_or_inconsistent_receipts(mutation: str) -> None:
    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics
    text = "A condition applies [P1]."
    report = _strict_report(text)
    if mutation == "text_hash":
        report["text_sha256"] = "0" * 64
    if mutation == "evidence_hash":
        report["evidence"]["P1"]["text_sha256"] = "0" * 64
    if mutation == "coverage":
        report["coverage"]["evaluated"] = 0
    if mutation == "range":
        report["findings"][0]["end"] = len(text) + 1
    if mutation == "unsupported_claimed_supported":
        report["findings"][0]["status"] = "unsupported"
    if mutation == "missing":
        report = None
    checks = verification_metrics(report, text, {"P1": "Exact source."})
    assert checks["integrity_valid"] is False
    assert checks["all_claims_supported"] is False
    assert checks["human_review"] == "pending"


@pytest.mark.parametrize("mutation", ["status_list", "coverage_bool", "empty_id", "missing_provider"])
def test_malformed_strict_receipts_fail_closed_without_crashing(mutation):
    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics
    text = "Claim [P1]"
    report = _strict_report(text)
    if mutation == "status_list":
        report["findings"][0]["status"] = []
    if mutation == "coverage_bool":
        report["coverage"]["total"] = True
    if mutation == "empty_id":
        report["findings"][0]["id"] = ""
    if mutation == "missing_provider":
        report.pop("provider")
    assert verification_metrics(report, text, {"P1": "Exact source."})["integrity_valid"] is False


def test_strict_receipt_accepts_only_punctuation_gaps_skipped_by_real_parser():
    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics
    text = "Claim [P1]."
    report = _strict_report(text)
    report["findings"][0].update(text=text[:-1], end=len(text)-1)
    assert verification_metrics(report, text, {"P1": "Exact source."})["all_claims_supported"] is True
    report["findings"][0].update(text=text[2:-1], start=2)
    assert verification_metrics(report, text, {"P1": "Exact source."})["integrity_valid"] is False


@pytest.mark.parametrize("status", ["supported", "unsupported", "error", "missing"])
def test_summary_verification_counts_all_batches_chapters_and_overview(status):
    from Helper_Scripts.Evals.political_pdf_eval_report import summary_verification_metrics
    text = "Claim [P1]"
    batch = {"status": "completed", "verification_text": text,
        "evidence": [{"id": "P1", "quote": "Exact source.", "claim": "Claim"}]}
    section = {"title": "Chapter", "status": "completed", "content": text, "batches": [batch]}
    analysis = {"sections": [section], "synthesis": text}
    if status != "missing":
        batch["claim_verification"] = _strict_report(text, status)
        section["claim_verification"] = _strict_report(text)
        analysis["synthesis_verification"] = _strict_report(text)
    metrics = summary_verification_metrics(analysis)
    assert metrics["expected_reports"] == {"batches": 1, "chapters": 1, "overview": 1}
    assert metrics["valid_reports"] == {"batches": int(status != "missing"), "chapters": int(status != "missing"), "overview": int(status != "missing")}
    assert metrics["all_claims_supported"] is (status == "supported")
    assert metrics["human_review"] == "pending"


def test_strict_receipt_cannot_substitute_a_different_valid_citation():
    import hashlib

    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics
    report = _strict_report("Claim [P1]")
    report["findings"][0]["evidence_ids"] = ["P2"]
    report["evidence"]["P2"] = {"text_sha256": hashlib.sha256(b"Other source.").hexdigest()}
    assert verification_metrics(report, "Claim [P1]", {"P1": "Exact source.", "P2": "Other source."})["integrity_valid"] is False


@pytest.mark.asyncio
async def test_offline_report_accepts_real_shared_parser_and_support_receipt():
    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics

    from tldw_Server_API.app.core.Claims_Extraction.citation_support import verify_cited_text
    text = "A condition applies [P1].\n\n* A second conditional claim [P1]."
    async def provider(*args, **kwargs):
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": "supported", "confidence": 1, "rationale": "Test source support"})}}]}
    receipt = await verify_cited_text(text, {"P1": {"text": "Exact source."}}, provider="openai", model="test", analyze_fn=provider)
    metrics = verification_metrics(receipt, text, {"P1": "Exact source."})
    assert metrics["integrity_valid"] is True
    assert metrics["coverage"]["total"] == 2
    assert metrics["all_claims_supported"] is True


def test_unavailable_whole_text_error_is_valid_uncovered_evidence_not_supported():
    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics

    from tldw_Server_API.app.core.Research_Workspace.section_verification import unavailable_verification
    text = "A claim [P1]. " * 300
    metrics = verification_metrics(unavailable_verification(text, provider="openai", model="test"), text, {})
    assert metrics["integrity_valid"] is True
    assert metrics["all_claims_supported"] is False
    assert metrics["coverage"]["uncovered"] == 1


@pytest.mark.parametrize("field", ["api_provider", "model"])
def test_offline_strict_receipt_binds_the_requested_verifier_identity(run_dir, field):
    text = "Claim [P1]"
    request = {"text": text, "api_provider": "openai", "model": "test-verifier", "evidence": [{"id": "P1", "quote": "Exact source."}]}
    request[field] = "different-requested-identity"
    _json(run_dir / "requests/verify.json", {"body": request})
    _json(run_dir / "responses/verify.json", {"status_code": 200, "body": _strict_report(text)})
    _event(run_dir, event="started", stage="qa_verification", operation_id="verify", request_path="requests/verify.json")
    _event(run_dir, stage="qa_verification", operation_id="verify", outcome="completed", response_path="responses/verify.json")
    metrics = _report(run_dir)["operations"][0]["attempts"][0]["metrics"]["strict_verification"]
    assert metrics["integrity_valid"] is False
    assert metrics["all_claims_supported"] is False


@pytest.mark.parametrize("annotation,accepted", [
    ("1, PDF p. 872; printed p. 839", True),
    ("1, PDF p. 692, printed p. 659", True),
    ("1, PDF pp. 846–847, printed pp. 813–814", True),
    ("1, unrelated-source-id", False),
    ("1, PDF p. 692, printed p. 659, P2", False),
    ("1-2, PDF p. 692, printed p. 659", False),
    ("1, PDF p. unknown; printed p. 839", False),
    ("1, PDF p. 0; printed p. 839", False),
])
def test_page_annotated_citation_keeps_location_separate_from_evidence_identity(annotation, accepted):
    import hashlib

    from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics
    text = f"A source proposal [{annotation}]"
    report = _strict_report(text)
    report["findings"][0]["evidence_ids"] = ["1"]
    report["evidence"] = {"1": {"text_sha256": hashlib.sha256(b"Exact source.").hexdigest()}}
    checks = verification_metrics(report, text, {"1": "Exact source."})
    assert checks["integrity_valid"] is accepted
    assert checks["all_claims_supported"] is accepted


def _finding_run(run_dir: Path, *, stage: str = "summary", missing: bool = False) -> None:
    """Freeze one exact claim at each summary target or in a verifier request."""
    text = "Claim [P1]"
    source = {"id": "P1", "quote": "Exact source.", "claim": "Claim", "source_id": 7,
        "media_id": 42, "navigation_version": "nav-v1", "start_char": 100, "end_char": 113,
        "page_spans": [{"page_number": 872, "printed_page_label": "839"}]}
    receipt = _strict_report(text)
    receipt["findings"][0]["confidence"] = 0.8
    receipt["evidence"]["P1"].update({key: source[key] for key in (
        "source_id", "navigation_version", "start_char", "end_char", "page_spans")})
    request = {"body": {"text": text, "api_provider": "openai", "model": "test-verifier",
        "evidence": [source]}}
    if stage == "summary":
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["config"]["models"][0]["model"] = "test-verifier"
        _json(run_dir / "manifest.json", manifest)
        batch = {"status": "completed", "verification_text": text, "evidence": [source],
            "claim_verification": receipt}
        section = {"title": "Financial Regulatory Agencies", "status": "completed", "content": text,
            "source_id": 7, "media_id": 42, "node_id": "chapter-27", "navigation_version": "nav-v1",
            "batches": [batch], "claim_verification": receipt}
        analysis = {"sections": [section], "synthesis": text, "synthesis_verification": receipt}
        if missing:
            section.pop("claim_verification")
        body = {"artifact": {"id": 9, "version_metadata": {"workspaceClient": {"data": {
            "sectionAnalysis": analysis}}}}}
    else:
        body = None if missing else receipt
    _json(run_dir / "requests/findings.json", request)
    _json(run_dir / "responses/findings.json", {"body": body})
    _event(run_dir, event="started", operation_id="findings", stage=stage,
        request_path="requests/findings.json")
    _event(run_dir, operation_id="findings", stage=stage, response_path="responses/findings.json",
        outcome="partial" if missing else "completed", checks={"passed": True})


@pytest.mark.parametrize("stage,count", [("summary", 3), ("qa_verification", 1), ("verification_regression", 1)])
def test_finding_packet_exposes_exact_claims_and_frozen_source_bindings(run_dir: Path, stage, count):
    _finding_run(run_dir, stage=stage)
    report = _report(run_dir)
    packet = json.loads((run_dir / "semantic-review.json").read_text())
    item = packet["items"][0]
    rows = item["finding_reviews"]
    assert len(rows) == count
    assert len({row["target"]["location"] for row in rows}) == count
    for row in rows:
        assert (row["finding_id"], row["text"], row["start"], row["end"]) == ("C1", "Claim [P1]", 0, 10)
        assert row["automated_finding"]["confidence"] == 0.8
        assert row["automated_finding"]["reason"] == "Recorded cited-source verdict"
        assert row["receipt"]["integrity_valid"] is True
        source = row["sources"][0]
        assert source["status"] == "resolved"
        assert source["quote"] == "Exact source."
        assert (source["start_char"], source["end_char"], source["source_id"]) == (100, 113, 7)
        assert source["page_spans"] == [{"page_number": 872, "printed_page_label": "839"}]
        assert source["attribution_context"] is None
        assert row["agent_review"] is None and row["human_review"] is None
        assert len(row["row_sha256"]) == 64
    assert packet["human_calibration"] == "not_computed"
    assert report["operations"][0]["latest_outcome"] == "completed"


@pytest.mark.parametrize("stage", ["summary", "qa_verification"])
def test_missing_finding_receipts_remain_explicit_unknowns(run_dir: Path, stage):
    _finding_run(run_dir, stage=stage, missing=True)
    report = _report(run_dir)
    item = json.loads((run_dir / "semantic-review.json").read_text())["items"][0]
    assert len(item["finding_review_gaps"]) == 1
    assert item["finding_review_gaps"][0]["status"] == "unknown"
    assert len(item["finding_reviews"]) == (2 if stage == "summary" else 0)
    assert report["operations"][0]["latest_outcome"] == "partial"


def _decision(row: dict, provenance: str) -> dict:
    return {"provenance": provenance, "reviewer": "source-reader", "decision": "unsupported",
        "rationale": "The cited passage omits the qualifier.", "reviewed_at": "2026-11-17T20:00:00Z",
        "row_sha256": row["row_sha256"]}


def test_finding_decisions_preserve_agent_and_human_authority_separately(run_dir: Path):
    _finding_run(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    rows = packet["items"][0]["finding_reviews"]
    rows[0]["agent_review"] = _decision(rows[0], "agent_source_audit")
    rows[1]["human_review"] = _decision(rows[1], "human_source_review")
    _json(path, packet)
    _report(run_dir)
    regenerated = json.loads(path.read_text())
    assert regenerated["items"][0]["finding_reviews"] == rows
    assert regenerated["human_calibration"] == "not_computed"
    assert all(value is None for value in regenerated["items"][0]["labels"].values())


@pytest.mark.parametrize("mutation", ["text", "source", "receipt", "disappeared", "duplicate", "slot", "hash", "reviewer", "timestamp"])
def test_invalid_finding_review_cannot_replace_existing_packet(run_dir: Path, mutation):
    _finding_run(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    rows = packet["items"][0]["finding_reviews"]
    rows[0]["agent_review"] = _decision(rows[0], "agent_source_audit")
    if mutation == "text":
        rows[0]["text"] = "Other claim"
    elif mutation == "source":
        rows[0]["sources"][0]["quote"] = "Other passage"
    elif mutation == "receipt":
        rows[0]["automated_finding"]["reason"] = "Other rationale"
    elif mutation == "disappeared":
        rows[0]["finding_id"] = "missing-claim"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "slot":
        rows[0]["human_review"] = rows[0]["agent_review"]
    elif mutation == "hash":
        rows[0]["agent_review"]["row_sha256"] = "0" * 64
    elif mutation == "reviewer":
        rows[0]["agent_review"]["reviewer"] = ""
    else:
        rows[0]["agent_review"]["reviewed_at"] = "yesterday"
    _json(path, packet)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="[Ff]inding|provenance"):
        _report(run_dir)
    assert path.read_bytes() == before


def test_legacy_parent_review_upgrades_without_inventing_finding_labels(run_dir: Path):
    _finding_run(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    item = packet["items"][0]
    item.pop("finding_reviews", None)
    item.pop("finding_review_gaps", None)
    item.update(status="reviewed", reviewer="legacy-reader", notes="Keep my decision")
    item["labels"]["faithfulness"] = "fail"
    _json(path, packet)
    _report(run_dir)
    updated = json.loads(path.read_text())["items"][0]
    assert (updated["status"], updated["reviewer"], updated["labels"]["faithfulness"]) == ("reviewed", "legacy-reader", "fail")
    assert all(row["human_review"] is None for row in updated["finding_reviews"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different_hash", "conflicting_coordinates"])
def test_unresolved_finding_sources_are_unknown_not_invented(run_dir: Path, mutation):
    _finding_run(run_dir, stage="qa_verification")
    path = run_dir / "requests/findings.json"
    request = json.loads(path.read_text())
    if mutation == "missing":
        request["body"]["evidence"] = []
    elif mutation == "duplicate":
        request["body"]["evidence"].append(request["body"]["evidence"][0])
    elif mutation == "different_hash":
        request["body"]["evidence"][0]["quote"] = "Other source."
    else:
        request["body"]["evidence"][0]["start_char"] = 99
    _json(path, request)
    report = _report(run_dir)
    item = json.loads((run_dir / "semantic-review.json").read_text())["items"][0]
    source = item["finding_reviews"][0]["sources"][0]
    assert source["status"] == "unknown"
    assert source["reason"]
    assert item["finding_reviews"][0]["receipt"]["integrity_valid"] is False
    assert item["finding_reviews"][0]["automated_finding"]["status"] == "supported"
    assert report["operations"][0]["latest_outcome"] == "completed"
    assert item["finding_reviews"][0]["human_review"] is None


@pytest.mark.parametrize("field", ["source_sha256", "fixture_sha256", "git_sha"])
def test_finding_decision_cannot_follow_changed_frozen_run_identity(run_dir: Path, field):
    _finding_run(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    row = packet["items"][0]["finding_reviews"][0]
    row["agent_review"] = _decision(row, "agent_source_audit")
    _json(path, packet)
    before = path.read_bytes()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest[field] = "changed-frozen-identity"
    _json(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed"):
        _report(run_dir)
    assert path.read_bytes() == before


def test_missing_response_is_an_explicit_gap_without_fabricated_findings(run_dir: Path):
    _event(run_dir, event="started", stage="qa_verification")
    report = _report(run_dir)
    packet = json.loads((run_dir / "semantic-review.json").read_text())
    assert report["operations"][0]["latest_outcome"] == "unknown"
    assert packet["items"] == []
    assert packet["finding_review_gaps"] == [{"operation_id": "qa-luna-H05", "attempt": 1,
        "stage": "qa_verification", "status": "unknown", "reason": "response_not_saved"}]


def test_duplicate_frozen_finding_identity_is_rejected_before_output(run_dir: Path):
    _finding_run(run_dir, stage="qa_verification")
    path = run_dir / "responses/findings.json"
    response = json.loads(path.read_text())
    response["body"]["findings"].append(response["body"]["findings"][0])
    _json(path, response)
    with pytest.raises(ValueError, match="Duplicate frozen finding"):
        _report(run_dir)
    assert not (run_dir / "semantic-review.json").exists()


@pytest.mark.parametrize("case", ["matching", "mismatch", "missing_hash", "digest_only"])
def test_finding_attribution_context_requires_matching_receipt_digest(run_dir: Path, case):
    import hashlib

    _finding_run(run_dir, stage="qa_verification")
    request_path, response_path = run_dir / "requests/findings.json", run_dir / "responses/findings.json"
    request, response = json.loads(request_path.read_text()), json.loads(response_path.read_text())
    if case != "digest_only":
        request["body"]["evidence"][0]["attribution_context"] = "Incorrect context" if case == "mismatch" else "Exact context"
    if case != "missing_hash":
        response["body"]["evidence"]["P1"]["attribution_context_sha256"] = hashlib.sha256(b"Exact context").hexdigest()
    _json(request_path, request)
    _json(response_path, response)
    original_finding = response["body"]["findings"][0]
    report = _report(run_dir)
    row = json.loads((run_dir / "semantic-review.json").read_text())["items"][0]["finding_reviews"][0]
    source = row["sources"][0]
    assert source["attribution_context_status"] == ("resolved" if case == "matching" else "unknown")
    assert source["attribution_context"] == request["body"]["evidence"][0].get("attribution_context")
    assert source["status"] == ("unknown" if case in {"mismatch", "missing_hash"} else "resolved")
    assert row["receipt"]["integrity_valid"] is (case in {"matching", "digest_only"})
    assert row["automated_finding"] == original_finding
    assert report["operations"][0]["latest_outcome"] == "completed"


@pytest.mark.parametrize("field", ["source_sha256", "fixture_sha256"])
def test_legacy_parent_review_rejects_changed_packet_identity_without_writing(run_dir: Path, field):
    _finding_run(run_dir)
    _report(run_dir)
    path = run_dir / "semantic-review.json"
    packet = json.loads(path.read_text())
    parent = packet["items"][0]
    parent.pop("finding_reviews")
    parent.pop("finding_review_gaps")
    parent.update(status="reviewed", reviewer="legacy-reader", notes="Retain my original decision")
    parent["labels"]["faithfulness"] = "fail"
    _json(path, packet)
    before = {name: (run_dir / name).read_bytes() for name in ("semantic-review.json", "report.json", "report.md")}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest[field] = "changed-frozen-identity"
    _json(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed"):
        _report(run_dir)
    assert before == {name: (run_dir / name).read_bytes() for name in before}
