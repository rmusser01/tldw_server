"""Offline safety and public-API contracts for the political PDF evaluation CLI."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pymupdf
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def runner_module():
    if sys.platform == "win32":
        pytest.skip("Political PDF CLI requires Unix fcntl locking and O_NOFOLLOW; native Windows is unsupported")
    return importlib.import_module("Helper_Scripts.Evals.run_political_pdf_eval")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_TEST_KEY", "offline-auth-secret")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "The commission proposes keeping the regional offices.")
    pdf = document.tobytes()
    source = tmp_path / "source.pdf"
    source.write_bytes(pdf)
    page_hash = hashlib.sha256(page.get_text().encode()).hexdigest()
    document.close()
    digest = hashlib.sha256(pdf).hexdigest()
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps({"schema_version": 1, "source_sha256": digest, "pdf_pages": 1,
        "cases": [{"id": "R01", "question": "Which offices are proposed?", "source_pages": [
            {"pdf_page": 1, "printed_page": "1", "extracted_page_sha256": page_hash}],
            "expected_source_facts": ["Do not put these expected facts into the prompt."],
            "human_review": {"status": "pending", "decision": None}}]}))
    config = {"schema_version": 1, "api": {"base_url": "http://127.0.0.1:8000", "api_key_env": "EVAL_TEST_KEY"},
        "source": {"path": str(source), "sha256": digest, "pdf_pages": 1, "media_id": 7},
        "cases": str(fixture), "workspace_id": "ws", "retrieval": {"modes": ["text"], "top_k": 3, "hybrid_alpha": 0.5},
        "models": [{"key": "test", "provider": "openai", "model": "test-model", "max_tokens": 1000}],
        "qa": {"contexts": ["reference_pages", "retrieved"], "retrieval_mode": "text"},
        "summary": {"sections": [{"source_id": "source-7", "node_id": "chapter", "navigation_version": "v1"}], "max_tokens": 1000}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_plan_is_offline_and_verifies_source_pages(runner_module, config_file):
    plan = runner_module.load_plan(config_file)
    assert plan["cases"][0]["id"] == "R01"
    assert plan["checks"]["page_fingerprints_match"] is True
    assert plan["paid_operations"]["qa"] == 2


@pytest.mark.parametrize("field,value", [("api_key", "secret"), ("headers", {"Authorization": "secret"})])
def test_config_rejects_embedded_credentials(runner_module, config_file, field, value):
    config = json.loads(config_file.read_text())
    config["api"][field] = value
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner_module.load_plan(config_file)


def test_unknown_paid_outcome_never_automatically_repeats(runner_module, config_file, tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_TEST_KEY", "super-secret")
    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs)
        raise httpx.ReadTimeout("server echoed super-secret")

    plan = runner_module.load_plan(config_file)
    directory = tmp_path / "run"
    with runner_module.Runner(plan, directory, fetch=interrupted) as run:
        first = run.request("qa-test", "qa", "POST", "/api/v1/chat/completions", body={"model": "test"})
        assert first["outcome"] == "unknown"
    with runner_module.Runner(plan, directory, fetch=interrupted, resume=True) as run:
        with pytest.raises(runner_module.UnsafeResume):
            run.request("qa-test", "qa", "POST", "/api/v1/chat/completions", body={"model": "test"})
    assert len(calls) == 1
    assert "super-secret" not in "".join(p.read_text() for p in directory.rglob("*.json*"))


def test_explicit_retry_retains_first_failure_and_resume_reuses_success(runner_module, config_file, tmp_path):
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return httpx.Response(500 if len(calls) == 1 else 200, json={"attempt": len(calls)})

    plan = runner_module.load_plan(config_file)
    directory = tmp_path / "run"
    with runner_module.Runner(plan, directory, fetch=responses) as run:
        assert run.request("op", "qa", "POST", "/chat")["outcome"] == "failed"
    with runner_module.Runner(plan, directory, fetch=responses, resume=True, retry_operations={"op"}) as run:
        assert run.request("op", "qa", "POST", "/chat")["attempt"] == 2
    with runner_module.Runner(plan, directory, fetch=responses, resume=True) as run:
        assert run.body(run.request("op", "qa", "POST", "/chat")) == {"attempt": 2}
    assert len(calls) == 2
    events = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    assert [event["outcome"] for event in events if event["event"] == "finished"] == ["failed", "completed"]


def test_started_request_and_identity_drift_block_resume(runner_module, config_file, tmp_path):
    plan = runner_module.load_plan(config_file)
    directory = tmp_path / "run"
    with runner_module.Runner(plan, directory, fetch=lambda **_: None) as run:
        run.append({"schema_version": 1, "event": "started", "operation_id": "op", "attempt": 1, "stage": "qa"})
    with runner_module.Runner(plan, directory, resume=True) as run:
        with pytest.raises(runner_module.UnsafeResume):
            run.request("op", "qa", "POST", "/chat")
    changed = dict(plan, git_sha="different")
    with pytest.raises(ValueError, match="identity"):
        with runner_module.Runner(changed, directory, resume=True):
            pass


def test_concurrent_runner_cannot_duplicate_dispatch(runner_module, config_file, tmp_path):
    plan = runner_module.load_plan(config_file)
    with runner_module.Runner(plan, tmp_path / "run"):
        with pytest.raises(runner_module.UnsafeResume):
            with runner_module.Runner(plan, tmp_path / "run", resume=True):
                pass


def test_reference_and_retrieved_qa_preserve_public_responses(runner_module, config_file, tmp_path):
    config = json.loads(config_file.read_text())
    calls = []

    def server(**kwargs):
        calls.append(kwargs)
        path = kwargs["url"].split(":8000")[1]
        if path.endswith("/file"):
            return httpx.Response(200, content=Path(config["source"]["path"]).read_bytes(), headers={"content-type": "application/pdf"})
        if path.endswith("/navigation"):
            return httpx.Response(200, json={"media_id": 7, "available": True, "navigation_version": "v1", "nodes": [{"id": "chapter"}]})
        if path == "/api/v1/rag/search":
            assert kwargs["json"]["include_media_ids"] == [7]
            assert kwargs["json"]["enable_generation"] is False
            return httpx.Response(200, json={"results": [{"content": "The regional offices.", "metadata": {"media_id": 7, "page_number": 1}}], "errors": []})
        assert path == "/api/v1/chat/completions"
        assert "expected facts" not in json.dumps(kwargs["json"])
        assert kwargs["json"]["api_provider"] == "openai"
        assert kwargs["json"]["save_to_db"] is False
        content = '{"summary":"Regional offices.","evidence":[{"claim":"Keep offices","source_span_ids":["P1"]}]}'
        return httpx.Response(200, json={"model": "returned-model", "choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}})

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        results = run.execute(["source", "retrieval", "qa"])
    qa = [row for row in results if row["stage"] == "qa"]
    assert len(qa) == 2
    assert all(row["checks"]["citation_ids_valid"] for row in qa)
    assert all(row["checks"]["usage_available"] for row in qa)
    assert len([call for call in calls if call["url"].endswith("chat/completions")]) == 2
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report
    report = write_report(tmp_path / "run")
    assert len([row for row in report["operations"] if row["stage"] == "qa"]) == 2


@pytest.mark.parametrize("content,finish_reason", [("{broken", "stop"), ('{"summary":"Partial","evidence":[]}', "length")])
def test_bad_outputs_retained_without_claiming_success(runner_module, content, finish_reason):
    checks = runner_module.completion_checks({"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}, [{"id": "P1"}])
    assert checks["structured_output_valid"] is False
    assert checks["usage_available"] is False


def test_summary_resume_polls_known_job_without_new_paid_submission(runner_module, config_file, tmp_path):
    plan = runner_module.load_plan(config_file)
    directory = tmp_path / "run"
    submissions = []
    artifact = {"id": "summary-1", "status": "failed", "content": "Retained first portion", "version_metadata": {
        "workspaceClient": {"data": {"sectionAnalysis": {"sections": [{"status": "failed", "batches": [
            {"status": "completed", "evidence": []}, {"status": "failed"}]}]}}}}}

    def server(**kwargs):
        if kwargs["method"] == "POST":
            submissions.append(kwargs)
            return httpx.Response(202, json={"job_id": 12, "artifact_id": "summary-1", "status": "queued"})
        if kwargs["url"].endswith("/outputs/12"):
            return httpx.Response(200, json={"job_id": 12, "artifact_id": "summary-1", "status": "failed", "artifact": artifact})
        return httpx.Response(200, json=[artifact])

    with runner_module.Runner(plan, directory, fetch=server) as run:
        run.request("summary:test:submit", "summary", "POST", "/api/v1/workspaces/ws/outputs", body={})
    with runner_module.Runner(plan, directory, fetch=server, resume=True) as run:
        result = run.summarize(plan["config"]["models"][0])
    assert len(submissions) == 1
    assert result["checks"]["job_status"] == "failed"
    assert result["checks"]["completed_portions"] == 1
    assert result["checks"]["failed_portions"] == 1
    assert result["checks"]["artifact_readback_matches"] is True
    assert result["checks"]["usage_available"] is False


def test_pack_hashes_and_original_source_are_verified(runner_module, tmp_path):
    files = {"workspace.json": b"{}", "notes.json": b"[]", "chatbook.zip": b"nested", "offline/sources/media-7.pdf": b"source"}
    manifest = {"format": "tldw.complete-research-pack", "schemaVersion": 1, "sourceWorkspaceId": "ws", "mediaIds": ["7"], "noteIds": [],
        "files": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for path, data in files.items()]}
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
        archive.writestr("research-pack.json", json.dumps(manifest))
    checks = runner_module.pack_checks(pack, hashlib.sha256(b"source").hexdigest())
    assert checks["verified_file_count"] == 4
    assert checks["original_sha256_matches"] is True
    with zipfile.ZipFile(pack, "a") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ValueError):
        runner_module.pack_checks(pack, hashlib.sha256(b"source").hexdigest())


def test_retrieval_page_coverage_does_not_hide_foreign_scope_or_wide_range(runner_module):
    case = {"source_pages": [{"pdf_page": 7}, {"pdf_page": 8}]}
    body = {"results": [{"content": "Text", "metadata": {"media_id": 8, "page_start": 1, "page_end": 20}}]}
    checks = runner_module.retrieval_checks(body, case, 7, 22)
    assert checks["covered_pages"] == []
    assert checks["source_scope_valid"] is False
    body["results"][0]["metadata"]["media_id"] = 7
    checks = runner_module.retrieval_checks(body, case, 7, 22)
    assert checks["covered_pages"] == [7, 8]
    assert checks["overbroad_ranges"] == [[1, 20]]


def test_failed_check_keeps_raw_response_and_error_diagnostic(runner_module, config_file, tmp_path):
    def broken_check(_):
        raise ValueError("Cannot parse returned shape")

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=lambda **_: httpx.Response(200, json={"raw": "failed output"})) as run:
        result = run.request("malformed", "qa", "POST", "/chat", checks=broken_check)
        assert result["outcome"] == "failed"
        assert run.body(result) == {"raw": "failed output"}
        assert result["checks"]["error_type"] == "ValueError"


def test_resume_rejects_unsafe_or_corrupt_journal(runner_module, config_file, tmp_path):
    plan = runner_module.load_plan(config_file)
    directory = tmp_path / "run"
    with runner_module.Runner(plan, directory):
        pass
    (directory / "attempts.jsonl").write_text(json.dumps({"schema_version": 1, "operation_id": "op", "attempt": 1, "stage": "qa", "event": "finished", "outcome": "completed", "response_path": "../private.json"}) + "\n")
    with pytest.raises(ValueError):
        with runner_module.Runner(plan, directory, resume=True):
            pass


def test_retrieval_uses_current_public_request_schema(runner_module, config_file, tmp_path):
    from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedRAGRequest

    def public_schema(**kwargs):
        parsed = UnifiedRAGRequest.model_validate(kwargs["json"])
        assert parsed.fts_level == "chunk"
        assert parsed.enable_text_late_chunking is False
        return httpx.Response(200, json={"mode": parsed.search_mode, "results": []})

    plan = runner_module.load_plan(config_file)
    with runner_module.Runner(plan, tmp_path / "run", fetch=public_schema) as run:
        result = run.retrieve(plan["cases"][0], "text")
        assert result["outcome"] == "completed"
        assert run.body(result)["mode"] == "fts"


def test_summary_can_adopt_verified_existing_job_without_post(runner_module, config_file, tmp_path):
    config = json.loads(config_file.read_text())
    config["summary"]["job_ids"] = {"test": 99}
    config_file.write_text(json.dumps(config))
    artifact = {"id": "summary-99", "artifact_type": "summary", "status": "failed", "content": "First output", "producer_metadata": {"settings": {
        "provider": "openai", "model": "test-model", "max_tokens": 1000, "sections": config["summary"]["sections"]}}}

    def server(**kwargs):
        assert kwargs["method"] == "GET"
        if kwargs["url"].endswith("/outputs/99"):
            return httpx.Response(200, json={"job_id": 99, "artifact_id": "summary-99", "status": "failed", "artifact": artifact})
        return httpx.Response(200, json=[artifact])

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        result = run.summarize(run.config["models"][0])
        assert result["checks"]["adopted_job"] is True
        assert result["checks"]["job_status"] == "failed"


def test_summary_pending_status_is_journaled_without_a_success_claim(runner_module, config_file, tmp_path, monkeypatch):
    config = json.loads(config_file.read_text())
    config["summary"]["wait_seconds"] = 60
    config_file.write_text(json.dumps(config))
    elapsed = [0]

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(runner_module.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(runner_module.time, "sleep", sleep)
    def server(**kwargs):
        if kwargs["method"] == "POST":
            return httpx.Response(202, json={"job_id": 12, "artifact_id": "summary-1", "status": "queued"})
        return httpx.Response(200, json={"job_id": 12, "artifact_id": "summary-1", "status": "processing"})
    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        result = run.summarize(run.config["models"][0])
        assert result["outcome"] == "pending"
        assert run.events[-1]["outcome"] == "pending"


@pytest.mark.parametrize("wait_seconds,expected_sleeps", [(20, [15, 5]), (4, [4])])
def test_summary_poll_cadence_is_bounded_by_deadline_and_resume_keeps_job(
    runner_module, config_file, tmp_path, monkeypatch, wait_seconds, expected_sleeps
):
    config = json.loads(config_file.read_text())
    config["summary"]["wait_seconds"] = wait_seconds
    config_file.write_text(json.dumps(config))
    elapsed, sleeps, methods = [0], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds

    def server(**kwargs):
        methods.append(kwargs["method"])
        if kwargs["method"] == "POST":
            return httpx.Response(202, json={"job_id": 12, "artifact_id": "summary-1", "status": "queued"})
        assert kwargs["url"].endswith("/outputs/12")
        return httpx.Response(200, json={"job_id": 12, "artifact_id": "summary-1", "status": "processing"})

    monkeypatch.setattr(runner_module.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(runner_module.time, "sleep", sleep)
    plan, directory = runner_module.load_plan(config_file), tmp_path / "run"
    for resume in (False, True):
        with runner_module.Runner(plan, directory, fetch=server, resume=resume) as run:
            result = run.summarize(run.config["models"][0])
            assert result["outcome"] == "pending"
    assert sleeps == expected_sleeps * 2
    assert elapsed[0] == wait_seconds * 2
    assert methods.count("POST") == 1
    assert methods.count("GET") == (len(expected_sleeps) + 1) * 2


def test_summary_exact_spans_detect_tampered_quotes_and_missing_coverage(runner_module):
    section = {"node_id": "chapter", "media_id": 7, "start_char": 10, "end_char": 16, "status": "completed", "batches": [
        {"start_char": 10, "end_char": 16, "status": "completed", "evidence": [{"start_char": 10, "end_char": 16, "quote": "Source"}]}]}
    content = {"node_id": "chapter", "media_id": 7, "resolved_start_char": 10, "resolved_end_char": 16, "content": "Source", "coverage_status": "exact"}
    assert runner_module.section_spans_valid([section], {"chapter": content}, 7)
    section["batches"][0]["evidence"][0]["quote"] = "Invented"
    assert not runner_module.section_spans_valid([section], {"chapter": content}, 7)
    section["batches"] = []
    assert not runner_module.section_spans_valid([section], {"chapter": content}, 7)


def test_output_symlink_is_not_followed(runner_module, config_file, tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged")
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "attempts.jsonl").symlink_to(outside)
    with pytest.raises(ValueError):
        with runner_module.Runner(runner_module.load_plan(config_file), directory):
            pass
    assert outside.read_text() == "unchanged"


def test_http_client_call_matches_actual_response_adapter(runner_module, config_file, tmp_path):
    from tldw_Server_API.app.core import http_client

    def adapter_contract(**kwargs):
        inspect.signature(http_client.HttpxAdapter.request).bind(None, **kwargs)
        assert kwargs["retry"].attempts == 1
        assert kwargs["allow_redirects"] is False
        return httpx.Response(200, json={"ok": True})

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=adapter_contract) as run:
        result = run.request("real-adapter-contract", "source", "GET", "/api/v1/health")
        assert result["outcome"] == "completed"


def test_complete_summary_checks_saved_selected_scope_and_exact_evidence(runner_module, config_file, tmp_path):
    section = {"source_id": "source-7", "node_id": "chapter", "navigation_version": "v1", "media_id": 7,
        "start_char": 0, "end_char": 6, "status": "completed", "batches": [{"start_char": 0, "end_char": 6, "status": "completed",
            "evidence": [{"start_char": 0, "end_char": 6, "quote": "Source"}]}]}
    artifact = {"id": "summary-1", "artifact_type": "summary", "status": "complete", "content": "Summary", "version_metadata": {
        "workspaceClient": {"data": {"sectionAnalysis": {"sections": [section]}}}}}
    def server(**kwargs):
        path = kwargs["url"]
        if kwargs["method"] == "POST":
            return httpx.Response(202, json={"job_id": 12, "artifact_id": "summary-1", "status": "queued"})
        if "/outputs/12" in path:
            return httpx.Response(200, json={"job_id": 12, "artifact_id": "summary-1", "status": "completed", "artifact": artifact})
        if "/navigation/chapter/content" in path:
            return httpx.Response(200, json={"node_id": "chapter", "navigation_version": "v1", "media_id": 7,
                "resolved_start_char": 0, "resolved_end_char": 6, "content": "Source", "coverage_status": "exact"})
        return httpx.Response(200, json=[artifact])
    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        result = run.summarize(run.config["models"][0])
        assert result["outcome"] == "completed"
        assert result["checks"]["evidence_spans_valid"] is True
        assert result["checks"]["selected_scope_valid"] is True
        assert result["checks"]["job_elapsed_seconds"] >= 0
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report
    report = write_report(tmp_path / "run")
    assert next(row for row in report["operations"] if row["stage"] == "summary")["latest_outcome"] == "completed"
    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server, resume=True,
                              retry_operations={"summary:test:submit"}) as run:
        with pytest.raises(runner_module.UnsafeResume, match="known summary job"):
            run.summarize(run.config["models"][0])


@pytest.mark.parametrize("artifact_status", ["complete", "failed"])
@pytest.mark.parametrize("source", ["Exact plain source.", "## Trade\n|Thailand<br>Taiwan|\n|Exports &amp; imports|"])
def test_summary_verifies_intrinsic_text_offsets_independently_of_synthesis_status(
    runner_module, config_file, tmp_path, source, artifact_status
):
    from tldw_Server_API.app.core.Media import navigation

    section = {"source_id": "source-7", "node_id": "chapter", "navigation_version": "v1", "media_id": 7,
        "start_char": 0, "end_char": len(source), "status": "completed", "batches": [
            {"start_char": 0, "end_char": len(source), "status": "completed", "evidence": [
                {"id": "E1.1.1", "start_char": 0, "end_char": len(source), "quote": source}]}]}
    analysis = {"sections": [section]}
    if artifact_status == "failed":
        analysis["synthesis_error_code"] = "unknown_evidence_reference"
    artifact = {"id": "summary-1", "artifact_type": "summary", "status": artifact_status,
        "version_metadata": {"workspaceClient": {"data": {"sectionAnalysis": analysis}}}}

    def server(**kwargs):
        path = kwargs["url"]
        if kwargs["method"] == "POST":
            return httpx.Response(202, json={"job_id": 12, "artifact_id": "summary-1", "status": "queued"})
        if "/outputs/12" in path:
            return httpx.Response(200, json={"job_id": 12, "artifact_id": "summary-1", "status": "completed", "artifact": artifact})
        if "/navigation/chapter/content" in path:
            # Exercise the public endpoint's actual rendering contract: plain HTML
            # display text changes both tags/entities and canonical offset lengths.
            variants, intrinsic = navigation._build_content_variants(source)
            content_format, content = navigation._resolve_requested_content(
                parse_qs(urlsplit(path).query)["format"][0], variants, intrinsic
            )
            return httpx.Response(200, json={"node_id": "chapter", "navigation_version": "v1", "media_id": 7,
                "resolved_start_char": 0, "resolved_end_char": len(source), "coverage_status": "exact",
                "content_format": content_format, "content": content})
        return httpx.Response(200, json=[artifact])

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        result = run.summarize(run.config["models"][0])
        assert result["checks"]["evidence_spans_valid"] is True
        assert result["outcome"] == ("completed" if artifact_status == "complete" else "failed")
        assert run.body(result)["artifact"] == artifact


def test_model_key_suffix_cannot_alias_another_models_summary_submission(runner_module, config_file, tmp_path):
    config = json.loads(config_file.read_text())
    config["models"].append({**config["models"][0], "key": "test-submit"})
    config_file.write_text(json.dumps(config))
    submitted = []
    def server(**kwargs):
        if kwargs["method"] == "POST":
            submitted.append(kwargs["json"])
            return httpx.Response(202, json={"job_id": len(submitted), "artifact_id": f"summary-{len(submitted)}", "status": "queued"})
        if "/outputs/" in kwargs["url"]:
            return httpx.Response(200, json={"job_id": len(submitted), "artifact_id": f"summary-{len(submitted)}", "status": "failed", "artifact": {}})
        return httpx.Response(200, json=[])
    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=server) as run:
        results = [run.summarize(model) for model in run.config["models"]]
        assert len(submitted) == 2
        assert all(result["stage"] == "summary" for result in results)
        assert len({result["operation_id"] for result in results}) == 2


def test_summary_rejects_unrepresentable_reasoning_setting_before_post(runner_module, config_file, tmp_path):
    plan = runner_module.load_plan(config_file)
    model = {**plan["config"]["models"][0], "reasoning_effort": "none"}
    calls = []
    with runner_module.Runner(plan, tmp_path / "run", fetch=lambda **kw: calls.append(kw)) as run:
        with pytest.raises(ValueError, match="reasoning"):
            run.summarize(model)
    assert not calls


def test_oversized_retrieved_context_is_rejected_before_paid_generation(runner_module, config_file, tmp_path):
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import BATCH_CHARS

    calls = []
    def server(**kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"documents": [{"id": "media_7_chunk_1", "content": "x" * (BATCH_CHARS + 1),
            "metadata": {"source": "media_db", "media_id": "7", "page_spans": [{"page_number": 1, "start_char": 0, "end_char": BATCH_CHARS + 1}]}}], "errors": []})
    plan = runner_module.load_plan(config_file)
    with runner_module.Runner(plan, tmp_path / "run", fetch=server) as run:
        with pytest.raises(ValueError, match="context.*budget"):
            run.answer(plan["cases"][0], plan["config"]["models"][0], "retrieved")
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/rag/search")


def test_whole_book_identity_is_not_relaxed_to_pass_media_scope(runner_module):
    checks = runner_module.retrieval_checks({"documents": [{"id": "1", "content": "Whole book",
        "metadata": {"source": "media_db", "source_id": "1", "source_type": "media_db", "page_map": [{"page_number": 906}]}}]},
        {"source_pages": [{"pdf_page": 906}]}, 1, 922)
    assert checks["source_scope_valid"] is False
    assert checks["covered_pages"] == []

