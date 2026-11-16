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
    with runner_module.Runner(plan, directory, fetch=lambda **_:
        None) as run:
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

    with runner_module.Runner(runner_module.load_plan(config_file), tmp_path / "run", fetch=lambda **_:
        httpx.Response(200, json={"raw": "failed output"})) as run:
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
    with runner_module.Runner(plan, tmp_path / "run", fetch=lambda **kw:
        calls.append(kw)) as run:
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


@pytest.fixture
def strict_config(config_file):
    config = json.loads(config_file.read_text())
    config["verification"] = {"required": True, "provider": "openai", "model": "test-verifier"}
    config_file.write_text(json.dumps(config))
    return config_file


def _strict_receipt(text, evidence, status="supported"):
    import re
    findings, cursor = [], 0
    for index, paragraph in enumerate(text.split("\n\n"), 1):
        start = text.index(paragraph, cursor)
        end = start + len(paragraph)
        findings.append({"id": f"C{index}", "text": paragraph, "start": start, "end": end,
            "evidence_ids": re.findall(r"\[(P\d+|E\d+\.\d+\.\d+)\]", paragraph), "status": status, "reason": "Cited-source check"})
        cursor = end
    count = len(findings)
    evaluated = count if status in {"supported", "unsupported", "contradicted"} else 0
    return {"version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "provider": "openai", "model": "test-verifier",
        "verdict": "supported" if status == "supported" else "needs_revision" if status in {"unsupported", "contradicted", "uncited"} else "unverified",
        "coverage": {"total": count, "evaluated": evaluated, "supported": count if status == "supported" else 0,
            "uncovered": count - evaluated, "complete": count > 0 and evaluated == count}, "findings": findings,
        "evidence": {row["id"]: {"text_sha256": hashlib.sha256(row["quote"].encode()).hexdigest()} for row in evidence}, "limits": {}, "context": {}}


@pytest.mark.parametrize("status", ["supported", "unsupported", "contradicted", "uncited", "error"])
def test_strict_qa_uses_canonical_owned_ranges_and_retains_real_verifier_receipts(runner_module, strict_config, tmp_path, status):
    config = json.loads(strict_config.read_text())
    canonical = "Canonical commission: retain regional offices only if funded."
    calls = []
    def server(**kwargs):
        calls.append(kwargs)
        path, body = kwargs["url"].split(":8000")[1], kwargs.get("json", {})
        if path.endswith("/file"):
            return httpx.Response(200, content=Path(config["source"]["path"]).read_bytes())
        if path.endswith("/navigation"):
            return httpx.Response(200, json={"media_id": 7, "available": True, "navigation_version": "v1", "nodes": [{"id": "chapter"}]})
        if "/versions?" in path:
            return httpx.Response(200, json=[{"media_id": 7, "version_number": 1, "content": canonical,
                "safe_metadata": {"page_map_content_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                    "page_map": [{"page_number": 1, "start_char": 0, "end_char": len(canonical)}]}}])
        if path.endswith("/sources"):
            return httpx.Response(200, json=[{"id": "source-7", "workspace_id": "ws", "media_id": 7}])
        if path == "/api/v1/rag/search":
            return httpx.Response(200, json={"results": [{"content": canonical,
                "metadata": {"media_id": 7, "page_number": 1, "start_char": 0, "end_char": len(canonical)}}], "errors": []})
        if path == "/api/v1/chat/completions":
            prompt = body["messages"][0]["content"]
            assert canonical in prompt
            assert "The commission proposes keeping" not in prompt
            assert "cite" in prompt.lower()
            content = json.dumps({"summary": "Retain offices only if funded [P1].", "evidence": [{"claim": "Retain offices only if funded", "author": "Alice", "interpretation": "A conditional proposal", "limitations": "Funding required", "source_span_ids": ["P1"]}]})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
        assert path == "/api/v1/research-workspace/citations/verify"
        assert body["workspace_id"] == "ws"
        assert body["api_provider"] == "openai" and body["model"] == "test-verifier"
        assert body["text"] == "Retain offices only if funded [P1].\n\nRetain offices only if funded Attributed to: Alice. Interpretation: A conditional proposal. Limitations: Funding required. [P1]"
        assert body["evidence"] == [{"id": "P1", "source_id": "source-7", "quote": canonical,
            "start_char": 0, "end_char": len(canonical), "navigation_version": "v1"}]
        return httpx.Response(200, json=_strict_receipt(body["text"], body["evidence"], status))

    with runner_module.Runner(runner_module.load_plan(strict_config), tmp_path / "run", fetch=server) as run:
        results = run.execute(["qa"])
    generation = [row for row in results if row["stage"] == "qa"]
    verification = [row for row in results if row["stage"] == "qa_verification"]
    assert len(generation) == len(verification) == 2
    assert all(row["outcome"] == "completed" for row in generation)
    assert all(row["checks"]["strict_verification"]["integrity_valid"] for row in verification)
    assert all(row["checks"]["passed"] is (status == "supported") for row in verification)
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report
    report = write_report(tmp_path / "run")
    assert len([op for op in report["operations"] if op["stage"] == "qa_verification"]) == 2
    assert sum(call["url"].endswith("/citations/verify") for call in calls) == 2


def test_strict_verifier_unknown_outcome_resumes_without_regenerating_answer(runner_module, strict_config, tmp_path):
    # Reuse the existing immutable ledger contract for a paid verifier operation.
    plan = runner_module.load_plan(strict_config)
    calls = []
    def timeout(**kwargs):
        calls.append(kwargs)
        raise httpx.ReadTimeout("Lost strict verifier response")
    with runner_module.Runner(plan, tmp_path / "run", fetch=timeout) as run:
        event = run.request("qa-verification:test:R01:reference_pages", "qa_verification", "POST", "/api/v1/research-workspace/citations/verify", body={"text": "Claim [P1]"})
        assert event["outcome"] == "unknown"
    with runner_module.Runner(plan, tmp_path / "run", fetch=timeout, resume=True) as run:
        with pytest.raises(runner_module.UnsafeResume):
            run.request("qa-verification:test:R01:reference_pages", "qa_verification", "POST", "/api/v1/research-workspace/citations/verify", body={"text": "Claim [P1]"})
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["supported", "unsupported", "error"])
def test_strict_summary_generation_status_is_separate_from_all_report_coverage(runner_module, strict_config, tmp_path, status):
    text, quote = "Claim [P1]", "Source"
    evidence = [{"id": "P1", "quote": quote, "claim": "Claim", "start_char": 0, "end_char": len(quote)}]
    report = {**_strict_receipt(text, evidence, status), "model": "test-model"}
    section = {"source_id": "source-7", "node_id": "chapter", "navigation_version": "v1", "media_id": 7,
        "start_char": 0, "end_char": len(quote), "status": "completed", "content": text, "claim_verification": report,
        "batches": [{"start_char": 0, "end_char": len(quote), "status": "completed", "evidence": evidence,
            "verification_text": text, "claim_verification": report}]}
    analysis = {"sections": [section], "synthesis": text, "synthesis_verification": report}
    artifact = {"id": "summary", "artifact_type": "summary", "status": "complete", "content": text,
        "version_metadata": {"workspaceClient": {"data": {"sectionAnalysis": analysis}}}}
    def server(**kwargs):
        path = kwargs["url"]
        if kwargs["method"] == "POST":
            return httpx.Response(202, json={"job_id": 12})
        if "/outputs/12" in path:
            return httpx.Response(200, json={"status": "completed", "artifact": artifact})
        if "/navigation/chapter/content" in path:
            return httpx.Response(200, json={"node_id": "chapter", "navigation_version": "v1", "media_id": 7,
                "resolved_start_char": 0, "resolved_end_char": len(quote), "content": quote, "coverage_status": "exact"})
        return httpx.Response(200, json=[artifact])
    with runner_module.Runner(runner_module.load_plan(strict_config), tmp_path / "run", fetch=server) as run:
        result = run.summarize(run.config["models"][0])
    assert result["checks"]["structural_passed"] is True
    assert result["checks"]["job_status"] == "completed"
    assert result["checks"]["passed"] is (status == "supported")
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report
    metrics = next(row for row in write_report(tmp_path / "run")["operations"] if row["stage"] == "summary")["attempts"][0]["metrics"]["strict_verification"]
    assert metrics["expected_reports"] == {"batches": 1, "chapters": 1, "overview": 1}
    assert metrics["all_claims_supported"] is (status == "supported")


@pytest.mark.parametrize("status", ["unsupported", "supported", "error"])
def test_fixed_source_audited_regression_uses_actual_verifier_without_generation(runner_module, strict_config, tmp_path, status):
    config = json.loads(strict_config.read_text())
    canonical = "The source requires purchasing at least 64 new interceptors."
    case = {"id": "number-regression", "text": "Maintain 64 existing interceptors [P1]",
        "evidence": [{"id": "P1", "quote": canonical}], "expected_statuses": ["unsupported", "contradicted"],
        "label_provenance": "agent_source_audit", "human_review": {"status": "pending", "decision": None}}
    fixture = tmp_path / "regressions.json"
    fixture.write_text(json.dumps({"schema_version": 1, "source_sha256": config["source"]["sha256"], "cases": [case]}))
    config["verification_regressions"] = {"path": str(fixture), "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
    strict_config.write_text(json.dumps(config))
    posts = []
    def server(**kwargs):
        path = kwargs["url"]
        if path.endswith("/sources"):
            return httpx.Response(200, json=[{"id": "source-7", "workspace_id": "ws", "media_id": 7}])
        if path.endswith("/navigation"):
            return httpx.Response(200, json={"media_id": 7, "navigation_version": "v1"})
        if "/versions?" in path:
            return httpx.Response(200, json=[{"media_id": 7, "content": canonical, "safe_metadata": {
                "page_map_content_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                "page_map": [{"page_number": 1, "start_char": 0, "end_char": len(canonical)}]}}])
        assert path.endswith("/citations/verify") and kwargs["method"] == "POST"
        body = kwargs["json"]
        assert body["text"] == case["text"]
        assert body["evidence"][0]["quote"] == canonical
        assert body["evidence"][0]["source_id"] == "source-7"
        posts.append(body)
        return httpx.Response(200, json=_strict_receipt(body["text"], body["evidence"], status))
    plan = runner_module.load_plan(strict_config)
    assert plan["verification_regressions_sha256"] == config["verification_regressions"]["sha256"]
    with runner_module.Runner(plan, tmp_path / "run", fetch=server) as run:
        result = run.verification_regression(case, run.config["models"][0])
    assert result["checks"]["passed"] is (status == "unsupported")
    assert result["checks"]["strict_verification"]["all_claims_supported"] is (status == "supported")
    with runner_module.Runner(plan, tmp_path / "run", fetch=server, resume=True) as run:
        assert run.verification_regression(case, run.config["models"][0]) == result
    assert len(posts) == 1
    from Helper_Scripts.Evals.political_pdf_eval_report import write_report
    observed = next(row for row in write_report(tmp_path / "run")["operations"] if row["stage"] == "verification_regression")
    assert observed["attempts"][0]["metrics"]["regression_expectation_met"] is (status == "unsupported")
    fixture.write_text(fixture.read_text() + " ")
    with pytest.raises(ValueError, match="fingerprint"):
        runner_module.load_plan(strict_config)


def test_strict_unknown_check_reuses_generated_answer_before_stopping_resume(runner_module, strict_config, tmp_path):
    plan, posts = runner_module.load_plan(strict_config), []
    raw = {"choices": [{"message": {"content": json.dumps({"summary": "Claim [P1]", "evidence": [{"claim": "Claim", "source_span_ids": ["P1"]}]})}}]}
    def provider(**kwargs):
        posts.append(kwargs["url"])
        if kwargs["url"].endswith("/chat/completions"):
            return httpx.Response(200, json=raw)
        raise httpx.ReadTimeout("Lost verifier response")
    def answer(run):
        return run.request("qa:test:R01:reference_pages", "qa", "POST", "/api/v1/chat/completions", body={},
            metadata={"model_key": "test", "case_id": "R01", "context": "reference_pages"},
            context_evidence={"passages": [{"id": "P1", "text": "Exact source.", "source_id": "source-7", "start_char": 0, "end_char": 13, "navigation_version": "v1"}]})
    with runner_module.Runner(plan, tmp_path / "run", fetch=provider) as run:
        generated = answer(run)
        assert run.verify_answer(generated, run.config["models"][0])["outcome"] == "unknown"
    with runner_module.Runner(plan, tmp_path / "run", fetch=provider, resume=True) as run:
        assert answer(run) == generated
        with pytest.raises(runner_module.UnsafeResume):
            run.verify_answer(generated, run.config["models"][0])
    assert len(posts) == 2


@pytest.mark.parametrize("problem", ["wrong_media", "stale_hash", "ambiguous_enrollment", "unmatched_retrieval"])
def test_canonical_source_mismatch_stops_before_any_paid_request(runner_module, strict_config, tmp_path, problem):
    canonical = "Canonical source."
    def server(**kwargs):
        path = kwargs["url"]
        if path.endswith("/sources"):
            row = {"id": "source-7", "workspace_id": "ws", "media_id": 7}
            return httpx.Response(200, json=[row, {**row, "id": "other"}] if problem == "ambiguous_enrollment" else [row])
        if path.endswith("/navigation"):
            return httpx.Response(200, json={"media_id": 7, "navigation_version": "v1"})
        if "/versions?" in path:
            return httpx.Response(200, json=[{"media_id": 8 if problem == "wrong_media" else 7, "content": canonical, "safe_metadata": {
                "page_map_content_hash": "stale" if problem == "stale_hash" else hashlib.sha256(canonical.encode()).hexdigest(),
                "page_map": [{"page_number": 1, "start_char": 0, "end_char": len(canonical)}]}}])
        assert path.endswith("/rag/search")
        return httpx.Response(200, json={"results": [{"content": "Not canonical", "metadata": {"media_id": 7, "page_number": 1}}]})
    with runner_module.Runner(runner_module.load_plan(strict_config), tmp_path / "run", fetch=server) as run:
        with pytest.raises(ValueError):
            run.answer(run.plan["cases"][0], run.config["models"][0], "retrieved")
        assert not any(row["stage"] in {"qa", "qa_verification"} for row in run.events)
