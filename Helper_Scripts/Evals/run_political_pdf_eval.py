#!/usr/bin/env python3
"""Explicit, resumable political-PDF evaluations through the public application APIs.

Uses an existing owned Media item/workspace; does not upload, accept reviews or retry
paid work automatically. See Docs/Design/POLITICAL_PDF_MODEL_EVALS_2026_11_15.md.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess  # nosec B404
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("source", "retrieval", "qa", "summary", "pack", "verification_regressions")
SECRET_FIELD = re.compile(r"(?:api_?key|authorization|password|secret|credential|access_token|refresh_token|jwt)", re.I)


class UnsafeResume(ValueError):
    """An uncertain request or concurrent run needs explicit operator resolution."""


def digest(data: bytes) -> str:
    """Return the evidence fingerprint."""
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    """Use stable JSON for identity and evidence files."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode()


def fields(value: dict, allowed: str) -> None:
    """Reject unknown configuration fields, including credential injection."""
    if not isinstance(value, dict) or set(value) - set(allowed.split()):
        raise ValueError("Unsupported configuration fields; credentials belong in environment variables")


def load_plan(config_path: Path) -> dict:
    """Validate the immutable local inputs without contacting the server."""
    import pymupdf

    config = json.loads(config_path.read_text())
    fields(config, "schema_version api source cases case_ids workspace_id retrieval models qa summary pack verification verification_regressions")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration version")
    fields(config["api"], "base_url api_key_env jwt_env timeout_seconds")
    origin = urlsplit(config["api"]["base_url"])
    if origin.scheme not in {"http", "https"} or origin.hostname not in {"localhost", "127.0.0.1", "::1"} or origin.username or origin.password or origin.query or origin.fragment or origin.path not in {"", "/"}:
        raise ValueError("The evaluation API must be a credential-free loopback origin")
    for key in ("api_key_env", "jwt_env"):
        if key in config["api"] and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api"][key]):
            raise ValueError("Authentication configuration must name an environment variable")
    verification = config.get("verification", {})
    fields(verification, "required provider model")
    if "required" in verification and type(verification["required"]) is not bool:
        raise ValueError("Verification required must be boolean")
    if any(key in verification for key in ("provider", "model")) and not all(isinstance(verification.get(key), str) and verification[key].strip() for key in ("provider", "model")):
        raise ValueError("Verification requires an exact provider and model pair")
    fields(config["source"], "path sha256 pdf_pages media_id")
    source = config["source"]
    if type(source["media_id"]) is not int or source["media_id"] < 1:
        raise ValueError("An existing owned Media ID is required; ingestion is reused")
    for key in ("path",):
        source[key] = str(Path(source[key]).expanduser().resolve())
    config["cases"] = str(Path(config["cases"]).expanduser().resolve())
    source_bytes = Path(source["path"]).read_bytes()
    fixture_bytes = Path(config["cases"]).read_bytes()
    fixture = json.loads(fixture_bytes)
    if digest(source_bytes) != source["sha256"] or fixture["source_sha256"] != source["sha256"]:
        raise ValueError("Source fingerprint mismatch")
    cases = fixture["cases"]
    selected = config.get("case_ids")
    if selected is not None:
        if not selected or set(selected) - {row["id"] for row in cases}:
            raise ValueError("Unknown or empty selected case IDs")
        cases = [row for row in cases if row["id"] in selected]
    if not cases or len({row["id"] for row in cases}) != len(cases) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", row["id"]) for row in cases):
        raise ValueError("Cases must have unique IDs")
    with pymupdf.open(stream=source_bytes, filetype="pdf") as pdf:
        if len(pdf) != source["pdf_pages"] or len(pdf) != fixture["pdf_pages"]:
            raise ValueError("Physical PDF page count mismatch")
        for case in cases:
            if not case.get("question") or not case.get("source_pages"):
                raise ValueError("Cases require a question and source pages")
            for page in case["source_pages"]:
                number = page["pdf_page"]
                if type(number) is not int or not 1 <= number <= len(pdf) or digest(pdf[number - 1].get_text().encode()) != page["extracted_page_sha256"]:
                    raise ValueError("Frozen reference-page fingerprint mismatch")
    fields(config["retrieval"], "modes top_k hybrid_alpha")
    retrieval = config["retrieval"]
    if not retrieval["modes"] or set(retrieval["modes"]) - {"text", "vector", "hybrid"} or not 1 <= retrieval["top_k"] <= 100 or not 0 <= retrieval["hybrid_alpha"] <= 1:
        raise ValueError("Invalid retrieval configuration")
    fields(config["qa"], "contexts retrieval_mode")
    if set(config["qa"]["contexts"]) - {"reference_pages", "retrieved"}:
        raise ValueError("Unknown QA context mode")
    if config["qa"].get("retrieval_mode", retrieval["modes"][0]) not in retrieval["modes"]:
        raise ValueError("QA retrieval mode must be frozen in the retrieval matrix")
    keys = set()
    for model in config["models"]:
        fields(model, "key provider model max_tokens temperature top_p reasoning_effort")
        if not all(isinstance(model.get(key), str) and model[key].strip() for key in ("key", "provider", "model")) or not re.fullmatch(r"[A-Za-z0-9_.-]+", model["key"]) or model["key"] in keys or not 1 <= model["max_tokens"] <= 32768:
            raise ValueError("Models require unique keys, exact provider/model IDs and bounded token budgets")
        keys.add(model["key"])
    fields(config.get("summary", {}), "sections artifact_type max_tokens temperature top_p summary_instruction job_ids resume_artifact_ids wait_seconds")
    wait = config.get("summary", {}).get("wait_seconds", 3600)
    if type(wait) is not int or not 1 <= wait <= 86400:
        raise ValueError("Summary wait_seconds must be an integer between 1 and 86400")
    for key, job_id in config.get("summary", {}).get("job_ids", {}).items():
        if key not in keys or type(job_id) is not int or job_id < 1:
            raise ValueError("Adopted summary jobs require a configured model key and positive job ID")
    resumes = config.get("summary", {}).get("resume_artifact_ids", {})
    if (not isinstance(resumes, dict) or any(key not in keys or not isinstance(value, str) or not 0 < len(value) <= 128
                                            for key, value in resumes.items())
            or set(resumes) & config.get("summary", {}).get("job_ids", {}).keys()
            or len(resumes) != len(set(resumes.values()))):
        raise ValueError("Each resumed artifact requires a distinct model entry without an adopted job ID")
    for section in config.get("summary", {}).get("sections", []):
        fields(section, "source_id node_id navigation_version")
        if not all(isinstance(section.get(key), str) and section[key] for key in ("source_id", "node_id", "navigation_version")):
            raise ValueError("Summary sections require exact source/node/navigation identities")
    if config.get("pack"):
        fields(config["pack"], "path sha256")
        config["pack"]["path"] = str(Path(config["pack"]["path"]).expanduser().resolve())
        if digest(Path(config["pack"]["path"]).read_bytes()) != config["pack"]["sha256"]:
            raise ValueError("Pack fingerprint mismatch")
    regression_cases, regression_hash = [], None
    if config.get("verification_regressions"):
        regression_cases, regression_hash = load_verification_regressions(config)
    # Fixed read-only command; no config or source content enters its arguments.
    revision_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)  # nosec B603 B607
    revision = revision_result.stdout.strip()
    return {"schema_version": 1, "config": config, "config_sha256": digest(encoded(config)),
            "source_sha256": source["sha256"], "fixture_sha256": digest(fixture_bytes), "git_sha": revision, "cases": cases, "verification_regressions": regression_cases, "verification_regressions_sha256": regression_hash,
            "checks": {"pdf_pages_match": True, "page_fingerprints_match": True, "ingestion": "reused"},
            "paid_operations": {"verification_regressions": len(regression_cases) * len(config["models"]), "qa_verification": len(cases) * len(config["models"]) * len(config["qa"]["contexts"]) if verification.get("required") else 0, "qa": len(cases) * len(config["models"]) * len(config["qa"]["contexts"]),
                                "summary_jobs_to_submit_if_selected": len(config["models"]) - len(config.get("summary", {}).get("job_ids", {})),
                                "summary_jobs_to_adopt_if_selected": len(config.get("summary", {}).get("job_ids", {})), "summary_provider_calls": "unavailable until execution"}}



def load_verification_regressions(config: dict) -> tuple[list[dict], str]:
    """Freeze explicit source-audited expectations independently of generated answers."""
    spec = config["verification_regressions"]
    fields(spec, "path sha256")
    spec["path"] = str(Path(spec["path"]).expanduser().resolve())
    raw = Path(spec["path"]).read_bytes()
    if digest(raw) != spec["sha256"]:
        raise ValueError("Verification regression fingerprint mismatch")
    fixture = json.loads(raw)
    if fixture.get("schema_version") != 1 or fixture.get("source_sha256") != config["source"]["sha256"]:
        raise ValueError("Verification regression source fingerprint mismatch")
    cases, identifiers = fixture.get("cases"), set()
    if not isinstance(cases, list) or not cases or len(cases) > 100:
        raise ValueError("Verification regressions require 1 to 100 frozen cases")
    for case in cases:
        fields(case, "id text evidence expected_statuses label_provenance human_review reason")
        if not isinstance(case.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]) or case["id"] in identifiers:
            raise ValueError("Verification regression IDs must be unique")
        identifiers.add(case["id"])
        if not isinstance(case.get("text"), str) or not 0 < len(case["text"]) <= 100000 or not isinstance(case.get("evidence"), list):
            raise ValueError("Verification regressions require text and evidence")
        evidence_ids = set()
        for row in case["evidence"]:
            fields(row, "id quote")
            if not isinstance(row.get("id"), str) or not re.fullmatch(r"P[1-9][0-9]*", row["id"]) or row["id"] in evidence_ids or not isinstance(row.get("quote"), str) or not row["quote"]:
                raise ValueError("Verification regressions require distinct passage IDs and exact quotes")
            evidence_ids.add(row["id"])
        if (not isinstance(case.get("expected_statuses"), list) or not case["expected_statuses"]
            or any(value not in ("supported", "unsupported", "contradicted", "uncited", "not_checked", "error") for value in case["expected_statuses"])
            or case.get("label_provenance") != "agent_source_audit"
            or case.get("human_review") != {"status": "pending", "decision": None}):
            raise ValueError("Regression expectations must retain their source-audited, human-pending provenance")
    return cases, digest(raw)

def result_documents(body: dict) -> list[dict]:
    """Read the public RAG result list without dropping raw response evidence."""
    return body.get("results", body.get("documents", [])) if isinstance(body, dict) else []


def retrieval_checks(body: dict, case: dict, media_id: int, page_count: int) -> dict:
    """Measure expected physical-page coverage, preserving scope and range failures."""
    expected = {page["pdf_page"] for page in case["source_pages"]}
    covered, ranges, overbroad = set(), [], []
    scoped = True
    for doc in result_documents(body):
        metadata = {**doc, **(doc.get("metadata") or {})}
        same_source = str(metadata.get("media_id", "")) == str(media_id)
        scoped &= same_source
        spans = metadata.get("page_spans") or [metadata]
        for span in spans:
            start = span.get("page_number", span.get("page_start"))
            end = span.get("page_end", start)
            if type(start) is not int or type(end) is not int or not 1 <= start <= end <= page_count:
                continue
            ranges.append([start, end])
            pages = set(range(start, end + 1))
            if same_source:
                covered.update(expected & pages)
            if pages - expected:
                overbroad.append([start, end])
    errors = body.get("errors", []) if isinstance(body, dict) else ["invalid_response"]
    return {"expected_pages": sorted(expected), "covered_pages": sorted(covered), "all_expected_pages_covered": covered == expected,
            "returned_page_ranges": ranges, "source_scope_valid": scoped, "pipeline_errors": errors, "overbroad_ranges": overbroad}


def completion_checks(body: dict, passages: list[dict]) -> dict:
    """Check the production extraction shape and IDs; never grade claim faithfulness."""
    choices = body.get("choices") or [] if isinstance(body, dict) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    content = (choice.get("message") or {}).get("content")
    parsed, valid, ids_valid = None, False, False
    try:
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content)) if isinstance(content, str) else None
        valid = isinstance(parsed, dict) and isinstance(parsed.get("summary"), str) and bool(parsed["summary"].strip()) and isinstance(parsed.get("evidence"), list) and bool(parsed["evidence"])
        allowed = {row["id"] for row in passages}
        if valid:
            ids_valid = all(isinstance(row, dict) and isinstance(row.get("claim"), str) and bool(row["claim"].strip()) and isinstance(row.get("source_span_ids"), list) and 1 <= len(row["source_span_ids"]) <= len(passages) and all(isinstance(value, str) and value in allowed for value in row["source_span_ids"]) for row in parsed["evidence"])
            valid = bool(ids_valid)
    except (ValueError, TypeError):
        valid = False
    finish = choice.get("finish_reason")
    return {"structured_output_valid": bool(valid), "citation_ids_valid": bool(ids_valid), "output_nonempty": isinstance(content, str) and bool(content.strip()),
            "finish_reason": finish, "completion_not_truncated": finish == "stop", "usage_available": isinstance(body, dict) and isinstance(body.get("usage"), dict) and bool(body["usage"]),
            "semantic_review": "pending", "passed": bool(valid) and finish == "stop"}


def pack_checks(path: Path, source_sha256: str) -> dict:
    """Verify bounded archive inventory/hashes; this is not a browser restore certification."""
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Pack exceeds bounded inspection size")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [row.filename for row in entries if not row.is_dir()]
        if len(entries) > 10000 or len(names) != len(set(names)) or sum(row.file_size for row in entries) > 512 * 1024 * 1024:
            raise ValueError("Invalid or oversized archive inventory")
        if any(row.file_size > 128 * 1024 * 1024 or row.flag_bits & 1 or PurePosixPath(row.filename).is_absolute() or ".." in PurePosixPath(row.filename).parts or "\\" in row.filename for row in entries):
            raise ValueError("Unsafe archive entry")
        manifest = json.loads(archive.read("research-pack.json"))
        if manifest.get("format") != "tldw.complete-research-pack" or manifest.get("schemaVersion") != 1:
            raise ValueError("Unsupported research pack manifest")
        paths, original = set(), False
        for item in manifest["files"]:
            name = item["path"]
            if name in paths or name not in names:
                raise ValueError("Invalid research pack inventory")
            paths.add(name)
            data = archive.read(name)
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError("Research pack integrity mismatch")
            original |= name.startswith("offline/sources/") and name.endswith(".pdf") and digest(data) == source_sha256
        if paths != set(names) - {"research-pack.json"} or not {"workspace.json", "notes.json", "chatbook.zip"} <= paths:
            raise ValueError("Missing or unlisted research pack entries")
    return {"manifest_hashes_valid": True, "verified_file_count": len(paths), "original_sha256_matches": original,
            "browser_restore_verified": False, "passed": original}


def section_spans_valid(sections: list[dict], contents: dict, media_id: int) -> bool:
    """Verify complete selected ranges and exact materialized quotes against navigation content."""
    if not sections or {row.get("node_id") for row in sections} != set(contents) or len(sections) != len(contents):
        return False
    for section in sections:
        content = contents[section["node_id"]]
        start, end = content.get("resolved_start_char"), content.get("resolved_end_char")
        if content.get("coverage_status") != "exact" or content.get("media_id") != media_id or content.get("node_id") != section["node_id"] or content.get("navigation_version") != section.get("navigation_version") or section.get("media_id") != media_id or section.get("status") != "completed" or type(start) is not int or type(end) is not int or section.get("start_char") != start or section.get("end_char") != end or len(content.get("content", "")) != end - start:
            return False
        cursor = start
        for batch in section.get("batches", []):
            if batch.get("status") != "completed" or batch.get("start_char") != cursor or not cursor < batch.get("end_char", cursor) <= end or not batch.get("evidence"):
                return False
            for item in batch["evidence"]:
                left, right = item.get("start_char"), item.get("end_char")
                if type(left) is not int or type(right) is not int or not cursor <= left < right <= batch["end_char"] or item.get("quote") != content["content"][left - start:right - start]:
                    return False
            cursor = batch["end_char"]
        if cursor != end:
            return False
    return True


class Runner:
    """Single-writer evidence journal; completed operations are immutable and reused."""

    def __init__(self, plan: dict, directory: Path, *, fetch: Callable | None = None, resume: bool = False, retry_operations: set[str] | None = None):
        self.plan, self.config, self.directory = plan, plan["config"], directory
        self.fetch, self.resume, self.retry_operations = fetch, resume, set(retry_operations or ())
        self.events, self.lock = [], None
        self.secrets = [value for key, value in os.environ.items() if value and SECRET_FIELD.search(key)]
        self.headers = {}
        for setting, header in (("api_key_env", "X-API-KEY"), ("jwt_env", "Authorization")):
            name = self.config["api"].get(setting)
            if name:
                value = os.environ.get(name, "")
                if not value and fetch is None:
                    raise ValueError(f"Missing authentication environment variable: {name}")
                if value:
                    self.secrets.append(value)
                    self.headers[header] = f"Bearer {value}" if setting == "jwt_env" else value

    def redact(self, value: Any) -> Any:
        """Redact credentials before writing even successful server responses."""
        if isinstance(value, dict):
            return {key: "[REDACTED]" if SECRET_FIELD.search(key) and not key.endswith("_env") else self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            for secret in sorted(set(self.secrets), key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
        return value

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in (".lock", "manifest.json", "attempts.jsonl", "evidence"):
            self.checked_path(name)
        self.lock = os.fdopen(os.open(self.checked_path(".lock"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.directory / "manifest.json"
            if manifest.exists():
                if not self.resume or json.loads(manifest.read_text()) != self.plan:
                    raise ValueError("Existing run identity differs; use the original config/revision and resume")
                journal = self.directory / "attempts.jsonl"
                self.events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
                pairs = {}
                for event in self.events:
                    if event.get("schema_version") != 1 or type(event.get("attempt")) is not int or event["attempt"] < 1:
                        raise ValueError("Malformed run journal")
                    pair = (event["operation_id"], event["attempt"])
                    if event.get("event") == "started" and pair not in pairs:
                        pairs[pair] = "started"
                    elif event.get("event") == "finished" and pairs.get(pair) == "started":
                        pairs[pair] = "finished"
                    else:
                        raise ValueError("Malformed run journal event sequence")
                    for key in ("request_path", "response_path", "artifact_path"):
                        if key in event:
                            target = (self.directory / event[key]).resolve()
                            if not target.is_relative_to(self.directory.resolve()) or not target.is_file():
                                raise ValueError("Invalid saved evidence path")
            else:
                if self.resume:
                    raise ValueError("Cannot resume a missing manifest")
                self.save("manifest.json", self.plan)
        except BlockingIOError as error:
            self.lock.close()
            raise UnsafeResume("Another runner holds this run directory") from error
        except BaseException:
            self.lock.close()
            raise
        return self

    def __exit__(self, *_):
        self.lock.close()

    def checked_path(self, name: str) -> Path:
        """Refuse symlink/hardlink outputs before they can overwrite or leak evidence."""
        path = self.directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("Output path escaped run directory")
        for entry in (self.directory, path, *path.parents):
            if entry == self.directory.parent:
                break
            if entry.is_symlink() or (entry.is_file() and entry.stat().st_nlink > 1):
                raise ValueError("Linked output paths are not permitted")
        return path

    def save(self, name: str, value: Any, *, binary: bool = False) -> str:
        """Write an immutable file before its journal reference is committed."""
        path = self.checked_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(value if binary else encoded(self.redact(value)))
            stream.flush()
            os.fsync(stream.fileno())
        return name

    def append(self, event: dict) -> None:
        """Durably append before dispatch, including unknown outcomes after interruption."""
        event = self.redact(event)
        with os.fdopen(os.open(self.checked_path("attempts.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600), "ab") as stream:
            stream.write(encoded(event).replace(b"\n", b" ") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def body(self, event: dict) -> Any:
        """Read a recorded response without making a request."""
        path = (self.directory / event["response_path"]).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Evidence path escaped the run directory")
        return json.loads(path.read_text())["body"]

    def request(self, operation_id: str, stage: str, method: str, path: str, *, body: Any = None, metadata: dict | None = None,
                context_evidence: dict | None = None, checks: Callable | None = None, binary: bool = False, local: Callable | None = None) -> dict:
        """Dispatch once, retaining responses before checks; uncertain work requires explicit retry."""
        previous = [row for row in self.events if row["operation_id"] == operation_id]
        if previous and operation_id not in self.retry_operations:
            last = previous[-1]
            if last["event"] == "started" or last.get("outcome") == "unknown":
                raise UnsafeResume(f"Unknown outcome for {operation_id}; reconcile it before explicit --retry-operation")
            return last
        self.retry_operations.discard(operation_id)
        attempt = max((row["attempt"] for row in previous), default=0) + 1
        stem = f"evidence/{digest(operation_id.encode())[:20]}-{attempt}"
        request_path = self.save(stem + ".request.json", {"method": method, "path": path, "body": body, "context_evidence": context_evidence})
        event = {"schema_version": 1, "operation_id": operation_id, "attempt": attempt, "stage": stage, **(metadata or {}),
                 "utc_time": datetime.now(timezone.utc).isoformat(), "request_path": request_path}
        self.append({**event, "event": "started"})
        started = time.monotonic()
        status, result_checks, artifact_path, response_path = None, {}, None, None
        try:
            if local:
                payload, status = local(), 200
            else:
                from tldw_Server_API.app.core import http_client
                fetch = self.fetch or http_client.fetch
                response = fetch(method=method, url=self.config["api"]["base_url"].rstrip("/") + path, headers=self.headers,
                                 timeout=self.config["api"].get("timeout_seconds", 180), retry=http_client.RetryPolicy(attempts=1), allow_redirects=False,
                                 **({"json": body} if body is not None else {}))
                status = response.status_code
                if binary and status == 200:
                    artifact_path = self.save(stem + ".pdf", response.content, binary=True)
                    payload = {"sha256": digest(response.content), "bytes": len(response.content)}
                else:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = response.text
            response_path = self.save(stem + ".response.json", {"status_code": status, "body": payload})
            outcome = "completed" if 200 <= status < 300 else "failed"
            if outcome == "completed" and checks:
                result_checks = checks(payload)
                if result_checks.get("passed") is False:
                    outcome = "failed"
                elif result_checks.get("pending"):
                    outcome = "pending"
        except Exception as error:  # noqa: BLE001 -- every failed dispatch must retain a durable outcome
            outcome = "unknown" if method == "POST" and status is None else "failed"
            result_checks = {"error_type": type(error).__name__, "error": self.redact(str(error))}
            if response_path is None:
                response_path = self.save(stem + ".error.json", {"status_code": status, "body": result_checks})
        finished = {**event, "event": "finished", "outcome": outcome, "elapsed_seconds": time.monotonic() - started,
                    "http_status": status, "response_path": response_path, "checks": result_checks}
        if artifact_path:
            finished["artifact_path"] = artifact_path
        if "payload" in locals() and isinstance(payload, dict):
            for key in ("job_id", "artifact_id"):
                if payload.get(key) is not None:
                    finished[key] = payload[key]
        self.append(finished)
        return finished

    def source(self) -> list[dict]:
        """Verify retained original and frozen outline identity for reused ingestion."""
        source = self.config["source"]
        original = self.request("source-original", "source", "GET", f"/api/v1/media/{source['media_id']}/file", binary=True,
            checks=lambda body: {**self.plan["checks"], "original_sha256_matches": body["sha256"] == source["sha256"], "passed": body["sha256"] == source["sha256"]})
        def navigation(body):
            nodes = {node["id"] for node in body.get("nodes", [])}
            valid = body.get("media_id") == source["media_id"] and bool(body.get("navigation_version")) and bool(body.get("available"))
            valid &= all(section["node_id"] in nodes and section["navigation_version"] == body.get("navigation_version") for section in self.config.get("summary", {}).get("sections", []))
            return {"navigation_valid": valid, "navigation_version": body.get("navigation_version"), "passed": valid}
        nav = self.request("source-navigation", "source", "GET", f"/api/v1/media/{source['media_id']}/navigation", checks=navigation)
        return [original, nav]

    def retrieve(self, case: dict, mode: str) -> dict:
        """Freeze source-scoped retrieval with model-driven enhancements disabled."""
        body = {"query": case["question"], "sources": ["media_db"], "include_media_ids": [self.config["source"]["media_id"]],
                "search_mode": "fts" if mode == "text" else mode, "top_k": self.config["retrieval"]["top_k"], "hybrid_alpha": self.config["retrieval"]["hybrid_alpha"],
                "fts_level": "chunk", "enable_text_late_chunking": False,
                "enable_generation": False, "enable_reranking": False, "enable_cache": False,
                **dict.fromkeys(("adaptive_cache", "adaptive_hybrid_weights", "adaptive_rerun_include_generation",
                    "adaptive_rerun_on_low_confidence", "enable_dynamic_granularity", "enable_document_grading",
                    "enable_evidence_accumulation", "enable_gap_analysis", "enable_hyde", "enable_intent_routing",
                    "enable_post_verification", "enable_prf", "enable_query_decomposition", "enable_query_rewriting_loop",
                    "enable_web_fallback", "expand_query", "spell_check", "enable_pre_retrieval_clarification"), False)}
        return self.request(f"retrieval:{case['id']}:{mode}", "retrieval", "POST", "/api/v1/rag/search", body=body,
            metadata={"case_id": case["id"], "mode": mode}, checks=lambda payload: retrieval_checks(payload, case, self.config["source"]["media_id"], self.config["source"]["pdf_pages"]))

    def canonical_source(self) -> dict:
        """Freeze one owned canonical version and its physical-page and enrollment receipts."""
        from tldw_Server_API.app.core.Media.page_provenance import page_map_for_content

        workspace, media_id = self.config.get("workspace_id"), self.config["source"]["media_id"]
        if not workspace:
            raise ValueError("Strict QA requires an existing workspace with the source enrolled")
        def read(operation, path):
            event = self.request(operation, "qa_source", "GET", path)
            if event["outcome"] != "completed":
                raise ValueError("Canonical QA source read failed; no paid request submitted")
            return self.body(event)
        sources = read("qa-source-enrollment", f"/api/v1/workspaces/{quote(workspace, safe='')}/sources")
        enrolled = [row for row in sources if row.get("media_id") == media_id and row.get("workspace_id") == workspace]
        if len(enrolled) != 1 or not isinstance(enrolled[0].get("id"), str) or not enrolled[0]["id"]:
            raise ValueError("Canonical QA requires one unambiguous owned source enrollment")
        versions = read("qa-source-version", f"/api/v1/media/{media_id}/versions?include_content=true&limit=1")
        if not isinstance(versions, list) or len(versions) != 1 or versions[0].get("media_id") != media_id or not isinstance(versions[0].get("content"), str):
            raise ValueError("Canonical QA source version is unavailable")
        content = versions[0]["content"]
        page_map = page_map_for_content(versions[0].get("safe_metadata"), content)
        if not page_map or len(page_map) != self.config["source"]["pdf_pages"]:
            raise ValueError("Canonical QA page map does not match its source content")
        nav = read("qa-source-navigation", f"/api/v1/media/{media_id}/navigation")
        if nav.get("media_id") != media_id or not nav.get("navigation_version"):
            raise ValueError("Canonical QA navigation identity is unavailable")
        return {"content": content, "page_map": page_map, "source_id": enrolled[0]["id"], "navigation_version": nav["navigation_version"]}

    def canonical_qa_passages(self, case: dict, context: str) -> tuple[list[dict], list[int]]:
        """Clip reference pages or exact retrieved fragments without bridging page gaps."""
        from tldw_Server_API.app.core.Research_Workspace.section_summaries import (
            BATCH_CHARS,
            build_section_extraction_prompt,
        )

        snapshot = self.canonical_source()
        content, page_map = snapshot["content"], snapshot["page_map"]
        fragments = []
        if context == "reference_pages":
            pages = [row["pdf_page"] for row in case["source_pages"]]
            fragments = [(page_map[page - 1]["start_char"], page_map[page - 1]["end_char"]) for page in pages]
        else:
            mode = self.config["qa"].get("retrieval_mode", self.config["retrieval"]["modes"][0])
            event = self.retrieve(case, mode)
            if event["outcome"] != "completed" or not event["checks"].get("source_scope_valid"):
                raise UnsafeResume("Retrieval failed or escaped the owned source; no QA request submitted")
            pages = event["checks"]["covered_pages"]
            for doc in result_documents(self.body(event)):
                text = str(doc.get("content", doc.get("text", doc.get("pageContent", ""))))
                metadata = {**doc, **(doc.get("metadata") or {})}
                start, end = metadata.get("start_char"), metadata.get("end_char")
                if not (type(start) is int and type(end) is int and 0 <= start < end <= len(content) and content[start:end] == text):
                    start = content.find(text) if text else -1
                    if start < 0 or content.find(text, start + 1) >= 0:
                        raise ValueError("Retrieved QA evidence is not an exact unambiguous canonical source excerpt")
                    end = start + len(text)
                fragments.append((start, end))
        if not fragments or sum(end - start for start, end in fragments) > BATCH_CHARS:
            raise ValueError("Canonical QA context is empty or exceeds the production extraction budget")
        passages = []
        for start, end in fragments:
            _, split = build_section_extraction_prompt(case["id"], "", content[start:end])
            for passage in split:
                passages.append({**passage, "id": f"P{len(passages) + 1}", "start_char": start + passage["start_char"],
                    "end_char": start + passage["end_char"], "source_id": snapshot["source_id"], "navigation_version": snapshot["navigation_version"]})
        return passages, pages

    def verify_answer(self, answer: dict, model: dict) -> dict:
        """Check the retained answer against its actual owned canonical P passages."""
        from Helper_Scripts.Evals.political_pdf_eval_report import verification_metrics

        request = json.loads((self.directory / answer["request_path"]).read_text())
        passages = request["context_evidence"]["passages"]
        content = self.body(answer)["choices"][0]["message"]["content"]
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content))
        from tldw_Server_API.app.core.Research_Workspace.section_verification import batch_claim_text
        claims = [{**row, "id": identifier} for row in parsed["evidence"] for identifier in row["source_span_ids"]]
        text = parsed["summary"] + "\n\n" + batch_claim_text({"evidence": claims})
        config = self.config.get("verification", {})
        body = {"workspace_id": self.config["workspace_id"], "text": text,
            "api_provider": config.get("provider", model["provider"]), "model": config.get("model", model["model"]),
            "evidence": [{"id": row["id"], "quote": row["text"], **{key: row[key] for key in ("source_id", "start_char", "end_char", "navigation_version")}} for row in passages]}
        def checks(report):
            metrics = verification_metrics(report, text, {row["id"]: row["text"] for row in passages}, expected_identity=(body["api_provider"], body["model"]))
            return {"strict_verification": metrics, "passed": metrics["all_claims_supported"], "semantic_review": "pending"}
        return self.request(f"qa-verification:{model['key']}:{answer['case_id']}:{answer['context']}", "qa_verification", "POST",
            "/api/v1/research-workspace/citations/verify", body=body, metadata={key: answer[key] for key in ("case_id", "model_key", "context")}, checks=checks)

    def verification_regression(self, case: dict, model: dict) -> dict:
        """Submit a frozen fixed claim without generating or repairing it first."""
        from Helper_Scripts.Evals.political_pdf_eval_report import regression_metrics

        snapshot = self.canonical_source()
        evidence = []
        for row in case["evidence"]:
            start = snapshot["content"].find(row["quote"])
            if start < 0 or snapshot["content"].find(row["quote"], start + 1) >= 0:
                raise ValueError("Regression quote is not an exact unique canonical source excerpt")
            evidence.append({**row, "source_id": snapshot["source_id"], "navigation_version": snapshot["navigation_version"],
                "start_char": start, "end_char": start + len(row["quote"])})
        config = self.config.get("verification", {})
        body = {"workspace_id": self.config["workspace_id"], "text": case["text"], "evidence": evidence,
            "api_provider": config.get("provider", model["provider"]), "model": config.get("model", model["model"])}
        def checks(report):
            metrics = regression_metrics(report, body, case["expected_statuses"])
            return {**metrics, "passed": metrics["regression_expectation_met"], "human_review": "pending"}
        return self.request(f"verification-regression:{model['key']}:{case['id']}", "verification_regression", "POST",
            "/api/v1/research-workspace/citations/verify", body=body, metadata={"model_key": model["key"], "context": case["id"]},
            context_evidence={"expected_statuses": case["expected_statuses"], "label_provenance": "agent_source_audit", "human_review": "pending"}, checks=checks)

    def answer(self, case: dict, model: dict, context: str) -> dict:
        """Ask the production extraction prompt against reference or frozen retrieved evidence."""
        import pymupdf

        from tldw_Server_API.app.core.Research_Workspace.section_summaries import (
            BATCH_CHARS,
            build_section_extraction_prompt,
        )
        pages = [page["pdf_page"] for page in case["source_pages"]]
        strict = self.config.get("verification", {}).get("required", False)
        if strict:
            passages, pages = self.canonical_qa_passages(case, context)
            source = "\n\n".join(row["text"] for row in passages)
        elif context == "reference_pages":
            with pymupdf.open(self.config["source"]["path"]) as pdf:
                source = "\n\n".join(f"Physical PDF page {page}:\n{pdf[page - 1].get_text()}" for page in pages)
        else:
            mode = self.config["qa"].get("retrieval_mode", self.config["retrieval"]["modes"][0])
            retrieval = self.retrieve(case, mode)
            if retrieval["outcome"] != "completed" or not retrieval["checks"].get("source_scope_valid"):
                raise UnsafeResume("Retrieval failed or escaped the owned source; no QA request submitted")
            documents = result_documents(self.body(retrieval))
            source = "\n\n".join(str(doc.get("content", doc.get("text", doc.get("pageContent", "")))) for doc in documents)
            pages = retrieval["checks"]["covered_pages"]
        if not source.strip():
            raise ValueError("No evidence text is available; no QA request submitted")
        if len(source) > BATCH_CHARS:
            raise ValueError("QA context exceeds the production extraction budget; no paid request submitted")
        instruction = "Answer this question using only the supplied evidence: " + case["question"]
        if strict:
            prompt, _ = build_section_extraction_prompt(case["id"], instruction + " Cite supporting [P...] IDs inline in summary text as well as source_span_ids. Do not invent citations.", "")
            prompt += "\n\n".join(f"[{row['id']}] source characters [{row['start_char']}, {row['end_char']})\n{row['text']}" for row in passages)
        else:
            prompt, passages = build_section_extraction_prompt(case["id"], instruction, source)
        body = {"api_provider": model["provider"], "model": model["model"], "messages": [{"role": "user", "content": prompt}], "stream": False, "save_to_db": False,
                **{key: model[key] for key in ("max_tokens", "temperature", "top_p") if key in model}}
        if "reasoning_effort" in model:
            body["extra_body"] = {"reasoning_effort": model["reasoning_effort"]}
        return self.request(f"qa:{model['key']}:{case['id']}:{context}", "qa", "POST", "/api/v1/chat/completions", body=body,
            metadata={"model_key": model["key"], "case_id": case["id"], "context": context},
            context_evidence={"passages": passages, "pages": pages, "media_id": self.config["source"]["media_id"], "context_origin": "canonical_media_version" if strict else "legacy_pdf_or_retrieval"}, checks=lambda payload: completion_checks(payload, passages))

    def summarize(self, model: dict) -> dict:
        """Poll a durable job; terminal failures retain portions and never trigger a new POST."""
        from tldw_Server_API.app.core.Research_Workspace.section_overview_coverage import overview_coverage

        summary = self.config.get("summary", {})
        if "reasoning_effort" in model:
            raise ValueError("Summary Jobs cannot represent explicit reasoning_effort; use a separate default-reasoning summary config")
        if not summary.get("sections") or not self.config.get("workspace_id"):
            raise ValueError("Summary requires an existing workspace and explicit selected outline nodes")
        workspace = quote(self.config["workspace_id"], safe="")
        final_id = f"summary:{model['key']}"
        if any(row.get("job_id") and row.get("model_key") == model["key"] for row in self.events) and any(key == final_id or key.startswith(final_id + ":") for key in self.retry_operations):
            raise UnsafeResume("A known summary job cannot be resubmitted; resume it or use a new intentional run")
        prior = [row for row in self.events if row["operation_id"] == final_id and row["event"] == "finished"]
        if prior:
            return prior[-1]
        settings = {**{key: value for key, value in summary.items() if key not in {"artifact_type", "job_ids", "resume_artifact_ids", "wait_seconds"}},
                    **{key: model[key] for key in ("provider", "model", "max_tokens", "temperature", "top_p") if key in model}}
        job_id = summary.get("job_ids", {}).get(model["key"])
        adopted = job_id is not None
        submitted = None
        if not adopted:
            resume_id = summary.get("resume_artifact_ids", {}).get(model["key"])
            submitted = self.request(f"summary:{model['key']}:submit", "summary_submit", "POST", f"/api/v1/workspaces/{workspace}/outputs",
                body={"artifact_type": summary.get("artifact_type", "summary"), "source_ids": list(dict.fromkeys(row["source_id"] for row in summary["sections"])), "settings": settings,
                      **({"resume_artifact_id": resume_id} if resume_id else {})}, metadata={"model_key": model["key"]})
            if submitted["outcome"] != "completed":
                return submitted
            job_id = self.body(submitted)["job_id"]
        deadline = time.monotonic() + summary.get("wait_seconds", 3600)
        while True:
            sequence = sum(row["event"] == "started" and row["operation_id"].startswith(f"summary:{model['key']}:poll:") for row in self.events)
            polled = self.request(f"summary:{model['key']}:poll:{sequence}", "summary_poll", "GET", f"/api/v1/workspaces/{workspace}/outputs/{job_id}", metadata={"model_key": model["key"]},
                checks=lambda body: {"job_status": body.get("status"), "pending": body.get("status") not in {"completed", "failed", "cancelled"}})
            if polled["outcome"] not in {"completed", "pending"}:
                return polled
            result = self.body(polled)
            if adopted:
                recorded = (result.get("artifact") or {}).get("producer_metadata", {}).get("settings", {})
                if any(recorded.get(key) != value for key, value in settings.items()) or result.get("artifact_type", (result.get("artifact") or {}).get("artifact_type")) != summary.get("artifact_type", "summary"):
                    raise UnsafeResume("Adopted job does not match configured provider/model/selected sections")
            if result["status"] in {"completed", "failed", "cancelled"}:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return polled
            time.sleep(min(15, remaining))
        artifact = result.get("artifact") or {}
        readback = self.request(f"summary:{model['key']}:readback", "summary_readback", "GET", f"/api/v1/workspaces/{workspace}/artifacts", metadata={"model_key": model["key"]})
        saved = self.body(readback)
        rows = saved if isinstance(saved, list) else saved.get("artifacts", [])
        analysis = artifact.get("version_metadata", {}).get("workspaceClient", {}).get("data", {}).get("sectionAnalysis", {})
        sections = analysis.get("sections", [])
        batches = [batch for section in sections for batch in section.get("batches", [])]
        contents = {}
        for selected in summary["sections"]:
            node_id, version = selected["node_id"], selected["navigation_version"]
            node = self.request(f"summary-content:{node_id}", "summary_source", "GET",
                f"/api/v1/media/{self.config['source']['media_id']}/navigation/{quote(node_id, safe='')}/content?format=auto&navigation_version={quote(version, safe='')}")
            if node["outcome"] == "completed" and isinstance(self.body(node), dict):
                contents[node_id] = self.body(node)
        spans_valid = section_spans_valid(sections, contents, self.config["source"]["media_id"])
        selected_scope_valid = len(sections) == len(summary["sections"]) and {(row.get("source_id"), row.get("node_id"), row.get("navigation_version")) for row in sections} == {(row["source_id"], row["node_id"], row["navigation_version"]) for row in summary["sections"]}
        # The rendered artifact also contains individual chapter summaries and evidence lists.
        coverage = overview_coverage(analysis.get("synthesis", ""), sections)
        synthesis_failed = bool(analysis.get("synthesis_error_code") or analysis.get("synthesis_error"))
        structural_passed = (result["status"] == "completed" and artifact.get("status") in {"complete", "completed"}
            and artifact in rows and spans_valid and selected_scope_valid and coverage["complete"] and not synthesis_failed)
        checks = {"job_status": result["status"], "total_sections": len(sections), "completed_sections": sum(row.get("status") == "completed" for row in sections),
                  "failed_sections": sum(row.get("status") == "failed" for row in sections), "completed_portions": sum(row.get("status") == "completed" for row in batches),
                  "failed_portions": sum(row.get("status") == "failed" for row in batches), "artifact_readback_matches": artifact in rows, "usage_available": False,
                  "adopted_job": adopted, "job_elapsed_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(submitted["utc_time"])).total_seconds() if submitted else None,
                  "job_latency_basis": "submission-to-observed-terminal wall clock, including resume downtime" if submitted else "unavailable for adopted job",
                  "evidence_spans_valid": spans_valid, "selected_scope_valid": selected_scope_valid,
                  "overview_coverage": coverage, "overview_synthesis_failed": synthesis_failed,
                  "structural_passed": structural_passed, "passed": structural_passed}
        if self.config.get("verification", {}).get("required"):
            from Helper_Scripts.Evals.political_pdf_eval_report import summary_verification_metrics
            checks["strict_verification"] = summary_verification_metrics(analysis, expected_identity=(model["provider"], model["model"]))
            checks["passed"] &= checks["strict_verification"]["all_claims_supported"]
        return self.request(final_id, "summary", "LOCAL", "saved-job-artifact", local=lambda: result,
                            metadata={"model_key": model["key"]}, checks=lambda _: checks)

    def execute(self, stages: list[str]) -> list[dict]:
        """Execute explicitly selected stages; prerequisite source verification always runs."""
        if "summary" in stages and any("reasoning_effort" in model for model in self.config["models"]):
            raise ValueError("Summary Jobs cannot represent explicit reasoning_effort; select QA only or use default reasoning")
        results = self.source() if set(stages) - {"pack"} else []
        if any(row["outcome"] != "completed" for row in results):
            return results
        if "retrieval" in stages:
            results.extend(self.retrieve(case, mode) for case in self.plan["cases"] for mode in self.config["retrieval"]["modes"])
        if "qa" in stages:
            for model in self.config["models"]:
                for case in self.plan["cases"]:
                    for context in self.config["qa"]["contexts"]:
                        answer = self.answer(case, model, context)
                        results.append(answer)
                        if self.config.get("verification", {}).get("required") and answer["outcome"] == "completed":
                            results.append(self.verify_answer(answer, model))
        if "verification_regressions" in stages:
            if not self.plan.get("verification_regressions"):
                raise ValueError("The verification_regressions stage requires its frozen fixture config")
            results.extend(self.verification_regression(case, model) for model in self.config["models"] for case in self.plan["verification_regressions"])
        if "summary" in stages:
            results.extend(self.summarize(model) for model in self.config["models"])
        if "pack" in stages:
            results.append(self.request("pack-integrity", "pack", "LOCAL", "supplied-research-pack", local=lambda: pack_checks(Path(self.config["pack"]["path"]), self.plan["source_sha256"]), checks=lambda body: body))
        return results


def main(argv: list[str] | None = None) -> int:
    """Command entrypoint; plan/report are offline and execution is explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "resume", "report"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stages", default="source,retrieval", help="Comma-separated source,retrieval,qa,summary,pack,verification_regressions; summary is opt-in")
    parser.add_argument("--retry-operation", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from Helper_Scripts.Evals.political_pdf_eval_report import write_report
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            write_report(args.run_dir)
            return 0
        if not args.config:
            raise ValueError("--config is required")
        plan = load_plan(args.config)
        if args.command == "plan":
            print(encoded(plan).decode())
            return 0
        if not args.execute or not args.run_dir:
            raise ValueError("Execution requires --execute and --run-dir")
        stages = args.stages.split(",")
        if not set(stages) <= set(STAGES):
            raise ValueError("Unknown stage")
        from Helper_Scripts.common.repo_utils import configure_local_egress
        configure_local_egress(plan["config"]["api"]["base_url"])
        with Runner(plan, args.run_dir, resume=args.command == "resume", retry_operations=set(args.retry_operation)) as run:
            results = run.execute(stages)
        from Helper_Scripts.Evals.political_pdf_eval_report import write_report
        write_report(args.run_dir)
        return int(any(row["outcome"] != "completed" for row in results))
    except (ValueError, OSError, KeyError) as error:
        print(f"Evaluation stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
