"""Offline mechanical comparisons of frozen political-PDF evaluation evidence.

This module does no inference or network I/O. Passage IDs and page coverage are
structural checks; they do not establish that a generated claim is faithful.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tldw_Server_API.app.core.Claims_Extraction.citation_syntax import extract_citation_ids

_IDENTITY = ("stage", "case_id", "model_key", "context", "mode")
_PATHS = ("request_path", "response_path", "artifact_path")
_OUTPUT_NAMES = ("report.json", "report.md", "semantic-review.json")
_LIMITATIONS = [
    "Structural checks do not measure semantic correctness; human review remains pending.",
    "Strict support verdicts are uncalibrated automated judgments, not human acceptance or real-world truth.",
    "H05 is a tuned regression, not an untouched held-out case.",
    "Summary provider-call usage is unavailable through the public Jobs API.",
    "Backend evidence does not certify browser export, offline reading, or cross-server UI restore.",
]


def _path(root: Path, name: Any) -> Path:
    """Resolve evidence/output paths without following a link outside the run."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Evidence path must be relative to the run directory")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Evidence path escapes the run directory")
    return path


def _read_json(path: Path) -> Any:
    """Read saved JSON, reporting only its filename on malformed evidence."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Unreadable JSON evidence: {path.name}") from exc


def _load(root: Path) -> tuple[dict, dict[str, dict], list[dict]]:
    """Validate the ledger before any derived output is written."""
    manifest = _read_json(_path(root, "manifest.json"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(manifest.get("config"), dict) or not isinstance(manifest["config"].get("models", []), list):
        raise ValueError("Manifest config must contain a model list")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or any(not isinstance(case, dict) or
            not isinstance(case.get("id"), str) or not case["id"] for case in cases):
        raise ValueError("Manifest cases require unique IDs")
    by_case = {case["id"]: case for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("Manifest cases require unique IDs")
    attempts: dict[tuple[str, int], dict] = {}
    identities: dict[str, tuple] = {}
    try:
        lines = _path(root, "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("Unreadable attempts ledger") from exc
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError("Malformed attempts ledger") from exc
        if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Unsupported attempt schema")
        op, attempt = record.get("operation_id"), record.get("attempt")
        if not isinstance(op, str) or not op or type(attempt) is not int or attempt < 1:
            raise ValueError("Attempt requires an operation ID and positive attempt number")
        if not isinstance(record.get("stage"), str) or not record["stage"]:
            raise ValueError("Attempt requires a stage")
        if any(record.get(key) is not None and not isinstance(record[key], str) for key in _IDENTITY):
            raise ValueError("Attempt identity fields must be strings")
        try:
            timestamp = datetime.fromisoformat(record["utc_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Attempt requires a valid UTC timestamp") from exc
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("Attempt requires a valid UTC timestamp")
        if record.get("case_id") is not None and record["case_id"] not in by_case:
            raise ValueError("Attempt references an unknown frozen case")
        identity = tuple(record.get(key) for key in _IDENTITY)
        if op in identities and identities[op] != identity:
            raise ValueError("Operation identity changed in the ledger")
        identities[op] = identity
        for key in _PATHS:
            if key in record:
                evidence = _path(root, record[key])
                if evidence in {root / name for name in _OUTPUT_NAMES}:
                    raise ValueError("Evidence path collides with a derived report")
                if not evidence.is_file():
                    raise ValueError("Evidence path is not a saved file")
        elapsed = record.get("elapsed_seconds")
        if elapsed is not None and (type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Attempt latency must be finite and nonnegative")
        key = (op, attempt)
        if record.get("event") == "started":
            if key in attempts or any(old_op == op and old_attempt >= attempt for old_op, old_attempt in attempts):
                raise ValueError("Duplicate or out-of-order attempt")
            attempts[key] = {**record, "outcome": "unknown"}
        elif record.get("event") == "finished":
            if key not in attempts or attempts[key]["event"] == "finished":
                raise ValueError("Orphan or duplicate finished attempt")
            if not isinstance(record.get("outcome"), str) or not record["outcome"]:
                raise ValueError("Finished attempt requires an outcome")
            if not isinstance(record.get("checks", {}), dict):
                raise ValueError("Attempt checks must be an object")
            attempts[key].update(record)
        else:
            raise ValueError("Unknown attempt event")
    return manifest, by_case, list(attempts.values())


def _pages(case: dict) -> list[int]:
    """Use frozen physical pages as the coverage denominator."""
    pages = case.get("source_pages", [])
    if not isinstance(pages, list) or any(not isinstance(row, dict) or
            type(row.get("pdf_page")) is not int or row["pdf_page"] < 1 for row in pages):
        raise ValueError("Frozen case has invalid physical pages")
    return sorted({row["pdf_page"] for row in pages})


def _qa_metrics(body: Any, request: Any) -> tuple[dict, Any, Any]:
    """Check the actual saved extraction shape and supplied passage identifiers."""
    body = body if isinstance(body, dict) else {}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    parsed: Any = None
    if isinstance(content, str):
        try:
            parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content))
        except ValueError:
            pass
    context = request.get("context_evidence", {}) if isinstance(request, dict) else {}
    passages = context.get("passages", []) if isinstance(context, dict) else []
    ids = {item["id"] for item in passages if isinstance(item, dict) and isinstance(item.get("id"), str)}
    evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
    structured = (isinstance(parsed, dict) and isinstance(parsed.get("summary"), str)
        and bool(parsed["summary"].strip()) and isinstance(evidence, list) and bool(evidence))
    cited: list[str] = []
    if structured:
        for item in evidence:
            spans = item.get("source_span_ids") if isinstance(item, dict) else None
            if (not isinstance(item, dict) or not isinstance(item.get("claim"), str) or not item["claim"].strip()
                or not isinstance(spans, list) or not 1 <= len(spans) <= len(ids)
                or any(not isinstance(span, str) for span in spans)):
                structured = False
                break
            cited.extend(spans)
    finish = choice.get("finish_reason")
    metrics = {"output_nonempty": isinstance(content, str) and bool(content.strip()),
        "structured_output_valid": bool(structured), "citation_ids_valid": bool(structured and set(cited) <= ids),
        "cited_span_ids": sorted(set(cited)), "unknown_span_ids": sorted(set(cited) - ids),
        "finish_reason": finish, "completion_not_truncated": finish == "stop",
        "semantic_review": "pending"}
    raw_usage = body.get("usage")
    usage = {key: raw_usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in raw_usage} if isinstance(raw_usage, dict) else {}
    if not usage or any(type(value) is not int or value < 0 for value in usage.values()):
        usage = None
    model = body.get("model")
    return metrics, model if isinstance(model, str) else None, usage



def verification_metrics(report: Any, text: str, evidence: dict[str, str], *, expected_identity: tuple | None = None) -> dict:
    """Validate a saved strict receipt and expose automated outcomes without human labels."""
    result = {"available": isinstance(report, dict), "integrity_valid": False,
        "verdict": "unverified", "all_claims_supported": False, "all_claims_evaluated": False,
        "coverage": {}, "finding_status_counts": {}, "human_review": "pending"}
    if not isinstance(report, dict) or not isinstance(text, str) or not isinstance(evidence, dict) or any(not isinstance(value, str) for value in evidence.values()):
        return result
    findings, coverage, recorded_evidence = (report.get(key) for key in ("findings", "coverage", "evidence"))
    result.update({"verdict": report.get("verdict", "unverified"), "provider": report.get("provider"), "model": report.get("model")})
    if not isinstance(findings, list) or not isinstance(coverage, dict) or not isinstance(recorded_evidence, dict):
        return result
    statuses = {"supported", "unsupported", "contradicted", "uncited", "not_checked", "error"}
    counts: Counter = Counter()
    valid = type(report.get("version")) is int and report["version"] == 1 and all(isinstance(report.get(key), str) and report[key].strip() for key in ("provider", "model")) and report.get("text_sha256") == hashlib.sha256(text.encode()).hexdigest()
    if expected_identity is not None:
        valid &= (report.get("provider"), report.get("model")) == expected_identity
    previous_end, ids = 0, set()
    for finding in findings:
        if not isinstance(finding, dict):
            valid = False
            continue
        start, end, cited, status = (finding.get(key) for key in ("start", "end", "evidence_ids", "status"))
        bounded = type(start) is int and type(end) is int and previous_end <= start < end <= len(text)
        valid &= bounded and isinstance(finding.get("id"), str) and bool(finding["id"]) and finding["id"] not in ids
        valid &= isinstance(status, str) and status in statuses and isinstance(finding.get("reason"), str)
        if bounded:
            valid &= finding.get("text") == text[start:end] and not any(character.isalnum() for character in text[previous_end:start])
            previous_end = end
        if isinstance(finding.get("id"), str):
            ids.add(finding["id"])
        if isinstance(status, str):
            counts[status] += 1
        valid_citations = isinstance(cited, list) and all(isinstance(value, str) for value in cited)
        valid &= valid_citations
        literal_ids = extract_citation_ids(finding.get("text", "") if isinstance(finding.get("text"), str) else "")
        valid &= cited == list(dict.fromkeys(literal_ids)) or (status in ("error", "not_checked") and cited == [])
        if status in ("supported", "unsupported", "contradicted"):
            valid &= bool(cited) and valid_citations and all(value in evidence and value in recorded_evidence for value in cited)
    valid &= not any(character.isalnum() for character in text[previous_end:])
    for identifier, item in recorded_evidence.items():
        valid &= isinstance(item, dict) and identifier in evidence and item.get("text_sha256") == hashlib.sha256(evidence.get(identifier, "").encode()).hexdigest()
    evaluated = sum(counts[status] for status in ("supported", "unsupported", "contradicted"))
    expected = {"total": len(findings), "evaluated": evaluated, "supported": counts["supported"],
        "uncovered": len(findings) - evaluated, "complete": bool(findings) and evaluated == len(findings)}
    expected_verdict = ("supported" if expected["complete"] and counts["supported"] == len(findings)
        else "needs_revision" if any(counts[key] for key in ("unsupported", "contradicted", "uncited")) else "unverified")
    valid &= all(type(coverage.get(key)) is int for key in ("total", "evaluated", "supported", "uncovered")) and type(coverage.get("complete")) is bool
    valid &= coverage == expected and report.get("verdict") == expected_verdict
    result.update({"integrity_valid": bool(valid), "coverage": expected, "reported_coverage": coverage,
        "finding_status_counts": dict(counts), "all_claims_evaluated": bool(valid and expected["complete"]),
        "all_claims_supported": bool(valid and expected_verdict == "supported")})
    return result


def regression_metrics(report: Any, submitted: dict, expected: list[str]) -> dict:
    """Compare frozen source-audited expectations; never call this model accuracy."""
    evidence = {item["id"]: item["quote"] for item in submitted.get("evidence", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("quote"), str)}
    metrics = verification_metrics(report, submitted.get("text", ""), evidence, expected_identity=(submitted.get("api_provider"), submitted.get("model")))
    observed = set(metrics["finding_status_counts"])
    return {"strict_verification": metrics, "regression_expectation_met": bool(metrics["integrity_valid"] and observed and observed <= set(expected)),
        "expected_statuses": expected, "expectation_provenance": "agent_source_audit"}


def summary_verification_metrics(analysis: Any, *, expected_identity: tuple | None = None) -> dict:
    """Account for every selected chapter, extraction portion and final overview."""
    from tldw_Server_API.app.core.Research_Workspace.section_verification import batch_claim_text

    analysis = analysis if isinstance(analysis, dict) else {}
    sections = analysis.get("sections", [])
    rows, all_evidence = [], {}
    for section_index, section in enumerate(sections, 1):
        evidence = {item["id"]: item["quote"] for batch in section.get("batches", []) for item in batch.get("evidence", [])
            if isinstance(item.get("id"), str) and isinstance(item.get("quote"), str)}
        all_evidence.update(evidence)
        for batch_index, batch in enumerate(section.get("batches", []), 1):
            own = {item["id"]: evidence[item["id"]] for item in batch.get("evidence", []) if item.get("id") in evidence}
            expected_text = batch_claim_text(batch) if all(isinstance(item.get("id"), str) for item in batch.get("evidence", [])) else ""
            metrics = verification_metrics(batch.get("claim_verification"), expected_text, own, expected_identity=expected_identity)
            if batch.get("verification_text") != expected_text:
                metrics.update(integrity_valid=False, all_claims_supported=False, all_claims_evaluated=False)
            rows.append({"kind": "batches", "section": section_index, "batch": batch_index, "generation_status": batch.get("status"), **metrics})
        rows.append({"kind": "chapters", "section": section_index, "title": section.get("title"), "generation_status": section.get("status"),
            **verification_metrics(section.get("claim_verification"), section.get("content", ""), evidence, expected_identity=expected_identity)})
    rows.append({"kind": "overview", **verification_metrics(analysis.get("synthesis_verification"), analysis.get("synthesis", ""), all_evidence, expected_identity=expected_identity)})
    kinds = ("batches", "chapters", "overview")
    return {"expected_reports": {kind: sum(row["kind"] == kind for row in rows) for kind in kinds},
        "valid_reports": {kind: sum(row["kind"] == kind and row["integrity_valid"] for row in rows) for kind in kinds},
        "all_claims_supported": bool(sections) and all(row["all_claims_supported"] for row in rows),
        "all_claims_evaluated": bool(sections) and all(row["all_claims_evaluated"] for row in rows),
        "human_review": "pending", "reports": rows}

def _attempt(root: Path, record: dict, case: dict, config: dict) -> dict:
    """Derive one attempt without dropping the runner's original checks."""
    row = {key: record[key] for key in ("attempt", "outcome", "utc_time", "elapsed_seconds", "http_status",
        "job_id", "artifact_id", *_PATHS) if key in record}
    checks = record.get("checks", {})
    row.update({"checks": checks, "metrics": dict(checks), "usage_status": "unavailable", "reported_usage": None})
    response = _read_json(_path(root, record["response_path"])) if "response_path" in record else {}
    body = response.get("body") if isinstance(response, dict) else None
    stage = record["stage"]
    if stage == "qa":
        request = _read_json(_path(root, record["request_path"])) if "request_path" in record else {}
        metrics, model, usage = _qa_metrics(body, request)
        row["metrics"].update(metrics)
        row.update({"returned_model": model, "reported_usage": usage,
            "usage_status": "reported" if usage is not None else "unavailable"})
    elif stage in ("qa_verification", "verification_regression"):
        request = _read_json(_path(root, record["request_path"])) if "request_path" in record else {}
        submitted = request.get("body", {}) if isinstance(request, dict) else {}
        evidence = {item["id"]: item["quote"] for item in submitted.get("evidence", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("quote"), str)}
        row["metrics"]["strict_verification"] = verification_metrics(body, submitted.get("text", ""), evidence, expected_identity=(submitted.get("api_provider"), submitted.get("model")))
        if stage == "verification_regression":
            expected = (request.get("context_evidence") or {}).get("expected_statuses", [])
            row["metrics"].update(regression_metrics(body, submitted, expected))
    elif stage == "summary" and isinstance(body, dict):
        analysis = (body.get("artifact") or {}).get("version_metadata", {}).get("workspaceClient", {}).get("data", {}).get("sectionAnalysis", {})
        configured = next((model for model in config.get("models", []) if model.get("key") == record.get("model_key")), {})
        row["metrics"]["strict_verification"] = summary_verification_metrics(analysis, expected_identity=(configured.get("provider"), configured.get("model")))
    elif stage == "retrieval":
        expected = _pages(case)
        covered = checks.get("covered_pages", [])
        if not isinstance(covered, list) or any(type(page) is not int or page < 1 for page in covered):
            raise ValueError("Invalid covered pages")
        matched = sorted(set(expected).intersection(covered))
        row["metrics"].update({"expected_pages": expected, "covered_expected_pages": matched,
            "expected_page_coverage": len(matched) / len(expected) if expected else None,
            "all_expected_pages_covered": bool(expected) and matched == expected})
    return row


def _model_comparisons(operations: list[dict], config: dict) -> list[dict]:
    """Keep per-model outcome/latency totals distinct from semantic accuracy."""
    models = {model["key"]: model for model in config.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("key"), str)}
    keys = dict.fromkeys([*models, *(op["model_key"] for op in operations if op.get("model_key"))])
    result = []
    for key in keys:
        owned = [op for op in operations if op.get("model_key") == key]
        attempts = [attempt for op in owned for attempt in op["attempts"]]
        model = models.get(key, {})
        result.append({"model_key": key, "provider": model.get("provider"), "requested_model": model.get("model"),
            "operation_count": len(owned), "attempt_count": len(attempts),
            "first_attempt_failures": sum(op["first_outcome"] in ("failed", "partial", "cancelled") for op in owned),
            "unknown_operations": sum(op["latest_outcome"] == "unknown" for op in owned),
            "latest_outcomes": dict(Counter(op["latest_outcome"] for op in owned)),
            "reported_elapsed_seconds": (sum(row["elapsed_seconds"] for row in attempts if "elapsed_seconds" in row)
                if any("elapsed_seconds" in row for row in attempts) else None),
            "latency_measurements": sum("elapsed_seconds" in row for row in attempts),
            "qa_usage_measurements": sum(row["usage_status"] == "reported" for row in attempts),
            "semantic_review": "pending"})
    return result


def _markdown(report: dict) -> str:
    """Render a compact comparison while retaining every attempt's disposition."""
    def cell(value: Any) -> str:
        return str(value if value is not None else "unavailable").replace("|", "\\|").replace("\n", " ")

    lines = ["# Political PDF evaluation report", "", *_LIMITATIONS, "",
        "| Model | Operations | First failures | Latest operation outcomes | Recorded operation seconds (not run wall time) |",
        "| --- | ---: | ---: | --- | ---: |"]
    for model in report["models"]:
        lines.append("| " + " | ".join(cell(model[key]) for key in (
            "model_key", "operation_count", "first_attempt_failures", "latest_outcomes", "reported_elapsed_seconds")) + " |")
    lines.extend(["", "## Attempts", "", "| Operation | Attempt | Outcome | Seconds | Usage | Returned model | Structural checks |",
        "| --- | ---: | --- | ---: | --- | --- | --- |"])
    for op in report["operations"]:
        for row in op["attempts"]:
            metrics = row["metrics"]
            detail = "See saved checks"
            if op["stage"] == "qa":
                detail = (f"shape={metrics['structured_output_valid']}; span IDs={metrics['citation_ids_valid']}; "
                    f"finish={metrics['finish_reason']}")
            elif op["stage"] == "retrieval":
                detail = (f"pages={len(metrics['covered_expected_pages'])}/{len(metrics['expected_pages'])}; "
                    f"scope={metrics.get('source_scope_valid', 'unavailable')}; "
                    f"overbroad={metrics.get('overbroad_ranges', 'unavailable')}")
            elif op["stage"] == "summary":
                detail = (f"sections={metrics.get('completed_sections', '?')}/{metrics.get('total_sections', '?')}; "
                    f"failed={metrics.get('failed_sections', '?')}; portions={metrics.get('completed_portions', '?')}; "
                    f"persisted={metrics.get('artifact_readback_matches', 'unavailable')}; "
                    f"job seconds={cell(metrics.get('job_elapsed_seconds'))} "
                    f"({metrics.get('job_latency_basis', 'unavailable')})")
            lines.append("| " + " | ".join(cell(value) for value in (op["operation_id"], row["attempt"],
                row["outcome"], row.get("elapsed_seconds"), row["usage_status"], row.get("returned_model"), detail)) + " |")
    strict_rows = [(op, row, row["metrics"].get("strict_verification")) for op in report["operations"] for row in op["attempts"]]
    if any(isinstance(checks, dict) for _, _, checks in strict_rows):
        lines.extend(["", "## Strict verification", "", "| Operation | Attempt | Receipt valid | Automated verdict | Evaluated / total | Supported | Uncovered |", "| --- | ---: | --- | --- | --- | ---: | ---: |"])
        for op, row, checks in strict_rows:
            if not isinstance(checks, dict):
                continue
            coverage = checks.get("coverage", {})
            lines.append("| " + " | ".join(cell(value) for value in (op["operation_id"], row["attempt"],
                checks.get("integrity_valid"), checks.get("verdict"), f"{coverage.get('evaluated', '?')} / {coverage.get('total', '?')}",
                coverage.get("supported"), coverage.get("uncovered"))) + " |")
    regressions = [(op, row) for op in report["operations"] if op["stage"] == "verification_regression" for row in op["attempts"]]
    if regressions:
        lines.extend(["", "## Source-audited regression expectations", "", "These are agent-authored source regressions, not human labels or a calibrated accuracy benchmark.", "", "| Operation | Expected statuses | Observed statuses | Expectation met |", "| --- | --- | --- | --- |"])
        for op, row in regressions:
            metrics = row["metrics"]
            lines.append("| " + " | ".join(cell(value) for value in (op["operation_id"], metrics.get("expected_statuses"), metrics.get("strict_verification", {}).get("finding_status_counts"), metrics.get("regression_expectation_met"))) + " |")
    summary_rows = [(op, row, checks) for op, row, checks in strict_rows if isinstance(checks, dict) and "expected_reports" in checks]
    if summary_rows:
        lines.extend(["", "## Saved summary verification coverage", "", "| Operation | Valid batch / chapter / overview receipts | Expected receipts | All text units evaluated | All supported |", "| --- | --- | --- | --- | --- |"])
        for op, _row, checks in summary_rows:
            lines.append("| " + " | ".join(cell(value) for value in (op["operation_id"], checks["valid_reports"], checks["expected_reports"], checks["all_claims_evaluated"], checks["all_claims_supported"])) + " |")
    lines.extend(["", "Detailed structural metrics and original check results: report.json.",
        "Human labels and frozen expected facts: semantic-review.json.", ""])
    return "\n".join(lines)


def _preserve_reviews(root: Path, reviews: list[dict]) -> None:
    """Keep human decisions only for the exact same frozen evidence."""
    path = _path(root, "semantic-review.json")
    if not path.exists():
        return
    existing = _read_json(path)
    if not isinstance(existing, dict) or existing.get("schema_version") != 1 or not isinstance(existing.get("items"), list):
        raise ValueError("Malformed existing semantic review packet")
    by_attempt = {}
    for item in existing["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("operation_id"), str) or type(item.get("attempt")) is not int:
            raise ValueError("Malformed existing semantic review item")
        key = (item["operation_id"], item["attempt"])
        if key in by_attempt:
            raise ValueError("Duplicate semantic review item")
        by_attempt[key] = item
    for item in reviews:
        previous = by_attempt.pop((item["operation_id"], item["attempt"]), None)
        if previous is not None:
            if previous.get("evidence_sha256") != item["evidence_sha256"]:
                raise ValueError("Previously reviewed evidence changed")
            for key in ("status", "labels", "reviewer", "notes"):
                if key in previous:
                    item[key] = previous[key]
    if by_attempt:
        raise ValueError("Previously reviewed attempts disappeared")


def write_report(run_dir: Path) -> dict:
    """Write deterministic offline reports from a version-1 immutable run ledger."""
    root = run_dir.resolve()
    destinations = {name: root / name for name in _OUTPUT_NAMES}
    if any(path.is_symlink() for path in destinations.values()):
        raise ValueError("Report output path must not be a symlink")
    manifest, cases, records = _load(root)
    operations: dict[str, dict] = {}
    reviews = []
    for record in records:
        case = cases.get(record.get("case_id"), {})
        if record["stage"] == "verification_regression":
            case = next((row for row in manifest.get("verification_regressions", []) if row.get("id") == record.get("context")), {})
        operation = operations.setdefault(record["operation_id"], {"operation_id": record["operation_id"],
            **{key: record[key] for key in _IDENTITY if key in record}, "attempts": []})
        row = _attempt(root, record, case, manifest["config"])
        operation["attempts"].append(row)
        operation["first_outcome"] = operation["attempts"][0]["outcome"]
        operation["latest_outcome"] = row["outcome"]
        if record["stage"] in ("qa", "qa_verification", "verification_regression", "summary") and "response_path" in record:
            frozen = {"case": case, "identity": {key: record.get(key) for key in _IDENTITY},
                "files": {key: hashlib.sha256(_path(root, record[key]).read_bytes()).hexdigest()
                    for key in _PATHS if key in record}}
            reviews.append({"operation_id": record["operation_id"], "attempt": record["attempt"],
                **{key: record[key] for key in (*_IDENTITY, *_PATHS) if key in record},
                "outcome": row["outcome"], "question": case.get("question"),
                "expected_source_facts": case.get("expected_source_facts", []),
                "regression_expected_statuses": case.get("expected_statuses"), "regression_reason": case.get("reason"),
                "source_pages": case.get("source_pages", []),
                "label_provenance": case.get("label_provenance"),
                "case_note": "H05 is a tuned regression" if case.get("id") == "H05" else "Frozen case",
                "evidence_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                "status": "pending", "labels": dict.fromkeys((
                    "faithfulness", "author_attribution", "disagreement", "completeness"))})
    report = {"schema_version": 1,
        **{key: manifest.get(key) for key in ("source_sha256", "fixture_sha256", "git_sha")},
        "operations": list(operations.values()), "limitations": _LIMITATIONS}
    report["models"] = _model_comparisons(report["operations"], manifest.get("config", {}))
    _preserve_reviews(root, reviews)
    outputs = {"report.json": json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        "report.md": _markdown(report),
        "semantic-review.json": json.dumps({"schema_version": 1, "source_sha256": manifest.get("source_sha256"),
            "fixture_sha256": manifest.get("fixture_sha256"), "items": reviews}, indent=2, ensure_ascii=False) + "\n"}
    # Replace finished files; truncating an existing hard link could erase input evidence.
    with TemporaryDirectory(dir=root, prefix=".report-") as temporary:
        for name, content in outputs.items():
            (Path(temporary) / name).write_text(content, encoding="utf-8")
        for name, destination in destinations.items():
            (Path(temporary) / name).replace(destination)
    return report
