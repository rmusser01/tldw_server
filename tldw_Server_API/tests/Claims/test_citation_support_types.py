"""Portable verification receipts remain data, with strict structural integrity."""

import copy
import hashlib
import json
import subprocess  # nosec B404 -- Fresh-interpreter regression with a fixed local script, no shell.
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given
from hypothesis import strategies as st


def receipt():
    text = "A historical claim [P1]"
    return {
        "version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verdict": "supported", "provider": "openai", "model": "test-model",
        "coverage": {"total": 1, "evaluated": 1, "supported": 1, "uncovered": 0, "complete": True},
        "findings": [{"id": "claim-1", "text": text, "start": 0, "end": len(text),
                      "evidence_ids": ["P1"], "status": "supported", "reason": "Source supports it."}],
        "evidence": {"P1": {"text_sha256": "a" * 64}}, "limits": {}, "context": {},
    }


def validate(raw):
    from tldw_Server_API.app.core.Claims_Extraction.citation_support_types import validate_claim_verification
    return validate_claim_verification(raw)


def test_historical_report_and_provenance_are_preserved_exactly_without_current_text_authority():
    raw = receipt()
    raw["evidence_sha256"] = "b" * 64
    raw["context"] = {"historical": True, "workspace_id": "old-workspace", "nested": [None, 1.5]}
    raw["evidence"]["P1"].update(source_id="old-source", media_id=42, page_spans=[{"page": 7}])
    original = copy.deepcopy(raw)
    assert validate(raw) is raw
    assert raw == original


@pytest.mark.parametrize("path,value", [
    (("version",), True), (("version",), 2), (("version",), "1"),
    (("text_sha256",), "a" * 63), (("text_sha256",), "G" * 64),
    (("evidence_sha256",), "invalid"), (("verdict",), "unverified"),
    (("provider",), None), (("model",), ""),
    (("coverage", "total"), 2), (("coverage", "total"), True),
    (("coverage", "evaluated"), "1"), (("coverage", "supported"), 0),
    (("coverage", "uncovered"), -1), (("coverage", "complete"), 1),
    (("coverage", "complete"), False), (("findings", 0, "start"), True),
    (("findings", 0, "start"), -1), (("findings", 0, "end"), 99),
    (("findings", 0, "text"), 2), (("findings", 0, "status"), "error"),
    (("findings", 0, "evidence_ids"), []), (("findings", 0, "evidence_ids"), ["P1", "P1"]),
    (("findings", 0, "evidence_ids"), ["P2"]), (("findings", 0, "reason"), 7),
    (("evidence", "P1", "text_sha256"), "invalid"),
    (("evidence", "P1", "attribution_context_sha256"), "invalid"),
    (("evidence",), {}), (("limits", "max_units"), True),
])
def test_malformed_receipt_is_rejected(path, value):
    raw = receipt()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate(raw)


@pytest.mark.parametrize("duplicate_id,overlap", [(True, False), (False, True)])
def test_repeated_finding_identity_and_overlapping_coordinates_are_rejected(duplicate_id, overlap):
    raw = receipt()
    extra = copy.deepcopy(raw["findings"][0])
    extra.update(id="claim-1" if duplicate_id else "claim-2", start=0 if overlap else extra["end"])
    extra["end"] = extra["start"] + len(extra["text"])
    raw["findings"].append(extra)
    raw["coverage"].update(total=2, evaluated=2, supported=2)
    with pytest.raises(ValueError):
        validate(raw)


@pytest.mark.parametrize("status,verdict,evaluated", [
    ("unsupported", "needs_revision", 1), ("contradicted", "needs_revision", 1),
    ("uncited", "needs_revision", 0), ("not_checked", "unverified", 0), ("error", "unverified", 0),
])
def test_all_existing_verdicts_keep_their_exact_counter_meaning(status, verdict, evaluated):
    raw = receipt()
    raw["findings"][0]["status"] = status
    if status == "uncited":
        raw["findings"][0]["evidence_ids"] = []
    raw["verdict"] = verdict
    raw["coverage"].update(evaluated=evaluated, supported=0, uncovered=1 - evaluated, complete=bool(evaluated))
    assert validate(raw) is raw


@pytest.mark.parametrize("reason", [None, "The client could not reach verification."])
def test_empty_and_unavailable_reports_allow_null_identity_without_claiming_support(reason):
    raw = receipt()
    raw.update(verdict="unverified", provider=None, model=None, findings=[], evidence={})
    raw["coverage"] = {"total": 0, "evaluated": 0, "supported": 0, "uncovered": 0, "complete": False}
    if reason:
        raw["context"]["unavailable_reason"] = reason
    assert validate(raw) is raw


@pytest.mark.parametrize("text", ["", "An unchecked historical answer."])
def test_existing_job_unavailable_receipt_remains_valid(text):
    from tldw_Server_API.app.core.Research_Workspace.section_verification import unavailable_verification
    raw = unavailable_verification(text, provider=None, model=None)
    assert validate(raw) is raw


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), {1: "non-string key"}, ("tuple",), {"set"}, object()])
def test_non_json_values_cannot_hide_in_preserved_context(invalid):
    raw = receipt()
    raw["context"]["extra"] = invalid
    with pytest.raises(ValueError):
        validate(raw)


def test_recursive_json_and_utf8_byte_overflow_are_rejected():
    recursive = receipt()
    recursive["context"]["self"] = recursive
    with pytest.raises(ValueError):
        validate(recursive)
    oversized = receipt()
    oversized["context"]["extra"] = "🦊" * 500_000
    with pytest.raises(ValueError):
        validate(oversized)


def test_overflow_findings_are_retained_beyond_provider_unit_limit():
    raw = receipt()
    raw.update(verdict="unverified", evidence={})
    raw["findings"] = [{"id": f"claim-{i}", "text": "x", "start": i, "end": i + 1,
                        "evidence_ids": [], "status": "not_checked", "reason": "text_or_unit_limit"}
                       for i in range(1100)]
    raw["coverage"] = {"total": 1100, "evaluated": 0, "supported": 0, "uncovered": 1100, "complete": False}
    assert validate(raw) is raw


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=500))
def test_unicode_finding_offsets_use_python_codepoints_and_survive_json_roundtrip(text):
    raw = receipt()
    raw["findings"][0].update(text=text, end=len(text))
    raw["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    restored = json.loads(json.dumps(raw, ensure_ascii=False))
    assert validate(restored) == raw


def test_api_response_names_reexport_the_core_models():
    from tldw_Server_API.app.api.v1.schemas import research_workspace_citations as api
    from tldw_Server_API.app.core.Claims_Extraction import citation_support_types as core
    from tldw_Server_API.app.core.Sync_Contracts import citation_support_types as contract
    for name in ("CitationSupportReport", "CitationSupportFinding", "CitationSupportCoverage"):
        assert getattr(api, name) is getattr(core, name) is getattr(contract, name)
    assert core.validate_claim_verification is contract.validate_claim_verification
    assert core.MAX_CLAIM_VERIFICATION_BYTES == contract.MAX_CLAIM_VERIFICATION_BYTES


def test_imported_chat_receipt_validates_without_loading_claim_engines_or_database():
    script = dedent("""
        import importlib.abc
        import json
        import sys
        from datetime import datetime, timezone

        class ForbidDatabaseAndClaims(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for prefix in ("tldw_Server_API.app.core.DB_Management", "tldw_Server_API.app.core.Claims_Extraction"):
                    if fullname == prefix or fullname.startswith(prefix + "."):
                        raise AssertionError("Pure chat import loaded " + fullname)

        sys.meta_path.insert(0, ForbidDatabaseAndClaims())
        from tldw_Server_API.app.core.Chat.conversation_import import ChatImportLimits, prepare_chat_import

        report = json.load(sys.stdin)
        metadata = {"claim_verification": report, "claim_verification_imported": True}
        moment = datetime(2026, 11, 16, tzinfo=timezone.utc)
        request = {
            "id": "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14", "title": "Historical import",
            "scope_type": "workspace", "workspace_id": "owned-workspace", "created_at": moment,
            "messages": [{"id": "m1", "parent_message_id": None, "role": "assistant",
                "content": report["findings"][0]["text"], "timestamp": moment, "images": [],
                "metadata_extra": metadata}],
        }
        prepared = prepare_chat_import(request, limits=ChatImportLimits(1000, 64, 64), now=moment,
            inspect_image=lambda _: "image/png")
        assert prepared.messages[0]["extra_metadata"] == {"sender_role": "assistant", **metadata}
        print("historical report preserved without database or claim engines")
    """)
    result = subprocess.run(  # nosec B603 -- Fixed interpreter/script; receipt enters only as JSON stdin.
        [sys.executable, "-c", script], input=json.dumps(receipt()),
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "historical report preserved without database or claim engines"
