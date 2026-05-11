from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json"
TAXONOMY_PATH = REPO_ROOT / "Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md"
CASE_ID_RE = re.compile(r"^PC-JUDGE-\d{3}$")
SOURCE_CASE_ID_RE = re.compile(r"^PC-CASE-\d{3}$")
LABEL_RE = re.compile(r"^PC-[A-Z]+-\d{3}$")
LOCAL_PATH_RE = re.compile(r"(/Users/|/private/|[A-Za-z]:\\|sqlite:///|ChaChaNotes\.db)")
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]|api[_-]?key|bearer\s+[A-Za-z0-9])", re.IGNORECASE)
PRIVATE_MARKER_RE = re.compile(r"(real user|private memory|production prompt)", re.IGNORECASE)


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _walk_strings(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _walk_strings(entry)]
    return []


def _taxonomy_labels() -> set[str]:
    text = TAXONOMY_PATH.read_text(encoding="utf-8")
    labels: set[str] = set()
    in_failure_table = False
    for line in text.splitlines():
        if line.startswith("## Failure Labels"):
            in_failure_table = True
            continue
        if in_failure_table and line.startswith("## "):
            break
        if in_failure_table and line.startswith("| PC-"):
            label = line.split("|", maxsplit=2)[1].strip()
            labels.add(label)
    return labels


def test_persona_chat_judge_contract_fixture_declares_offline_v1_contract() -> None:
    payload = _load_fixture()

    _require(
        payload["schema_version"] == "persona-chat-judge-contract/v1",
        "fixture must declare the Persona Chat judge contract schema version",
    )
    _require(payload["offline_only"] is True, "judge contract fixtures must be offline-only")
    _require(payload["runtime_gating_allowed"] is False, "V1 judge output cannot gate runtime chat")
    _require(
        payload["requires_human_calibration_before_trust"] is True,
        "judge output must require human calibration before use",
    )
    _require(
        payload["contract_doc"] == "Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md",
        "fixture must link to the contract document",
    )


def test_persona_chat_judge_contract_cases_are_bounded_and_redaction_safe() -> None:
    payload = _load_fixture()
    cases = payload["cases"]

    _require(len(cases) >= 2, "fixture must include at least two cases")
    _require(
        {case["expected_judge_output"]["verdict"] for case in cases} >= {"pass", "fail"},
        "fixture must include pass and fail calibration cases",
    )

    seen_case_ids: set[str] = set()
    for case in cases:
        _require(CASE_ID_RE.fullmatch(case["case_id"]), f"invalid case id: {case['case_id']}")
        _require(
            SOURCE_CASE_ID_RE.fullmatch(case["source_case_id"]),
            f"invalid source case id: {case['source_case_id']}",
        )
        _require(case["case_id"] not in seen_case_ids, f"duplicate case id: {case['case_id']}")
        seen_case_ids.add(case["case_id"])
        _require(case["judge_input"]["assistant_kind"] == "persona", "judge input must be persona-backed")
        _require(case["judge_input"]["assistant_id"].strip(), "judge input must include assistant id")
        _require(
            case["judge_input"]["persona_memory_mode"] in {"read_only", "read_write"},
            "judge input must use a supported persona memory mode",
        )
        _require(case["judge_input"]["user_input"].strip(), "judge input must include user input")
        _require(
            case["judge_input"]["response_observation"]["assistant_text"].strip(),
            "judge input must include assistant response observation",
        )

        joined = "\n".join(_walk_strings(case))
        _require(not LOCAL_PATH_RE.search(joined), f"case contains a local path marker: {case['case_id']}")
        _require(not SECRET_RE.search(joined), f"case contains a secret marker: {case['case_id']}")
        _require(
            not PRIVATE_MARKER_RE.search(joined),
            f"case contains a disallowed private-data marker: {case['case_id']}",
        )


def test_persona_chat_judge_outputs_match_taxonomy_and_calibration_contract() -> None:
    known_labels = _taxonomy_labels()
    _require(known_labels, "taxonomy failure labels must be discoverable")

    for case in _load_fixture()["cases"]:
        output = case["expected_judge_output"]
        _require(
            output["verdict"] in {"pass", "fail", "inconclusive"},
            f"invalid judge verdict for {case['case_id']}",
        )
        _require(isinstance(output["rationale"], str), f"rationale must be text for {case['case_id']}")
        _require(
            1 <= len(output["rationale"]) <= 400,
            f"rationale must be bounded for {case['case_id']}",
        )
        _require(output["evidence"], f"judge outputs must cite bounded evidence keys for {case['case_id']}")

        expected_flags = output["expected_flags"]
        _require(
            all(LABEL_RE.fullmatch(label) for label in expected_flags),
            f"expected flags must use PC label format for {case['case_id']}",
        )
        _require(
            set(expected_flags).issubset(known_labels),
            f"expected flags must exist in taxonomy for {case['case_id']}",
        )
        if output["verdict"] == "fail":
            _require(expected_flags, f"fail verdict must include labels for {case['case_id']}")
        if output["verdict"] == "pass":
            _require(expected_flags == [], f"pass verdict must not include labels for {case['case_id']}")

        for score_name, score in output["scores"].items():
            _require(
                score_name
                in {
                    "role_adherence",
                    "boundary_behavior",
                    "memory_semantics",
                    "exemplar_use",
                    "grounding_separation",
                },
                f"unexpected score name for {case['case_id']}: {score_name}",
            )
            _require(
                score is None or 0.0 <= float(score) <= 1.0,
                f"score must be null or within 0..1 for {case['case_id']}: {score_name}",
            )
