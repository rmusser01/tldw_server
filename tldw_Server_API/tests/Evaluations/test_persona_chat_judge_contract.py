"""Validate the offline Persona Chat judge contract fixture.

These tests keep the future LLM-as-judge slice contract-backed by checking
fixture redaction safety, taxonomy label linkage, and score schema before any
executable judge can trust the fixture data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json"
TAXONOMY_PATH = REPO_ROOT / "Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md"
CASE_ID_RE = re.compile(r"^PC-JUDGE-\d{3}$")
SOURCE_CASE_ID_RE = re.compile(r"^PC-CASE-\d{3}$")
LABEL_RE = re.compile(r"^PC-[A-Z]+-\d{3}$")
REQUIRED_SCORE_NAMES = frozenset(
    {
        "role_adherence",
        "boundary_behavior",
        "memory_semantics",
        "exemplar_use",
        "grounding_separation",
    }
)
FORBIDDEN_PRIVATE_PATTERNS = [
    ("macOS user path", re.compile(r"/Users/[^\s\"']+")),
    ("Linux home path", re.compile(r"/home/[^\s\"']+")),
    ("Linux root path", re.compile(r"/root(?:/|$)")),
    ("macOS private path", re.compile(r"/private/[^\s\"']+")),
    ("Windows user path", re.compile(r"\b[A-Za-z]:\\(?:Users|Documents and Settings)\\", re.IGNORECASE)),
    ("SQLite URL", re.compile(r"sqlite:///")),
    ("ChaChaNotes database path", re.compile(r"ChaChaNotes\.db")),
    ("API key token", re.compile(r"(sk-[A-Za-z0-9]|api[_-]?key|bearer\s+[A-Za-z0-9])", re.IGNORECASE)),
    ("private-data marker", re.compile(r"(real user|private memory|production prompt)", re.IGNORECASE)),
]


def _load_fixture() -> dict[str, Any]:
    """Load the checked-in judge contract fixture payload."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _require(condition: object, message: str) -> None:
    """Raise a pytest-friendly assertion error when a contract condition fails."""
    if not condition:
        raise AssertionError(message)


def _walk_strings(value: Any) -> list[str]:
    """Return all string leaves from a nested fixture record."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _walk_strings(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _walk_strings(entry)]
    return []


def _extract_taxonomy_labels(markdown: str) -> set[str]:
    """Extract `PC-*` labels from the taxonomy Failure Labels markdown table."""
    parts = markdown.split("## Failure Labels", maxsplit=1)
    if len(parts) != 2:
        return set()

    failure_labels_section = parts[1]
    next_heading = re.search(r"\n##\s+", failure_labels_section)
    if next_heading is not None:
        failure_labels_section = failure_labels_section[: next_heading.start()]

    labels: set[str] = set()
    for raw_line in failure_labels_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = line.strip().strip("|").split("|")
        if not cells:
            continue
        label = cells[0].strip().replace("`", "").replace("*", "").strip()
        if LABEL_RE.fullmatch(label):
            labels.add(label)
    return labels


def _taxonomy_labels() -> set[str]:
    """Load the current failure taxonomy labels from the repo review artifact."""
    text = TAXONOMY_PATH.read_text(encoding="utf-8")
    return _extract_taxonomy_labels(text)


def _assert_redaction_safe(text: str, case_id: str) -> None:
    """Assert fixture text contains no local paths, secrets, or private markers."""
    for pattern_name, pattern in FORBIDDEN_PRIVATE_PATTERNS:
        _require(not pattern.search(text), f"case contains {pattern_name}: {case_id}")


def _validate_deterministic_labels(labels: Any, known_labels: set[str], case_id: str) -> None:
    """Validate deterministic setup labels are present, unique, and known."""
    _require(
        isinstance(labels, list) and labels,
        f"deterministic labels must be a non-empty list for {case_id}",
    )
    _require(
        all(isinstance(label, str) and LABEL_RE.fullmatch(label) for label in labels),
        f"deterministic labels must use PC label format for {case_id}",
    )
    _require(len(set(labels)) == len(labels), f"deterministic labels must be unique for {case_id}")
    _require(set(labels).issubset(known_labels), f"deterministic labels must exist in taxonomy for {case_id}")


def _validate_scores(scores: Any, case_id: str) -> None:
    """Validate exact judge score axes and strict score value types."""
    _require(isinstance(scores, dict), f"scores must be an object for {case_id}")
    _require(
        set(scores.keys()) == REQUIRED_SCORE_NAMES,
        f"scores must contain exactly the required score fields for {case_id}",
    )
    for score_name in REQUIRED_SCORE_NAMES:
        score = scores[score_name]
        _require(
            score is None
            or (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and 0.0 <= score <= 1.0
            ),
            f"score must be null or a numeric value within 0..1 for {case_id}: {score_name}",
        )


def test_persona_chat_judge_contract_fixture_declares_offline_v1_contract() -> None:
    """Ensure fixture metadata preserves the offline V1 judge boundary."""
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
    """Validate case identity, persona input shape, labels, and redaction safety."""
    payload = _load_fixture()
    cases = payload["cases"]
    known_labels = _taxonomy_labels()
    _require(known_labels, "taxonomy failure labels must be discoverable")

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
        _validate_deterministic_labels(case["judge_input"].get("deterministic_labels"), known_labels, case["case_id"])

        joined = "\n".join(_walk_strings(case))
        _assert_redaction_safe(joined, case["case_id"])


def test_persona_chat_judge_outputs_match_taxonomy_and_calibration_contract() -> None:
    """Validate expected judge outputs against taxonomy and score contract."""
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

        _validate_scores(output["scores"], case["case_id"])


def test_taxonomy_label_parser_accepts_markdown_cell_variants() -> None:
    """Prove taxonomy extraction tolerates harmless markdown table formatting."""
    labels = _extract_taxonomy_labels(
        """
## Representative Case Set

| Case | Labels |
| --- | --- |
| PC-CASE-001 | `PC-ID-999` |

## Failure Labels

| Label | Name |
| --- | --- |
| PC-ID-001 | Plain spacing |
|`PC-MEM-001`| Backtick wrapped |
  | **PC-EX-003** | Bold wrapped with indentation |
| not-a-label | Ignored |

## Later Section

| PC-CASE-002 | Not a failure label |
| PC-ID-998 | Outside the Failure Labels table |
"""
    )

    _require(labels == {"PC-ID-001", "PC-MEM-001", "PC-EX-003"}, "parser must tolerate table formatting")


@pytest.mark.parametrize(
    "private_value",
    [
        "/Users/alice/project/private-note.txt",
        "/home/alice/project/private-note.txt",
        "/root/private-note.txt",
        r"C:\Users\Alice\private-note.txt",
    ],
)
def test_redaction_guard_rejects_common_local_path_variants(private_value: str) -> None:
    """Prove redaction checks catch common user-owned path shapes."""
    with pytest.raises(AssertionError):
        _assert_redaction_safe(private_value, "PC-JUDGE-999")


def test_score_validator_requires_complete_numeric_score_schema() -> None:
    """Prove score validation rejects missing axes and stringified numbers."""
    with pytest.raises(AssertionError):
        _validate_scores(
            {
                "role_adherence": 1.0,
                "boundary_behavior": 1.0,
            },
            "PC-JUDGE-999",
        )

    with pytest.raises(AssertionError):
        _validate_scores(
            {
                "role_adherence": "1.0",
                "boundary_behavior": 1.0,
                "memory_semantics": 1.0,
                "exemplar_use": None,
                "grounding_separation": 0.5,
            },
            "PC-JUDGE-999",
        )


def test_deterministic_label_validator_rejects_empty_unknown_or_malformed_labels() -> None:
    """Prove deterministic labels must be present, formatted, and known."""
    known_labels = {"PC-MEM-003"}

    for labels in ([], ["PC-MEM-003", ""], ["PC-MEM-999"], ["not-pc"]):
        with pytest.raises(AssertionError):
            _validate_deterministic_labels(labels, known_labels, "PC-JUDGE-999")
