"""Tests for the deterministic offline web-retrieval quality contract."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from Helper_Scripts.Evals.run_web_retrieval_quality_baseline import main

from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    get_accuracy,
    precision_score,
    recall_score,
    string_shingle_matching,
)
from tldw_Server_API.app.core.Evaluations.web_retrieval_quality import (
    ALGORITHM_VERSIONS,
    FIXTURE_SCHEMA_VERSION,
    FixtureValidationError,
    evaluate_fixture_suite,
    load_fixture_suite,
    render_human_summary,
    serialize_report,
    validate_fixture_suite,
)

pytestmark = pytest.mark.unit


REVISION = "f676e23549ea8ed82ef53493260621a05b281863"
REPOSITORY_ROOT = Path(__file__).parents[3]
FIXTURE_PATH = Path(__file__).parent / "fixtures/retrieval_quality/v1.json"
BASELINE_PATH = REPOSITORY_ROOT / "Docs/Evals/baselines/web_retrieval_quality_v1.json"


def _suite() -> dict[str, object]:
    """Return a complete in-memory fixture covering every v1 case kind."""
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "suite_id": "in-test-v1",
        "baseline_revision": REVISION,
        "cases": [
            {
                "id": "z-extraction",
                "kind": "extraction",
                "input": {
                    "url": "https://article.test/story",
                    "html": "<main>alpha beta gamma delta epsilon</main>",
                },
                "expected": {"text": "alpha beta gamma delta epsilon"},
                "observed": {
                    "text": "alpha beta gamma delta ending",
                    "output_text": "alpha beta gamma delta ending",
                },
            },
            {
                "id": "b-search",
                "kind": "search_order",
                "input": {
                    "provider_results": [
                        {
                            "provider": "one",
                            "url": "https://one.test/result",
                            "title": "One",
                        },
                        {
                            "provider": "two",
                            "url": "https://two.test/result",
                            "title": "Two",
                        },
                    ]
                },
                "expected": {
                    "ordered_urls": [
                        "https://one.test/result",
                        "https://two.test/result",
                    ]
                },
                "observed": {
                    "ordered_urls": [
                        "https://one.test/result",
                        "https://other.test/result",
                    ],
                    "output_text": "one then other",
                },
            },
            {
                "id": "a-crawl",
                "kind": "crawl_graph",
                "input": {
                    "start_url": "https://crawl.test/start",
                    "links": {
                        "https://crawl.test/start": ["https://crawl.test/next"],
                        "https://crawl.test/next": ["https://crawl.test/start"],
                    },
                    "page_limit": 2,
                },
                "expected": {
                    "visited_urls": [
                        "https://crawl.test/start",
                        "https://crawl.test/next",
                    ],
                    "stop_reason": "page_limit",
                },
                "observed": {
                    "visited_urls": [
                        "https://crawl.test/start",
                        "https://crawl.test/next",
                    ],
                    "stop_reason": "page_limit",
                    "output_text": "start next",
                },
            },
            {
                "id": "m-provenance",
                "kind": "provenance",
                "input": {"required_fields": ["source_url", "fingerprint"]},
                "expected": {},
                "observed": {
                    "record": {
                        "source_url": "https://source.test/item",
                        "fingerprint": "sha256:abc",
                    },
                    "output_text": "é🙂",
                },
            },
        ],
    }


def _case(report: dict[str, object], case_id: str) -> dict[str, object]:
    """Return one report case by its stable identifier."""
    cases = report["cases"]
    assert isinstance(cases, list)
    return next(case for case in cases if case["id"] == case_id)


def test_rejects_wrong_schema_version() -> None:
    """Reject fixture suites from unsupported schema versions."""
    suite = _suite()
    suite["schema_version"] = "future-version"

    with pytest.raises(FixtureValidationError, match="schema_version"):
        validate_fixture_suite(suite)


def test_rejects_duplicate_case_ids() -> None:
    """Reject duplicate stable case identifiers."""
    suite = _suite()
    cases = suite["cases"]
    assert isinstance(cases, list)
    cases[1]["id"] = cases[0]["id"]

    with pytest.raises(FixtureValidationError, match="duplicate case id"):
        validate_fixture_suite(suite)


@pytest.mark.parametrize("target", ["suite", "case"])
def test_rejects_unknown_fields(target: str) -> None:
    """Reject unknown fields at both suite and case boundaries."""
    suite = _suite()
    if target == "suite":
        suite["unexpected"] = True
    else:
        cases = suite["cases"]
        assert isinstance(cases, list)
        cases[0]["unexpected"] = True

    with pytest.raises(FixtureValidationError, match="unknown fields"):
        validate_fixture_suite(suite)


def test_rejects_unsupported_case_kind() -> None:
    """Reject fixture case kinds outside the v1 contract."""
    suite = _suite()
    cases = suite["cases"]
    assert isinstance(cases, list)
    cases[0]["kind"] = "browser_soak"

    with pytest.raises(FixtureValidationError, match="unsupported kind"):
        validate_fixture_suite(suite)


@pytest.mark.parametrize(
    ("case_index", "section", "field"),
    [
        (0, "input", "html"),
        (1, "expected", "ordered_urls"),
        (2, "observed", "stop_reason"),
        (3, "input", "required_fields"),
    ],
)
def test_rejects_missing_kind_specific_fields(
    case_index: int,
    section: str,
    field: str,
) -> None:
    """Reject missing required fields for each supported case kind."""
    suite = _suite()
    cases = suite["cases"]
    assert isinstance(cases, list)
    del cases[case_index][section][field]

    with pytest.raises(FixtureValidationError, match="missing fields"):
        validate_fixture_suite(suite)


def test_validates_all_four_kinds_and_returns_a_defensive_copy() -> None:
    """Validate all case kinds without retaining caller-owned containers."""
    suite = _suite()

    validated = validate_fixture_suite(suite)
    cases = suite["cases"]
    assert isinstance(cases, list)
    cases[0]["observed"]["text"] = "mutated"

    assert [case["kind"] for case in validated["cases"]] == [
        "extraction",
        "search_order",
        "crawl_graph",
        "provenance",
    ]
    assert validated["cases"][0]["observed"]["text"] != "mutated"


def test_rejects_boolean_page_limit() -> None:
    """Reject booleans masquerading as positive crawl limits."""
    suite = _suite()
    cases = suite["cases"]
    assert isinstance(cases, list)
    cases[2]["input"]["page_limit"] = True

    with pytest.raises(FixtureValidationError, match="page_limit"):
        validate_fixture_suite(suite)


def test_rejects_non_finite_observed_values() -> None:
    """Reject non-finite numbers from JSON-compatible observations."""
    suite = _suite()
    cases = suite["cases"]
    assert isinstance(cases, list)
    cases[3]["observed"]["record"]["confidence"] = float("inf")

    with pytest.raises(FixtureValidationError, match="non-finite"):
        validate_fixture_suite(suite)


def test_unicode_output_budget_distinguishes_characters_and_utf8_bytes() -> None:
    """Count Unicode characters separately from encoded byte length."""
    report = evaluate_fixture_suite(_suite())
    provenance = _case(report, "m-provenance")

    assert provenance["budget"] == {
        "characters": 2,
        "utf8_bytes": 6,
        "estimated_tokens": {
            "value": 1,
            "algorithm": "characters-ceil-div4-v1",
            "authoritative": False,
        },
    }


def test_extraction_metrics_reuse_existing_shingle_helpers() -> None:
    """Keep baseline extraction scores aligned with shared shingle helpers."""
    suite = _suite()
    extraction = suite["cases"][0]
    true = extraction["expected"]["text"]
    predicted = extraction["observed"]["text"]
    tp, fp, fn = string_shingle_matching(true=true, pred=predicted)

    report = evaluate_fixture_suite(suite)
    result = _case(report, "z-extraction")

    assert result["metrics"] == {
        "precision": round(precision_score(tp, fp, fn), 6),
        "recall": round(recall_score(tp, fp, fn), 6),
        "f1": round(
            2
            * precision_score(tp, fp, fn)
            * recall_score(tp, fp, fn)
            / (precision_score(tp, fp, fn) + recall_score(tp, fp, fn)),
            6,
        ),
        "accuracy": get_accuracy(true=true, pred=predicted),
    }


def test_report_is_sorted_and_stable_across_calls() -> None:
    """Produce sorted, byte-stable reports across equivalent inputs."""
    suite = _suite()

    first = evaluate_fixture_suite(suite)
    second = evaluate_fixture_suite(copy.deepcopy(suite))

    assert first == second
    assert [case["id"] for case in first["cases"]] == [
        "a-crawl",
        "b-search",
        "m-provenance",
        "z-extraction",
    ]
    assert serialize_report(first) == serialize_report(second)
    assert serialize_report(first).endswith("\n")


def test_algorithm_versions_exactly_match_the_v1_contract() -> None:
    """Pin every algorithm identifier declared by the v1 contract."""
    assert dict(ALGORITHM_VERSIONS) == {
        "budget": "char-utf8-budget-v1",
        "crawl": "ordered-visit-stop-v1",
        "extraction": "token-shingle-f1-v1",
        "provenance": "required-field-recall-v1",
        "search_order": "position-match-v1",
        "token_estimate": "characters-ceil-div4-v1",
    }


def test_human_summary_uses_stable_line_grammar() -> None:
    """Render a stable timestamp-free human summary grammar."""
    report = evaluate_fixture_suite(_suite())

    summary = render_human_summary(report)

    assert summary.splitlines()[0] == (
        f"suite=in-test-v1 baseline={REVISION} cases=4"
    )
    assert summary.splitlines()[-1].startswith(
        "total mean_case_score="
    )
    assert not summary.endswith("\n")


def test_checked_fixture_covers_the_minimal_current_dev_contract() -> None:
    """Ensure the checked fixture spans the current minimal retrieval surface."""
    suite = load_fixture_suite(FIXTURE_PATH)
    cases = {case["kind"]: case for case in suite["cases"]}

    assert set(cases) == {"extraction", "search_order", "crawl_graph", "provenance"}
    assert "<nav>" in cases["extraction"]["input"]["html"]
    assert "Home Topics Subscribe" not in cases["extraction"]["observed"]["text"]
    assert len(cases["search_order"]["input"]["provider_results"]) >= 2
    assert cases["crawl_graph"]["input"]["page_limit"] == 2
    assert "https://crawl.test/start" in cases["crawl_graph"]["input"]["links"][
        "https://crawl.test/one"
    ]
    assert cases["provenance"]["observed"]["record"]["truncated"] is False


def test_checked_fixture_generates_the_checked_baseline() -> None:
    """Regenerate the checked baseline exactly from the checked fixture."""
    suite = load_fixture_suite(FIXTURE_PATH)
    report = evaluate_fixture_suite(suite)

    assert serialize_report(report) == BASELINE_PATH.read_text(encoding="utf-8")


def test_cli_writes_json_and_stable_human_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write canonical JSON while printing the stable human summary."""
    destination = tmp_path / "report.json"

    assert main(["--fixture", str(FIXTURE_PATH), "--json-out", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8") == BASELINE_PATH.read_text(
        encoding="utf-8"
    )
    assert capsys.readouterr().out == render_human_summary(
        evaluate_fixture_suite(load_fixture_suite(FIXTURE_PATH))
    ) + "\n"
