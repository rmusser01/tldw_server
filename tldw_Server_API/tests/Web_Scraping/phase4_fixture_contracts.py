"""Registry and public assertion for Phase 4 predecessor differences."""

from __future__ import annotations

from collections.abc import Callable

from tldw_Server_API.tests.Web_Scraping.phase4_fixture_change1_contract import (
    CHANGE_1_CONTRACT,
)
from tldw_Server_API.tests.Web_Scraping.phase4_fixture_difference_engine import (
    ANY_PATH,
    Difference,
    DifferenceContract,
    DifferenceRule,
    collect_differences,
    format_path,
    path_matches,
    validate_json_value,
)

APPROVED_BEHAVIOR_CHANGES = {
    1: "default regex becomes non-terminal enrichment",
    2: "caller cancellation is re-raised",
    3: "sync entry points reject active event loops before side effects",
    4: "regex validation and execution failures are bounded and sanitized",
    5: "the individual URL service passes system_message",
    6: "raw-browser sync performs governed admission",
    7: "public exception text is replaced by stable sanitized codes",
    8: "extraction work submission is bounded",
    9: "direct Playwright routing installs egress controls",
    10: "moved observability removes sensitive and high-cardinality fields",
    11: "direct article acquisition enforces response size limits",
}


def _changed_from_to(expected: str, actual: str) -> Callable[[Difference], bool]:
    return lambda difference: (difference.expected == expected and difference.actual == actual)


def _unknown_strategy_metric_is_bounded(difference: Difference) -> bool:
    return isinstance(difference.expected, str) and difference.actual == "unknown"


_CHANGE_4_PREDECESSOR_PROFILE = {"outcome": "regex_error", "value": None}
_CHANGE_4_CURRENT_PROFILE = {
    "outcome": "returned",
    "value": {
        "cache_stats": {
            "selector_css_cache_size": 0,
            "selector_xpath_cache_size": 2,
        },
        "result": {
            "content": "AB123",
            "extraction_successful": True,
            "schema_fields": {"content": "AB123"},
            "schema_name": "regex_fallback",
            "url": "https://example.com/regex-fallback",
        },
    },
}


def _validate_change_4_selector_regex_failure_profile(actual: object, expected: object) -> None:
    assert not collect_differences(
        expected, _CHANGE_4_PREDECESSOR_PROFILE
    ), "Change 4 selector regex failure predecessor profile must be the sanitized regex_error outcome"
    assert not collect_differences(
        actual, _CHANGE_4_CURRENT_PROFILE
    ), "Change 4 selector regex failure current profile must return the original AB123 result and cache shape"


DIFFERENCE_CONTRACTS = {
    "change_1_default_regex_non_terminal": CHANGE_1_CONTRACT,
    "change_4_selector_regex_failure_returns_original": DifferenceContract(
        behavior_change=4,
        rules=(
            DifferenceRule(
                identifier="outcome",
                path=("outcome",),
                description="the escaped predecessor regex error becomes a safe returned result",
                validator=_changed_from_to("regex_error", "returned"),
                minimum_count=1,
            ),
            DifferenceRule(
                identifier="value",
                path=("value",),
                description="the predecessor has no value and current returns one JSON object",
                validator=lambda difference: difference.expected is None and type(difference.actual) is dict,
                minimum_count=1,
            ),
        ),
        profile_validator=_validate_change_4_selector_regex_failure_profile,
    ),
    "change_7_policy_error": DifferenceContract(
        behavior_change=7,
        rules=(
            DifferenceRule(
                identifier="policy_error",
                path=("result", "error"),
                description="policy exception text is replaced by a stable public code",
                validator=_changed_from_to(
                    "Outbound policy evaluation failed. Please contact system administrator.",
                    "policy_error",
                ),
                minimum_count=1,
            ),
        ),
        allow_predecessor_equality=True,
    ),
    "change_7_selector_invalid": DifferenceContract(
        behavior_change=7,
        rules=(
            DifferenceRule(
                identifier="selector_error",
                path=("result", "errors", ANY_PATH, "error"),
                description="selector parser text is replaced by a stable public code",
                validator=_changed_from_to("Invalid expression", "selector_invalid"),
                minimum_count=1,
            ),
        ),
        allow_predecessor_equality=True,
    ),
    "change_10_unknown_strategy_metric": DifferenceContract(
        behavior_change=10,
        rules=(
            DifferenceRule(
                identifier="unknown_strategy_metric",
                path=("metrics", ANY_PATH, "labels", "strategy"),
                description="unknown extraction strategies use the bounded metric label",
                validator=_unknown_strategy_metric_is_bounded,
                minimum_count=1,
            ),
        ),
    ),
    "change_11_response_too_large": DifferenceContract(
        behavior_change=11,
        rules=(
            DifferenceRule(
                identifier="response_size_error",
                path=("result", "error"),
                description="oversized direct responses return the approved stable code",
                validator=_changed_from_to(
                    "Upstream response accepted",
                    "response_too_large",
                ),
                minimum_count=1,
            ),
        ),
        allow_predecessor_equality=True,
    ),
}


def assert_predecessor_behavior(
    actual: object,
    expected: object,
    *,
    behavior_change: int | None = None,
    difference_contract: str | None = None,
) -> None:
    """Assert parity or validate every difference against one approved change contract."""
    validate_json_value(actual, label="actual")
    validate_json_value(expected, label="expected")
    if behavior_change is not None and (
        type(behavior_change) is not int or behavior_change not in APPROVED_BEHAVIOR_CHANGES
    ):
        raise ValueError("behavior_change must be one integer in range 1..11")
    differences = collect_differences(actual, expected)
    if behavior_change is None:
        if difference_contract is not None:
            raise ValueError("difference_contract requires behavior_change")
        if differences:
            difference = differences[0]
            issue = f": {difference.issue}" if difference.issue else ""
            raise AssertionError(f"Strict JSON comparison failed at {format_path(difference.path)}{issue}")
        return

    assert difference_contract is not None, "A difference contract is required for an approved change"
    try:
        contract = DIFFERENCE_CONTRACTS[difference_contract]
    except KeyError as exc:
        raise ValueError(f"Unknown difference contract: {difference_contract}") from exc
    if contract.behavior_change != behavior_change:
        raise ValueError(
            f"Difference contract {difference_contract!r} belongs to behavior change "
            f"{contract.behavior_change}, not {behavior_change}"
        )

    if not differences:
        assert contract.allow_predecessor_equality, (
            f"Difference contract {difference_contract!r} has no " "predecessor-equality profile"
        )
        if contract.profile_validator is not None:
            contract.profile_validator(actual, expected)
        return

    matched_counts = {rule.identifier: 0 for rule in contract.rules}
    assert len(matched_counts) == len(
        contract.rules
    ), f"Difference contract {difference_contract!r} contains duplicate rule identifiers"
    for difference in differences:
        matching_rules = [rule for rule in contract.rules if path_matches(rule.path, difference.path)]
        path = format_path(difference.path)
        issue = f" ({difference.issue})" if difference.issue else ""
        assert matching_rules, (
            f"Difference at {path}{issue} is not covered by difference contract " f"{difference_contract!r}"
        )
        assert len(matching_rules) == 1, (
            f"Difference at {path} is covered by more than one rule in " f"{difference_contract!r}"
        )
        rule = matching_rules[0]
        assert rule.validator(difference), f"Difference at {path} violates contract rule: {rule.description}"
        matched_counts[rule.identifier] += 1

    for rule in contract.rules:
        count = matched_counts[rule.identifier]
        assert rule.minimum_count <= count <= rule.maximum_count, (
            f"Difference contract {difference_contract!r} profile requires rule "
            f"{rule.identifier!r} count in {rule.minimum_count}..{rule.maximum_count}, "
            f"got {count}"
        )
    if contract.profile_validator is not None:
        contract.profile_validator(actual, expected)
