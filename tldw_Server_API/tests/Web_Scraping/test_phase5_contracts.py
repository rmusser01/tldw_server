"""Internal crawl values, presence resolution, and compatibility boundaries."""

from dataclasses import FrozenInstanceError, fields

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import (
    CrawlInterrupted,
    CrawlOutcome,
    CrawlReport,
    DiscoveredLink,
    DiscoveryResult,
    PageData,
)
from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest, ScrapeContext
from tldw_Server_API.app.core.Web_Scraping.contracts.results import CrawlResult
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CrawlLimits, CrawlPlan
from tldw_Server_API.app.core.Web_Scraping.crawl.profiles import resolve_crawl_plan

pytestmark = pytest.mark.unit

CRAWL_DEFAULTS = {
    "page_attempts": 1000,
    "network_requests": 20000,
    "browser_attempts": 1000,
    "active_probes": 2000,
    "transfer_bytes": 1073741824,
    "discovered_candidates": 50000,
    "max_url_bytes": 65536,
    "discovered_url_bytes": 67108864,
    "frontier_entries": 10000,
    "execution_seconds": 1800,
}
CHECKPOINT_DEFAULTS = {"max_file_bytes": 134217728, "max_candidates": 50000, "max_frontier": 10000, "max_results": 1000}


@pytest.mark.parametrize("kind, defaults", [(CrawlLimits, CRAWL_DEFAULTS), (CheckpointLimits, CHECKPOINT_DEFAULTS)])
def test_exact_limits_defaults(kind, defaults):
    value = kind()
    assert {field.name: getattr(value, field.name) for field in fields(value)} == defaults
    with pytest.raises(FrozenInstanceError):
        setattr(value, next(iter(defaults)), 1)


@pytest.mark.parametrize("kind, defaults", [(CrawlLimits, CRAWL_DEFAULTS), (CheckpointLimits, CHECKPOINT_DEFAULTS)])
@pytest.mark.parametrize("bad", [True, False, 0, -1, float("inf"), float("nan"), "1", None])
def test_limits_reject_invalid_values(kind, defaults, bad):
    for name in defaults:
        with pytest.raises(ValueError):
            kind(**{name: bad})


def test_count_limits_reject_fractional_but_duration_accepts_it():
    for name in CRAWL_DEFAULTS.keys() - {"execution_seconds"}:
        with pytest.raises(ValueError):
            CrawlLimits(**{name: 1.5})
    assert CrawlLimits(execution_seconds=0.5).execution_seconds == 0.5


def test_checkpoint_bounds_validate_against_effective_crawl_allowances():
    CheckpointLimits().validate_for(CrawlLimits())
    with pytest.raises(ValueError):
        CheckpointLimits().validate_for(CrawlLimits(page_attempts=1001))
    CheckpointLimits(max_results=1001).validate_for(CrawlLimits(page_attempts=1001))


def test_discovery_keeps_request_final_and_parent_locations_distinct():
    discovery = DiscoveryResult(
        "https://a/start", "https://a/final", (DiscoveredLink("https://a/next", "https://a/final"),)
    )
    page = PageData({"content": "article"}, discovery)
    assert page.discovery.links[0].parent_url == "https://a/final"
    with pytest.raises(FrozenInstanceError):
        page.discovery.final_url = "https://other/"


@pytest.mark.parametrize("reason", ["frontier_exhausted", "page_cap_reached"])
def test_completed_articles_are_not_partial_results(reason):
    report = CrawlReport(({"content": "ok"},), CrawlOutcome("completed", reason, False))
    assert report.outcome.has_partial_results is False


@pytest.mark.parametrize(
    "reason",
    [
        "deadline_exhausted",
        "budget_exhausted",
        "transport_unavailable",
        "policy_denied",
        "execution_failed",
        "checkpoint_failed",
        "queue_overloaded",
        "queue_stopping",
        "job_payload_too_large",
        "cancelled",
    ],
)
def test_interrupted_reports_classify_successful_data_only(reason):
    report = CrawlReport.interrupted([{"error": "private failure"}, {"content": "ok"}], reason)
    assert report.outcome == CrawlOutcome("cancelled" if reason == "cancelled" else "partial", reason, True)
    assert report.articles == ({"error": "private failure"}, {"content": "ok"})
    empty = CrawlReport.interrupted([{"error": "private failure"}, {"extraction_successful": False}], reason)
    assert empty.outcome == CrawlOutcome("cancelled" if reason == "cancelled" else "failed", reason, False)
    assert empty.articles == ({"error": "private failure"}, {"extraction_successful": False})


@pytest.mark.parametrize(
    "args",
    [
        ("completed", "execution_failed", False),
        ("partial", "budget_exhausted", False),
        ("failed", "budget_exhausted", True),
        ("cancelled", "execution_failed", False),
        ("completed", "page_cap_reached", True),
        ("unknown", "execution_failed", False),
        ("failed", "private exception", False),
        ("partial", "cancelled", True),
    ],
)
def test_outcome_rejects_incoherent_or_unallowlisted_values(args):
    with pytest.raises(ValueError):
        CrawlOutcome(*args)


def test_interruption_carries_sanitized_data_and_plain_extra_metadata():
    source = {
        "content": "ok",
        "metadata": {"nested": {"labels": ["kept"]}, "api_key": "secret"},
        "custom_extra": {"rank": 7},
        "custom_cookies": [{"value": "secret"}],
    }
    report = CrawlReport.interrupted([source], "execution_failed")
    source["metadata"]["nested"]["labels"].append("changed")
    error = CrawlInterrupted(report)
    assert error.report.articles[0]["metadata"] == {"nested": {"labels": ["kept"]}}
    assert error.report.articles[0]["custom_extra"] == {"rank": 7}
    assert "secret" not in repr(error.report)
    assert str(error) == "execution_failed"


@pytest.mark.parametrize(
    "credential_key",
    [
        "X-API-Key",
        "x_api_key",
        "client_secret",
        "clientSecret",
        "OPENAI_API_KEY",
        "anthropic-api-key",
        "provider_client_secret",
        "X-Auth-Token",
        "Set-Cookie",
        "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key",
        "private-key",
    ],
)
def test_retained_report_removes_nested_credential_aliases_but_preserves_prose_and_extras(credential_key):
    prose = "Documentation discusses X-API-Key and client_secret without exposing their values."
    source = {
        "content": prose,
        "metadata": {"nested": [{credential_key: "private-credential", "rank": 7}]},
        "custom_extra": {"token_count": 3, "secretary": "editor"},
    }
    report = CrawlReport.interrupted([source], "execution_failed")
    assert report.articles == (
        {
            "content": prose,
            "metadata": {"nested": [{"rank": 7}]},
            "custom_extra": {"token_count": 3, "secretary": "editor"},
        },
    )
    assert "private-credential" not in repr(CrawlInterrupted(report).report)


def resolve(profile="enhanced_recursive", options=None, supplied=frozenset(), config=None, request=None):
    return resolve_crawl_plan(
        request or CrawlRequest("https://example.com"),
        profile=profile,
        options=options or {},
        supplied=supplied,
        config=config or {},
        limits=CrawlLimits(),
    )


def test_presence_preserves_explicit_default_false_and_zero_against_config():
    config = {"web_crawl_include_external": True, "web_crawl_score_threshold": 0.99}
    omitted = resolve(config=config)
    explicit = resolve(
        config=config,
        options={"include_external": False, "score_threshold": 0.0},
        supplied=frozenset({"include_external", "score_threshold"}),
    )
    assert omitted.options["include_external"] is True
    assert omitted.options["score_threshold"] == 0.99
    assert explicit.options["include_external"] is False
    assert explicit.options["score_threshold"] == 0.0
    assert explicit.provenance["include_external"] == "supplied"
    assert omitted.provenance["include_external"] == "config"


def test_request_constructor_default_is_not_inferred_as_supplied():
    plan = resolve(
        config={"web_crawl_include_external": True}, request=CrawlRequest("https://example.com", max_pages=7)
    )
    assert plan.options["include_external"] is True
    assert plan.options["max_pages"] == 7
    assert plan.provenance["max_pages"] == "request"


def test_supplied_request_field_does_not_require_duplicate_options_value():
    plan = resolve(
        request=CrawlRequest("https://example.com", include_external=False),
        supplied=frozenset({"include_external"}),
        config={"web_crawl_include_external": True},
    )
    assert plan.options["include_external"] is False
    assert plan.provenance["include_external"] == "supplied"


def test_existing_override_names_preserve_false_zero_and_presence():
    plan = resolve(
        options={"include_external_override": False, "score_threshold_override": 0.0},
        supplied=frozenset({"include_external_override", "score_threshold_override"}),
        config={"web_crawl_include_external": True, "web_crawl_score_threshold": 0.9},
    )
    assert plan.options["include_external"] is False
    assert plan.options["score_threshold"] == 0.0


def test_options_without_supplied_marker_do_not_override_config():
    plan = resolve(options={"include_external": False}, config={"web_crawl_include_external": True})
    assert plan.options["include_external"] is True


def test_explicit_none_and_omission_have_distinct_provenance():
    omitted = resolve(config={"web_crawl_include_external": True})
    explicit = resolve(
        options={"include_external": None},
        supplied=frozenset({"include_external"}),
        config={"web_crawl_include_external": True},
    )
    assert dict(omitted.options) == dict(explicit.options)
    assert omitted.provenance["include_external"] == "config"
    assert explicit.provenance["include_external"] == "supplied:config"


@pytest.mark.parametrize("strategy", [None, "", "default", "best-first", "BESTFIRST"])
def test_strategy_retains_existing_truthiness_and_spelling(strategy):
    plan = resolve(options={"crawl_strategy": strategy}, supplied=frozenset({"crawl_strategy"}))
    assert plan.options["crawl_strategy"] == (strategy or "best_first").strip().lower()


@pytest.mark.parametrize("source", ["config", "supplied"])
@pytest.mark.parametrize("value", [{"api_key": "private-key"}, {}, ["private-key"], (), {"private-key"}, lambda: None])
def test_resolver_rejects_unsafe_strategy_before_coercion(source, value):
    with pytest.raises(ValueError, match="option") as caught:
        if source == "config":
            resolve(config={"web_crawl_strategy": value})
        else:
            resolve(options={"crawl_strategy": value}, supplied=frozenset({"crawl_strategy"}))
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("source", ["config", "supplied"])
def test_strategy_rejection_never_calls_private_object_string_conversion(source):
    calls = []

    class PrivateStrategy:
        def __str__(self):
            calls.append("coerced")
            return "private-key"

    with pytest.raises(ValueError) as caught:
        if source == "config":
            resolve(config={"web_crawl_strategy": PrivateStrategy()})
        else:
            resolve(options={"crawl_strategy": PrivateStrategy()}, supplied=frozenset({"crawl_strategy"}))
    assert calls == []
    assert str(caught.value) == "crawl option must be a scalar or a domain-weight mapping"


def test_explicit_none_uses_config_and_invalid_threshold_keeps_config():
    plan = resolve(
        options={"include_external": None, "score_threshold": "bad"},
        supplied=frozenset({"include_external", "score_threshold"}),
        config={"web_crawl_include_external": True, "web_crawl_score_threshold": 0.7},
    )
    assert plan.options["include_external"] is True
    assert plan.options["score_threshold"] == 0.7


def test_profiles_do_not_share_enhanced_defaults_with_legacy():
    legacy = resolve(
        "legacy_recursive", config={"web_crawl_include_external": True, "web_crawl_strategy": "best_first"}
    )
    enhanced = resolve()
    assert legacy.options["include_external"] is False
    assert "crawl_strategy" not in legacy.options
    assert enhanced.options["max_pages"] == 100
    assert enhanced.options["max_depth"] == 3


def test_plan_freezes_safe_config_only_and_discards_credential_metadata():
    config = {
        "web_crawl_domain_map": {"example.com": 0.5},
        "api_key": "secret",
        "headers": {"Authorization": "secret"},
        "web_crawl_enable_domain_map": True,
    }
    plan = resolve(
        config=config,
        options={"custom_cookies": [{"value": "secret"}]},
        supplied=frozenset({"custom_cookies"}),
        request=CrawlRequest("https://example.com", context=ScrapeContext(metadata={"config": config})),
    )
    config["web_crawl_domain_map"]["example.com"] = 1
    assert plan.options["domain_map"]["example.com"] == 0.5
    assert plan.request.context.metadata == {}
    assert "secret" not in repr(plan)
    with pytest.raises(TypeError):
        plan.options["domain_map"]["example.com"] = 1
    with pytest.raises(FrozenInstanceError):
        plan.profile = "other"


def test_resolver_preserves_existing_scalar_domain_weights_as_immutable_floats():
    weights = {
        "example.com": "0.5",
        "other.example": " 0.75 ",
        "small.example": "1e-2",
        "zero.example": 0,
        "float.example": 0.25,
        "bool.example": True,
    }
    plan = resolve(config={"web_crawl_enable_domain_map": True, "web_crawl_domain_map": weights})
    weights["example.com"] = "9.0"
    assert dict(plan.options["domain_map"]) == {
        "example.com": 0.5,
        "other.example": 0.75,
        "small.example": 0.01,
        "zero.example": 0.0,
        "float.example": 0.25,
        "bool.example": 1.0,
    }
    assert all(type(weight) is float for weight in plan.options["domain_map"].values())
    with pytest.raises(TypeError):
        plan.options["domain_map"]["example.com"] = 1.0


@pytest.mark.parametrize("weight", [{"api_key": "private-key"}, ["private-key"], lambda: 0.5, "private-key"])
def test_resolver_rejects_unsafe_domain_weights_without_private_errors(weight):
    with pytest.raises(ValueError, match="option") as caught:
        resolve(config={"web_crawl_enable_domain_map": True, "web_crawl_domain_map": {"example.com": weight}})
    assert "private-key" not in str(caught.value)


def test_domain_weight_rejection_does_not_execute_float_conversion():
    calls = []

    class ExecutableWeight:
        def __float__(self):
            calls.append("executed")
            return 0.5

    with pytest.raises(ValueError) as caught:
        resolve(
            config={"web_crawl_enable_domain_map": True, "web_crawl_domain_map": {"example.com": ExecutableWeight()}}
        )
    assert calls == []
    assert str(caught.value) == "domain_map option must map domain strings to numeric weights"


def test_direct_plan_constructor_cannot_bypass_safe_options_and_provenance():
    plan = CrawlPlan(
        CrawlRequest("https://example.com"),
        "enhanced_recursive",
        {"api_key": "secret", "max_pages": 1},
        {"api_key": "secret", "max_pages": "supplied"},
        CrawlLimits(),
    )
    assert dict(plan.options) == {"max_pages": 1}
    assert dict(plan.provenance) == {"max_pages": "supplied"}
    with pytest.raises(ValueError):
        resolve(profile="unknown")


@pytest.mark.parametrize(
    "option, value",
    [
        ("method", {"api_key": "secret"}),
        ("domain_map", {"example.com": {"api_key": "secret"}}),
        ("keywords", [{"token": "secret"}]),
    ],
)
def test_safe_option_names_do_not_allow_arbitrary_credential_bearing_subtrees(option, value):
    with pytest.raises(ValueError, match="option"):
        CrawlPlan(CrawlRequest("https://example.com"), "enhanced_recursive", {option: value}, {}, CrawlLimits())


def test_report_rejects_success_claim_without_successful_data():
    with pytest.raises(ValueError):
        CrawlReport(({"error": "failure"},), CrawlOutcome("partial", "budget_exhausted", True))


@pytest.mark.parametrize("articles", [({"content": "kept"},), ({"error": "failure"}, {"content": "kept"})])
def test_failed_report_cannot_deny_retained_successful_data(articles):
    with pytest.raises(ValueError, match="outcome"):
        CrawlReport(articles, CrawlOutcome("failed", "execution_failed", False))


def test_interruption_cannot_wrap_normal_completion():
    with pytest.raises(ValueError):
        CrawlInterrupted(CrawlReport((), CrawlOutcome("completed", "frontier_exhausted", False)))


def test_public_constructors_keep_their_exact_fields():
    assert [f.name for f in fields(CrawlRequest)] == [
        "base_url",
        "max_pages",
        "max_depth",
        "include_external",
        "context",
    ]
    assert [f.name for f in fields(CrawlResult)] == ["base_url", "results", "status", "failure"]
