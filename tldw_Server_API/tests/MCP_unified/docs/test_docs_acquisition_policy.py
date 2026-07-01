from __future__ import annotations

import pytest

from mcp_unified.docs.acquisition.models import (
    FetchResponse,
    FetchResultStatus,
    IngestStatus,
    NormalizedURL,
    PolicyStatus,
    Resolver,
    ResolvedAddress,
    SourceDecision,
    Transport,
    URLRequest,
)
from mcp_unified.docs.acquisition.policy import SourcePolicy, has_url_credentials, normalize_url


def test_models_define_acquisition_contract_names() -> None:
    assert PolicyStatus is not None  # nosec B101
    assert FetchResultStatus is not None  # nosec B101
    assert IngestStatus is not None  # nosec B101
    assert NormalizedURL is not None  # nosec B101
    assert SourceDecision is not None  # nosec B101
    assert URLRequest is not None  # nosec B101
    assert ResolvedAddress is not None  # nosec B101
    assert FetchResponse is not None  # nosec B101
    assert Resolver is not None  # nosec B101
    assert Transport is not None  # nosec B101


def test_locked_down_ignores_domain_only_allow_rules() -> None:
    policy = SourcePolicy(
        web_source_profile="locked_down",
        preapproved_domains=("docs.example.com",),
    )

    decision = policy.evaluate("https://docs.example.com/reference")

    assert decision.status == "approval_required"  # nosec B101
    assert decision.reason == "source_approval_required"  # nosec B101


def test_locked_down_allows_explicit_allowed_url_prefixes() -> None:
    policy = SourcePolicy(
        web_source_profile="locked_down",
        allowed_url_prefixes=("https://docs.example.com/reference/",),
    )

    decision = policy.evaluate("https://docs.example.com/reference/page")

    assert decision.status == "allowed"  # nosec B101


def test_host_case_and_default_port_are_normalized_for_prefix_matching() -> None:
    policy = SourcePolicy(
        web_source_profile="locked_down",
        allowed_url_prefixes=("https://docs.example.com/reference/",),
    )

    decision = policy.evaluate("https://DOCS.EXAMPLE.COM:443/reference/page")

    assert decision.status == "allowed"  # nosec B101
    assert decision.redacted_url == "https://docs.example.com/reference/page"  # nosec B101


def test_local_first_unknown_public_domain_requires_approval_with_hash() -> None:
    policy = SourcePolicy(web_source_profile="local_first")

    decision = policy.evaluate("https://unknown.example/reference?token=secret")

    assert decision.status == "approval_required"  # nosec B101
    assert decision.reason == "source_approval_required"  # nosec B101
    assert decision.safe_argument_hash  # nosec B101


def test_online_capable_unknown_public_domain_requires_arbitrary_domain_flag() -> None:
    policy = SourcePolicy(web_source_profile="online_capable")

    decision = policy.evaluate("https://unknown.example/reference")

    assert decision.status == "approval_required"  # nosec B101
    assert decision.reason == "source_approval_required"  # nosec B101


def test_online_capable_allows_unknown_public_domain_when_flag_is_true() -> None:
    policy = SourcePolicy(
        web_source_profile="online_capable",
        allow_arbitrary_public_domains=True,
    )

    decision = policy.evaluate("https://unknown.example/reference")

    assert decision.status == "allowed"  # nosec B101


@pytest.mark.parametrize(
    "raw_url",
    [
        "file:///etc/passwd",
        "ftp://example.com/reference",
        "https:///reference",
        "http://:80/reference",
    ],
)
def test_unsupported_or_malformed_urls_are_denied(raw_url: str) -> None:
    policy = SourcePolicy(web_source_profile="online_capable", allow_arbitrary_public_domains=True)

    decision = policy.evaluate(raw_url)

    assert decision.status == "denied"  # nosec B101


def test_credential_urls_are_denied_with_specific_reason() -> None:
    policy = SourcePolicy(web_source_profile="online_capable", allow_arbitrary_public_domains=True)

    decision = policy.evaluate("https://user:password@docs.example.com/reference")

    assert decision.status == "denied"  # nosec B101
    assert decision.reason == "url_credentials_denied"  # nosec B101
    assert has_url_credentials("https://user:password@docs.example.com/reference") is True  # nosec B101


def test_denied_domains_take_precedence_over_allowed_domains() -> None:
    policy = SourcePolicy(
        web_source_profile="local_first",
        preapproved_domains=("docs.example.com",),
        denied_domains=("docs.example.com",),
    )

    decision = policy.evaluate("https://docs.example.com/reference")

    assert decision.status == "denied"  # nosec B101
    assert decision.reason == "domain_denied"  # nosec B101


def test_exact_domain_rules_only_match_exact_hosts() -> None:
    policy = SourcePolicy(web_source_profile="local_first", preapproved_domains=("example.com",))

    exact = policy.evaluate("https://example.com/reference")
    prefixed = policy.evaluate("https://badexample.com/reference")
    subdomain = policy.evaluate("https://sub.example.com/reference")

    assert exact.status == "allowed"  # nosec B101
    assert prefixed.status == "approval_required"  # nosec B101
    assert subdomain.status == "approval_required"  # nosec B101


def test_wildcard_domain_rules_match_subdomains_but_not_apex() -> None:
    policy = SourcePolicy(web_source_profile="local_first", preapproved_domains=("*.example.com",))

    subdomain = policy.evaluate("https://docs.example.com/reference")
    apex = policy.evaluate("https://example.com/reference")

    assert subdomain.status == "allowed"  # nosec B101
    assert apex.status == "approval_required"  # nosec B101


@pytest.mark.parametrize(
    ("raw_url", "expected_status"),
    [
        ("https://example.com/docs/page", "allowed"),
        ("https://example.com/docs.evil", "approval_required"),
        ("https://example.com/docs%2Eevil", "approval_required"),
    ],
)
def test_url_prefix_matching_uses_decoded_path_segment_boundaries(
    raw_url: str,
    expected_status: str,
) -> None:
    policy = SourcePolicy(
        web_source_profile="locked_down",
        allowed_url_prefixes=("https://example.com/docs/",),
    )

    decision = policy.evaluate(raw_url)

    assert decision.status == expected_status  # nosec B101


def test_safe_argument_hash_changes_with_query_and_redacted_url_omits_query_details() -> None:
    policy = SourcePolicy(web_source_profile="online_capable", allow_arbitrary_public_domains=True)

    first = policy.evaluate("https://docs.example.com/reference?token=alpha#section")
    second = policy.evaluate("https://docs.example.com/reference?token=beta#section")

    assert first.safe_argument_hash != second.safe_argument_hash  # nosec B101
    assert first.redacted_url == "https://docs.example.com/reference"  # nosec B101
    assert "token" not in first.redacted_url  # nosec B101
    assert "alpha" not in first.redacted_url  # nosec B101
    assert "section" not in first.redacted_url  # nosec B101


def test_normalize_url_returns_safe_canonical_parts() -> None:
    normalized = normalize_url("HTTPS://Docs.Example.Com:443/reference/page?token=secret#fragment")

    assert normalized.scheme == "https"  # nosec B101
    assert normalized.host == "docs.example.com"  # nosec B101
    assert normalized.port is None  # nosec B101
    assert normalized.canonical_url == "https://docs.example.com/reference/page"  # nosec B101
    assert normalized.redacted_url == "https://docs.example.com/reference/page"  # nosec B101
    assert "token" not in normalized.redacted_url  # nosec B101
