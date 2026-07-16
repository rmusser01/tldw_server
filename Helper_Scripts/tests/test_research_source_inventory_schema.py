"""Schema checks for the frozen research-source inventory artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from Helper_Scripts.validate_research_source_inventory_schema import (
    INVENTORY_FORMAT_CHECKER,
    validate_document,
)
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = ROOT / "Docs" / "Design" / "research_source_inventory"
SCHEMA_PATH = INVENTORY_DIR / "research-source-inventory.schema.json"
MANIFEST_PATH = INVENTORY_DIR / "sourclip-research-sources-2026-07-13.json"
LEDGER_PATH = INVENTORY_DIR / "research-source-coverage-ledger-2026-07-13.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_inventory_documents_conform_to_exercised_schema() -> None:
    """Both checked-in documents validate against the shared schema bundle."""
    validator = Draft202012Validator(
        _load(SCHEMA_PATH),
        format_checker=INVENTORY_FORMAT_CHECKER,
    )

    assert list(validator.iter_errors(_load(MANIFEST_PATH))) == []
    assert list(validator.iter_errors(_load(LEDGER_PATH))) == []


def test_inventory_schema_rejects_unknown_top_level_fields() -> None:
    """The schema fails closed instead of silently accepting contract drift."""
    validator = Draft202012Validator(
        _load(SCHEMA_PATH),
        format_checker=INVENTORY_FORMAT_CHECKER,
    )
    ledger = copy.deepcopy(_load(LEDGER_PATH))
    ledger["unexpected"] = True

    errors = list(validator.iter_errors(ledger))

    assert any("unevaluated properties" in error.message.lower() for error in errors)


def test_authoritative_schema_checker_rejects_invalid_formats() -> None:
    """The CLI helper applies FormatChecker rather than accepting shape alone."""
    schema = _load(SCHEMA_PATH)
    manifest = copy.deepcopy(_load(MANIFEST_PATH))
    manifest["source"]["captured_at_utc"] = "2099-99-99T99:99:99Z"

    errors = validate_document(schema, manifest, "manifest")

    assert any("captured_at_utc" in error for error in errors)


def test_date_time_format_requires_strict_rfc3339_syntax() -> None:
    """RFC 3339 timezone, case, and leap-second forms are handled strictly."""
    valid_values = (
        "2026-07-13T12:00:00Z",
        "2026-07-13t12:00:00z",
        "2026-07-13T12:00:00.123+05:30",
        "2016-12-31T23:59:60Z",
        "2016-12-31T15:59:60-08:00",
        "2017-01-01T00:59:60+01:00",
    )
    invalid_values = (
        "20260713T120000+00:00",
        "2026-07-13T12:00:00",
        "2026-07-13T24:00:00Z",
        "2026-07-13T12:00:61Z",
        "2026-07-13T12:00:00+24:00",
        "٢٠٢٦-٠٧-١٣T١٢:٠٠:٠٠Z",
    )

    assert all(INVENTORY_FORMAT_CHECKER.conforms(value, "date-time") for value in valid_values)
    assert not any(INVENTORY_FORMAT_CHECKER.conforms(value, "date-time") for value in invalid_values)


def test_uri_format_rejects_whitespace_and_invalid_ports() -> None:
    """Absolute URIs require clean text and a valid optional port."""
    valid_values = (
        "https://example.test/path",
        "https://example.test:8443/path?query=yes#fragment",
        "https://example.test/%7Euser?q=a%20b",
    )
    invalid_values = (
        " https://example.test/path",
        "https://example.test/path ",
        "https://example.test /path",
        "https://example.test:not-a-port/path",
        "https://example.test:99999/path",
        "https://example.test/%ZZ",
        "https://exámple.test/path",
        "https://example.test/\x01",
        "https://example.test/{bad}",
        "https://example.test/[bad]",
        "https://example.test/a#b#c",
    )

    assert all(INVENTORY_FORMAT_CHECKER.conforms(value, "uri") for value in valid_values)
    assert not any(INVENTORY_FORMAT_CHECKER.conforms(value, "uri") for value in invalid_values)


def test_certification_artifacts_are_typed_and_substantive() -> None:
    """Empty hash targets cannot masquerade as fixture certification evidence."""
    schema = _load(SCHEMA_PATH)
    artifact = {
        "schema_version": "research-source-certification-artifact.v1",
        "artifact_type": "fixture",
        "route_candidate_id": "example_direct",
        "canonical_target": "example_source",
        "surface": "standalone_search",
        "route_candidate_sha256": "a" * 64,
        "route_policy_sha256": "b" * 64,
        "catalog_version": "research-discovery-v1",
        "policy_version": "discovery-egress-v1",
        "observed_at_utc": "2026-07-13T12:00:00Z",
        "sanitized": True,
        "outcome": "passed",
        "details": {
            "test_command": "python -m pytest tests/example.py -q",
            "test_count": 4,
            "fixture_cases": [
                "success",
                "valid_empty",
                "malformed",
                "partial_failure",
            ],
        },
    }

    assert validate_document(schema, artifact, "artifact") == []
    errors = validate_document(schema, {}, "artifact")
    assert any("schema_version" in error for error in errors)


def test_live_certification_requires_non_empty_gateway_policy_evidence() -> None:
    """A live check cannot certify an empty or gateway-bypassing request."""
    schema = _load(SCHEMA_PATH)
    artifact = {
        "schema_version": "research-source-certification-artifact.v1",
        "artifact_type": "live",
        "route_candidate_id": "example_direct",
        "canonical_target": "example_source",
        "surface": "standalone_search",
        "route_candidate_sha256": "a" * 64,
        "route_policy_sha256": "b" * 64,
        "catalog_version": "research-discovery-v1",
        "policy_version": "discovery-egress-v1",
        "observed_at_utc": "2026-07-13T12:00:00Z",
        "sanitized": True,
        "outcome": "passed",
        "details": {
            "checked_endpoint": "https://example.test/api/search",
            "request_method": "GET",
            "request_count": 1,
            "result_count": 1,
            "transport_origins": ["https://example.test"],
            "gateway_attested": True,
            "credential_mode": "none",
            "result_link_dereference_count": 0,
        },
    }

    assert validate_document(schema, artifact, "artifact") == []
    artifact["details"]["result_count"] = 0
    artifact["details"]["gateway_attested"] = False

    errors = validate_document(schema, artifact, "artifact")

    assert any("result_count" in error for error in errors)
    assert any("gateway_attested" in error for error in errors)


def test_reviewed_exclusion_schema_allows_at_most_one_canonical_target() -> None:
    """An exclusion cannot smuggle several product targets through contract freeze."""
    schema = _load(SCHEMA_PATH)
    ledger = copy.deepcopy(_load(LEDGER_PATH))
    row = next(row for row in ledger["rows"] if row["resolution"] == "credentialed_out_of_scope")
    row["canonical_targets"] = ["first_target", "second_target"]

    errors = validate_document(schema, ledger, "ledger")

    assert any("canonical_targets" in error and "too long" in error for error in errors)


def test_route_schema_rejects_aggregator_claiming_native_attribution() -> None:
    """Aggregator routes must retain provider-constrained provenance semantics."""
    schema = _load(SCHEMA_PATH)
    ledger = copy.deepcopy(_load(LEDGER_PATH))
    row = next(row for row in ledger["rows"] if row["resolution"] == "mapped")
    route = row["route_candidates"][0]
    route["route_kind"] = "aggregator"
    route["source_constraint"] = "native_corpus"
    route["attribution_basis"] = "native_response"

    errors = validate_document(schema, ledger, "ledger")

    assert any("provider_source_filter" in error or "provider_domain_filter" in error for error in errors)


def test_route_schema_requires_typed_aggregator_source_predicate() -> None:
    """Aggregator mappings identify the exact provider field and filter value."""
    schema = _load(SCHEMA_PATH)
    ledger = copy.deepcopy(_load(LEDGER_PATH))
    row = next(
        row for row in ledger["rows"] if any(route["route_kind"] == "aggregator" for route in row["route_candidates"])
    )
    route = next(route for route in row["route_candidates"] if route["route_kind"] == "aggregator")
    route["source_constraint_predicate"] = None

    errors = validate_document(schema, ledger, "ledger")

    assert any("source_constraint_predicate" in error and "object" in error for error in errors)
