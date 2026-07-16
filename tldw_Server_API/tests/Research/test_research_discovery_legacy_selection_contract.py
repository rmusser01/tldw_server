"""Golden characterization for the legacy research-discovery selectors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.research_discovery_schemas import (
    ResearchDiscoverySearchRequest,
)
from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryValidationError,
)
from tldw_Server_API.app.core.Research.discovery.service import (
    DEFAULT_SOURCE_CATEGORIES,
    ResearchDiscoveryService,
    _normalize_string_sequence,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "Docs" / "Design" / "research_source_inventory" / "research-discovery-legacy-selection-v1.json"
EXCEPTIONS = {
    "ResearchDiscoveryBadRequestError": (ResearchDiscoveryBadRequestError, 400),
    "ResearchDiscoveryValidationError": (ResearchDiscoveryValidationError, 422),
}


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _contract()["cases"], ids=lambda case: case["id"])
def test_legacy_selection_case_matches_frozen_contract(case: dict) -> None:
    """Each frozen request retains its current validation and resolution behavior."""
    request_data = {"query": "characterization query", **case["request_fields"]}
    if case["schema_status"] == "rejected":
        with pytest.raises(ValidationError):
            ResearchDiscoverySearchRequest.model_validate(request_data)
        assert case["http_status"] == 422
        return

    request = ResearchDiscoverySearchRequest.model_validate(request_data)
    source_ids = _normalize_string_sequence(request.source_ids)
    categories = _normalize_string_sequence(request.categories)
    defaulted_categories: list[str] = []
    if not source_ids and not categories:
        defaulted_categories = list(DEFAULT_SOURCE_CATEGORIES)
        categories = list(DEFAULT_SOURCE_CATEGORIES)

    expected_error = case.get("selection_error")
    service = ResearchDiscoveryService()
    if expected_error:
        exception_type, http_status = EXCEPTIONS[expected_error["exception"]]
        with pytest.raises(exception_type, match=f"^{expected_error['detail']}$"):
            service._resolve_sources(
                source_ids=source_ids,
                categories=categories,
                max_sources=case.get("service_max_sources"),
            )
        assert expected_error["http_status"] == http_status
        return

    selected = service._resolve_sources(
        source_ids=source_ids,
        categories=categories,
        max_sources=case.get("service_max_sources"),
    )

    assert source_ids == case["normalized_source_ids"]
    assert categories == case["normalized_categories"]
    assert defaulted_categories == case["defaulted_categories"]
    assert [source.source_id for source in selected] == case["resolved_source_ids"]


def test_legacy_selection_contract_pins_current_limits_and_catalog_version() -> None:
    """The golden fixture fails visibly if legacy caps or defaults are reinterpreted."""
    contract = _contract()
    service = ResearchDiscoveryService()

    assert contract["catalog_version"] == service._catalog.catalog_version
    assert contract["resolved_source_limit"] == service._catalog.max_selected_sources
    assert contract["raw_list_limit"] == 20
    assert contract["default_category"] == DEFAULT_SOURCE_CATEGORIES[0]
