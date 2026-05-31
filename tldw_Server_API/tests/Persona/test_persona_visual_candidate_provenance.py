"""Tests for Persona Visual candidate provenance normalization safety rules."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Persona.visual_candidate_provenance import (
    normalize_persona_visual_candidate_provenance,
)


pytestmark = pytest.mark.unit


def _normalize_recipe_provenance(recipe: dict[str, object]) -> dict[str, object]:
    provenance = normalize_persona_visual_candidate_provenance(
        {
            "generation_mode": "recipe_backed",
            "request_id": "request-1",
            "recipe": recipe,
        }
    )
    return provenance["recipe"]


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "Bearer:\tabcdefghijkl",
        "authorization = Bearer abcdefghijkl",
        "api key: abcdefghijkl",
        "x-api-key=abcdefghijkl",
        "client-secret: abcdefghijkl",
        "sk-live-abcdefghijkl",
    ],
)
def test_normalize_provenance_redacts_common_secret_variants(
    unsafe_value: str,
) -> None:
    recipe = _normalize_recipe_provenance({"neutral_anchor": unsafe_value})

    assert recipe["neutral_anchor"] == "[redacted]"


def test_normalize_provenance_keeps_legitimate_secret_and_token_words() -> None:
    recipe = _normalize_recipe_provenance(
        {
            "identity_brief": "secret agent character holding a brass token and authorization scroll",
            "review_checks": [
                "token charm stays visible",
                "secret-agent pose reads clearly",
                "standard Bearer character stays visible",
                "api key display label is omitted",
            ],
            "static_sheet": "pose guide uses \\alpha label markup",
        }
    )

    assert recipe["identity_brief"] == (
        "secret agent character holding a brass token and authorization scroll"
    )
    assert recipe["review_checks"] == [
        "token charm stays visible",
        "secret-agent pose reads clearly",
        "standard Bearer character stays visible",
        "api key display label is omitted",
    ]
    assert recipe["static_sheet"] == "pose guide uses \\alpha label markup"


@pytest.mark.parametrize(
    "path_value",
    [
        "/Users/macbook-dev/private/reference.png",
        "/home/persona/reference.png",
        "C:\\Users\\persona\\reference.png",
        "\\\\fileserver\\persona\\reference.png",
    ],
)
def test_normalize_provenance_redacts_specific_path_shapes(path_value: str) -> None:
    recipe = _normalize_recipe_provenance({"neutral_anchor": path_value})

    assert recipe["neutral_anchor"] == "[redacted]"
