from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import (
    embeddings_v5_production_enhanced as emb_mod,
)


pytestmark = pytest.mark.unit


def test_embeddings_build_user_metadata_skips_with_testing_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESTING", "y")

    assert emb_mod._build_user_metadata(SimpleNamespace(id="u-1")) is None


def test_embeddings_policy_enforcement_accepts_testing_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDINGS_ENFORCE_POLICY", raising=False)
    monkeypatch.setenv("TESTING", "y")

    assert emb_mod._should_enforce_policy() is True
