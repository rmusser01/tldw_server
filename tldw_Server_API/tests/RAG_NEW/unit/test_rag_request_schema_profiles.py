import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", ["fast", "balanced", "accuracy"])
def test_rag_profile_accepts_switchable_values(profile: str) -> None:
    req = UnifiedRAGRequest(query="q", rag_profile=profile)
    assert req.rag_profile == profile


def test_rag_profile_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        UnifiedRAGRequest(query="q", rag_profile="production")


@pytest.mark.parametrize("budget", [2200, 4096, 8192])
def test_max_generation_tokens_preserves_budgets_through_8192(budget: int) -> None:
    assert UnifiedRAGRequest(query="q", max_generation_tokens=budget).max_generation_tokens == budget


def test_max_generation_tokens_rejects_8193() -> None:
    with pytest.raises(ValidationError):
        UnifiedRAGRequest(query="q", max_generation_tokens=8193)
