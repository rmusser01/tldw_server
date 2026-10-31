"""Compatibility tests for centralized prompt-improvement exceptions."""

import pytest

from tldw_Server_API.app.core import exceptions
from tldw_Server_API.app.core.Prompt_Management import (
    prompt_improvement,
    prompt_improvement_dispatch,
)

pytestmark = pytest.mark.unit


def test_prompt_improvement_errors_are_reexported_with_existing_behavior():
    assert (
        prompt_improvement.PromptImprovementError
        is exceptions.PromptImprovementError
    )
    assert (
        prompt_improvement_dispatch.PromptImprovementDispatchError
        is exceptions.PromptImprovementDispatchError
    )

    domain_error = exceptions.PromptImprovementError("invalid_input", "Invalid draft")
    assert domain_error.code == "invalid_input"
    assert str(domain_error) == "Invalid draft"

    dispatch_error = exceptions.PromptImprovementDispatchError(
        "provider_rate_limited",
        internal_detail="must not escape",
        retryable=1,
        retry_after_seconds=100_000,
    )
    assert dispatch_error.code == "provider_rate_limited"
    assert str(dispatch_error) == "The active provider is temporarily rate limited."
    assert dispatch_error.retryable is True
    assert dispatch_error.retry_after_seconds == 86_400

    unknown_error = exceptions.PromptImprovementDispatchError(
        "not_public",
        retry_after_seconds=-1,
    )
    assert unknown_error.code == "internal_error"
    assert str(unknown_error) == "The prompt improvement request could not be completed."
    assert unknown_error.retry_after_seconds is None
