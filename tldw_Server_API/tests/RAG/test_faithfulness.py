"""
Tests for the Faithfulness Evaluator module.

These tests cover:
- ClaimVerification and FaithfulnessResult dataclasses and their to_dict() methods
- FaithfulnessEvaluator.evaluate_detailed() with various scenarios:
  - All claims supported
  - Some claims unsupported
  - Empty response (score=1.0)
  - Empty context (score=0.0)
  - LLM failure during claim extraction (score=0.0)
  - max_claims truncation
- FaithfulnessEvaluator.evaluate() convenience method
- _parse_json_array helper with valid, malformed, and embedded JSON
- _parse_json_object helper with valid, malformed, embedded, and nested-brace JSON
"""

import json
import pytest
from typing import Any

from tldw_Server_API.app.core.RAG.rag_service import faithfulness as faithfulness_mod
from tldw_Server_API.app.core.RAG.rag_service.faithfulness import (
    ClaimVerification,
    FaithfulnessResult,
    FaithfulnessEvaluator,
    LLMCallable,
    _parse_json_array,
    _parse_json_object,
)


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns controlled responses based on prompt content.

    Attributes:
        claim_extraction_response: JSON string returned for claim extraction prompts.
        verification_responses: Mapping of claim substring -> JSON verification response.
        default_verification: Default verification JSON when no mapping matches.
        call_log: List of prompts received, useful for assertions.
    """

    def __init__(
        self,
        claim_extraction_response: str = "[]",
        verification_responses: dict[str, str] | None = None,
        default_verification: str | None = None,
    ) -> None:
        self.claim_extraction_response = claim_extraction_response
        self.verification_responses = verification_responses or {}
        self.default_verification = default_verification or json.dumps(
            {"supported": True, "reasoning": "Supported by context."}
        )
        self.call_log: list[str] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_log.append(prompt)
        # Claim extraction prompts contain "Extract all factual claims"
        if "Extract all factual claims" in prompt:
            return self.claim_extraction_response
        # Verification prompts contain "Verify if the following claim"
        if "Verify if the following claim" in prompt:
            for claim_fragment, response in self.verification_responses.items():
                if claim_fragment in prompt:
                    return response
            return self.default_verification
        return ""


class FailingLLM:
    """Mock LLM that always raises an exception."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("LLM service unavailable")


class FailOnVerificationLLM:
    """Mock LLM that succeeds on claim extraction but fails on verification."""

    def __init__(self, claims_json: str) -> None:
        self.claims_json = claims_json

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if "Extract all factual claims" in prompt:
            return self.claims_json
        raise RuntimeError("Verification service unavailable")


class SensitiveExtractionFailureLLM:
    """Mock LLM that leaks sensitive details in extraction exceptions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError(
            "extract failed with sk-test-secret at /private/tmp/faithfulness.env"
        )


class SensitiveVerificationFailureLLM:
    """Mock LLM that leaks sensitive details in verification exceptions."""

    def __init__(self, claims_json: str) -> None:
        self.claims_json = claims_json

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if "Extract all factual claims" in prompt:
            return self.claims_json
        raise RuntimeError(
            "verify failed with sk-test-secret at /private/tmp/faithfulness.env"
        )


class WarningLoggerStub:
    """Capture warning messages rendered by loguru-style calls."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        if args or kwargs:
            self.warnings.append(message.format(*args, **kwargs))
        else:
            self.warnings.append(message)


def _assert_no_sensitive_fallback_leak(text: str) -> None:
    assert "sk-test-secret" not in text
    assert "/private/tmp/faithfulness.env" not in text
    assert "extract failed" not in text
    assert "verify failed" not in text


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClaimVerification:
    """Tests for the ClaimVerification frozen dataclass."""

    def test_create_supported_claim(self):
        cv = ClaimVerification(
            claim="Paris is the capital of France",
            supported=True,
            reasoning="Directly stated in context.",
        )
        assert cv.claim == "Paris is the capital of France"
        assert cv.supported is True
        assert cv.reasoning == "Directly stated in context."

    def test_create_unsupported_claim(self):
        cv = ClaimVerification(
            claim="The moon is made of cheese",
            supported=False,
            reasoning="Not mentioned in context.",
        )
        assert cv.supported is False

    def test_to_dict(self):
        cv = ClaimVerification(
            claim="Water boils at 100C",
            supported=True,
            reasoning="Confirmed.",
        )
        d = cv.to_dict()
        assert d == {
            "claim": "Water boils at 100C",
            "supported": True,
            "reasoning": "Confirmed.",
        }

    def test_frozen(self):
        cv = ClaimVerification(claim="test", supported=True, reasoning="ok")
        with pytest.raises(AttributeError):
            cv.claim = "modified"  # type: ignore[misc]


@pytest.mark.unit
class TestFaithfulnessResult:
    """Tests for the FaithfulnessResult frozen dataclass."""

    def test_create_result(self):
        claims = [
            ClaimVerification(claim="A", supported=True, reasoning="ok"),
            ClaimVerification(claim="B", supported=False, reasoning="nope"),
        ]
        result = FaithfulnessResult(score=0.5, claims=claims, reasoning="Mixed.")
        assert result.score == 0.5
        assert len(result.claims) == 2
        assert result.reasoning == "Mixed."

    def test_to_dict_counts(self):
        claims = [
            ClaimVerification(claim="A", supported=True, reasoning="ok"),
            ClaimVerification(claim="B", supported=True, reasoning="ok"),
            ClaimVerification(claim="C", supported=False, reasoning="nope"),
        ]
        result = FaithfulnessResult(score=2 / 3, claims=claims, reasoning="Mostly good.")
        d = result.to_dict()

        assert d["faithfulness_score"] == pytest.approx(2 / 3)
        assert d["total_claims"] == 3
        assert d["supported_claims"] == 2
        assert d["unsupported_claims"] == 1
        assert d["reasoning"] == "Mostly good."
        assert len(d["claim_verifications"]) == 3

    def test_to_dict_empty_claims(self):
        result = FaithfulnessResult(score=1.0, claims=[], reasoning="No claims.")
        d = result.to_dict()
        assert d["total_claims"] == 0
        assert d["supported_claims"] == 0
        assert d["unsupported_claims"] == 0
        assert d["claim_verifications"] == []

    def test_frozen(self):
        result = FaithfulnessResult(score=1.0, claims=[], reasoning="ok")
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JSON parsing helper tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseJsonArray:
    """Tests for the _parse_json_array helper function."""

    def test_valid_json_array(self):
        text = '["claim one", "claim two", "claim three"]'
        result = _parse_json_array(text)
        assert result == ["claim one", "claim two", "claim three"]

    def test_valid_json_with_whitespace(self):
        text = '  \n  ["a", "b"]  \n  '
        result = _parse_json_array(text)
        assert result == ["a", "b"]

    def test_empty_array(self):
        result = _parse_json_array("[]")
        assert result == []

    def test_malformed_text_returns_empty(self):
        result = _parse_json_array("this is not json at all")
        assert result == []

    def test_empty_string_returns_empty(self):
        result = _parse_json_array("")
        assert result == []

    def test_embedded_json_array_with_trailing_text(self):
        """Trailing text after the JSON array is tolerated thanks to raw_decode."""
        text = 'Here are the claims: ["Paris is in France", "Water is wet"] and that is all.'
        result = _parse_json_array(text)
        assert result == ["Paris is in France", "Water is wet"]

    def test_embedded_json_array_at_end(self):
        """When the JSON array is at the very end of the string (no trailing
        text after the closing bracket), the scanner succeeds."""
        text = 'Here are the claims: ["Paris is in France", "Water is wet"]'
        result = _parse_json_array(text)
        assert result == ["Paris is in France", "Water is wet"]

    def test_json_with_preamble(self):
        text = 'Sure, here is the JSON:\n["claim A", "claim B"]'
        result = _parse_json_array(text)
        assert result == ["claim A", "claim B"]

    def test_fenced_array_with_think_tags(self):
        text = (
            "<think>internal reasoning</think>\n"
            "```json\n"
            '["claim A", "claim B"]\n'
            "```"
        )
        result = _parse_json_array(text)
        assert result == ["claim A", "claim B"]

    def test_not_an_array_returns_empty(self):
        """If the JSON is valid but not an array, return empty list."""
        result = _parse_json_array('{"key": "value"}')
        assert result == []

    def test_nested_brackets(self):
        """Array with strings containing bracket characters."""
        text = '["array [0] access", "range [1,2]"]'
        result = _parse_json_array(text)
        assert result == ["array [0] access", "range [1,2]"]

    def test_multiple_arrays_returns_first_valid(self):
        """The scanner tries each '[' from left to right and raw_decode
        stops at the end of valid JSON, so the first array is returned.
        """
        text = '["first"] some text ["second"]'
        result = _parse_json_array(text)
        assert result == ["first"]

    def test_multiple_arrays_first_at_end(self):
        """When the first array sits at the end of the string, it is returned."""
        text = 'prefix ["first"]'
        result = _parse_json_array(text)
        assert result == ["first"]


@pytest.mark.unit
class TestParseJsonObject:
    """Tests for the _parse_json_object helper function."""

    def test_valid_json_object(self):
        text = '{"supported": true, "reasoning": "Found in context."}'
        result = _parse_json_object(text)
        assert result == {"supported": True, "reasoning": "Found in context."}

    def test_valid_json_with_whitespace(self):
        text = '  \n  {"supported": false, "reasoning": "nope"}  \n  '
        result = _parse_json_object(text)
        assert result["supported"] is False

    def test_empty_object(self):
        result = _parse_json_object("{}")
        assert result == {}

    def test_malformed_text_returns_empty(self):
        result = _parse_json_object("not json")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = _parse_json_object("")
        assert result == {}

    def test_embedded_json_object_with_trailing_text(self):
        """Trailing text after the closing brace is tolerated thanks to raw_decode."""
        text = 'Here is my answer: {"supported": true, "reasoning": "yes"} hope that helps!'
        result = _parse_json_object(text)
        assert result == {"supported": True, "reasoning": "yes"}

    def test_embedded_json_object_at_end(self):
        """When the JSON object sits at the very end, the scanner succeeds."""
        text = 'Here is my answer: {"supported": true, "reasoning": "yes"}'
        result = _parse_json_object(text)
        assert result["supported"] is True
        assert result["reasoning"] == "yes"

    def test_json_with_preamble(self):
        text = 'Sure, here is the verification:\n{"supported": false, "reasoning": "not found"}'
        result = _parse_json_object(text)
        assert result["supported"] is False

    def test_fenced_object_with_think_tags(self):
        text = (
            "<think>internal reasoning</think>\n"
            "```json\n"
            '{"supported": true, "reasoning": "in context"}\n'
            "```"
        )
        result = _parse_json_object(text)
        assert result["supported"] is True
        assert result["reasoning"] == "in context"

    def test_not_an_object_returns_empty(self):
        """If the JSON is valid but not a dict, return empty dict."""
        result = _parse_json_object('["a", "b"]')
        assert result == {}

    def test_multiple_braces_returns_first_valid(self):
        """When text contains multiple opening braces, find the first valid object."""
        text = 'prefix {invalid json here {"supported": true, "reasoning": "ok"}'
        result = _parse_json_object(text)
        assert result["supported"] is True

    def test_nested_braces_in_string_value(self):
        """JSON object with brace characters inside string values."""
        text = '{"supported": true, "reasoning": "function() { return x; }"}'
        result = _parse_json_object(text)
        assert result["supported"] is True
        assert "function()" in result["reasoning"]


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestFaithfulnessEvaluatorDetailed:
    """Tests for FaithfulnessEvaluator.evaluate_detailed()."""

    async def test_all_claims_supported(self):
        """When all extracted claims are supported, score should be 1.0."""
        llm = MockLLM(
            claim_extraction_response='["Paris is in France", "Berlin is in Germany"]',
            default_verification=json.dumps(
                {"supported": True, "reasoning": "Confirmed by context."}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Paris is in France. Berlin is in Germany.",
            context="Paris is the capital of France. Berlin is the capital of Germany.",
        )

        assert result.score == 1.0
        assert len(result.claims) == 2
        assert all(c.supported for c in result.claims)
        assert "All claims" in result.reasoning

    async def test_some_claims_unsupported(self):
        """When some claims are unsupported, score should reflect the ratio."""
        llm = MockLLM(
            claim_extraction_response='["Water boils at 100C", "The sun is cold", "Ice is frozen water"]',
            verification_responses={
                "Water boils at 100C": json.dumps(
                    {"supported": True, "reasoning": "Confirmed."}
                ),
                "The sun is cold": json.dumps(
                    {"supported": False, "reasoning": "Context says the sun is hot."}
                ),
                "Ice is frozen water": json.dumps(
                    {"supported": True, "reasoning": "Confirmed."}
                ),
            },
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Water boils at 100C. The sun is cold. Ice is frozen water.",
            context="Water boils at 100 degrees Celsius. The sun is extremely hot. Ice is frozen water.",
        )

        assert result.score == pytest.approx(2 / 3)
        assert len(result.claims) == 3
        supported = [c for c in result.claims if c.supported]
        unsupported = [c for c in result.claims if not c.supported]
        assert len(supported) == 2
        assert len(unsupported) == 1
        assert "2 out of 3" in result.reasoning

    async def test_no_claims_supported(self):
        """When no claims are supported, score should be 0.0."""
        llm = MockLLM(
            claim_extraction_response='["Claim A", "Claim B"]',
            default_verification=json.dumps(
                {"supported": False, "reasoning": "Not in context."}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Claim A. Claim B.",
            context="Completely unrelated context.",
        )

        assert result.score == 0.0
        assert len(result.claims) == 2
        assert all(not c.supported for c in result.claims)
        assert "None of the claims" in result.reasoning

    async def test_empty_response_returns_perfect_score(self):
        """An empty response has no claims, so it should be perfectly faithful."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="",
            context="Some context here.",
        )

        assert result.score == 1.0
        assert result.claims == []
        assert "Empty response" in result.reasoning
        # LLM should not have been called
        assert len(llm.call_log) == 0

    async def test_whitespace_only_response_returns_perfect_score(self):
        """A whitespace-only response should also be treated as empty."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="   \n\t  ",
            context="Some context here.",
        )

        assert result.score == 1.0
        assert result.claims == []
        assert len(llm.call_log) == 0

    async def test_empty_context_returns_zero_score(self):
        """No context means nothing to verify against; score should be 0.0."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some response with claims.",
            context="",
        )

        assert result.score == 0.0
        assert result.claims == []
        assert "No context provided" in result.reasoning
        assert len(llm.call_log) == 0

    async def test_whitespace_only_context_returns_zero_score(self):
        """Whitespace-only context should be treated as empty."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some response.",
            context="   \n\t  ",
        )

        assert result.score == 0.0
        assert result.claims == []

    async def test_llm_extraction_failure_returns_zero_score(self):
        """When the LLM fails during claim extraction, score should be 0.0."""
        llm = FailingLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some response.",
            context="Some context.",
        )

        assert result.score == 0.0
        assert result.claims == []
        assert "Claim extraction failed" in result.reasoning

    async def test_llm_extraction_failure_fallback_is_sanitized(self, monkeypatch):
        """Extraction fallback must not leak raw exception details."""
        logger_stub = WarningLoggerStub()
        monkeypatch.setattr(faithfulness_mod, "logger", logger_stub)
        llm = SensitiveExtractionFailureLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some response.",
            context="Some context.",
        )

        assert result.score == 0.0
        assert result.claims == []
        assert result.reasoning == "Claim extraction failed."
        rendered_logs = "\n".join(logger_stub.warnings)
        assert "Faithfulness: claim extraction failed" in rendered_logs
        _assert_no_sensitive_fallback_leak(result.reasoning)
        _assert_no_sensitive_fallback_leak(rendered_logs)

    async def test_llm_verification_failure_marks_claim_unsupported(self):
        """When verification fails for a claim, it should be marked unsupported."""
        llm = FailOnVerificationLLM(
            claims_json='["Claim that will fail verification"]'
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Claim that will fail verification.",
            context="Some context.",
        )

        assert result.score == 0.0
        assert len(result.claims) == 1
        assert result.claims[0].supported is False
        assert "Verification failed" in result.claims[0].reasoning

    async def test_llm_verification_failure_fallback_is_sanitized(self, monkeypatch):
        """Verification fallback must preserve status without leaking exceptions."""
        logger_stub = WarningLoggerStub()
        monkeypatch.setattr(faithfulness_mod, "logger", logger_stub)
        llm = SensitiveVerificationFailureLLM(
            claims_json='["Claim that will fail verification"]'
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Claim that will fail verification.",
            context="Some context.",
        )

        assert result.score == 0.0
        assert len(result.claims) == 1
        assert result.claims[0].claim == "Claim that will fail verification"
        assert result.claims[0].supported is False
        assert result.claims[0].reasoning == "Verification failed."
        assert result.reasoning == "None of the claims in the response are supported by the context."
        rendered_logs = "\n".join(logger_stub.warnings)
        assert "Faithfulness: claim verification failed for 'Claim that will fail verification...'" in rendered_logs
        _assert_no_sensitive_fallback_leak(result.claims[0].reasoning)
        _assert_no_sensitive_fallback_leak(rendered_logs)

    async def test_no_verifiable_claims_returns_perfect_score(self):
        """When extraction returns no claims, score should be 1.0."""
        llm = MockLLM(claim_extraction_response="[]")
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some vague response without factual claims.",
            context="Some context.",
        )

        assert result.score == 1.0
        assert result.claims == []
        assert "No verifiable claims" in result.reasoning

    async def test_max_claims_truncation(self):
        """When more claims are extracted than max_claims, truncate to max_claims."""
        many_claims = [f"Claim number {i}" for i in range(30)]
        llm = MockLLM(
            claim_extraction_response=json.dumps(many_claims),
            default_verification=json.dumps(
                {"supported": True, "reasoning": "ok"}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm, max_claims=5)

        result = await evaluator.evaluate_detailed(
            response="Response with many claims.",
            context="Comprehensive context.",
        )

        assert len(result.claims) == 5
        assert result.score == 1.0
        # Verify only the first 5 claims were used
        for i, cv in enumerate(result.claims):
            assert cv.claim == f"Claim number {i}"

    async def test_max_claims_default_is_25(self):
        """Default max_claims should be 25."""
        many_claims = [f"Claim {i}" for i in range(30)]
        llm = MockLLM(
            claim_extraction_response=json.dumps(many_claims),
            default_verification=json.dumps(
                {"supported": True, "reasoning": "ok"}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Response with many claims.",
            context="Context.",
        )

        assert len(result.claims) == 25

    async def test_claims_with_empty_strings_filtered(self):
        """Empty string claims should be filtered out by _extract_claims."""
        llm = MockLLM(
            claim_extraction_response='["Valid claim", "", "Another claim"]',
            default_verification=json.dumps(
                {"supported": True, "reasoning": "ok"}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Response.",
            context="Context.",
        )

        # Empty strings should be filtered out
        assert len(result.claims) == 2
        assert result.claims[0].claim == "Valid claim"
        assert result.claims[1].claim == "Another claim"

    async def test_verification_response_missing_fields(self):
        """When verification JSON lacks fields, defaults should be used."""
        llm = MockLLM(
            claim_extraction_response='["Some claim"]',
            default_verification="{}",  # No supported or reasoning fields
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Some claim.",
            context="Some context.",
        )

        assert len(result.claims) == 1
        # Missing "supported" defaults to False
        assert result.claims[0].supported is False
        # Missing "reasoning" defaults to "No reasoning provided."
        assert result.claims[0].reasoning == "No reasoning provided."

    async def test_to_dict_integration(self):
        """Verify the full to_dict output structure after evaluation."""
        llm = MockLLM(
            claim_extraction_response='["Claim X", "Claim Y"]',
            verification_responses={
                "Claim X": json.dumps({"supported": True, "reasoning": "Yes"}),
                "Claim Y": json.dumps({"supported": False, "reasoning": "No"}),
            },
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate_detailed(
            response="Claim X. Claim Y.",
            context="Context supports X only.",
        )
        d = result.to_dict()

        assert d["faithfulness_score"] == 0.5
        assert d["total_claims"] == 2
        assert d["supported_claims"] == 1
        assert d["unsupported_claims"] == 1
        assert len(d["claim_verifications"]) == 2
        assert d["claim_verifications"][0]["claim"] == "Claim X"
        assert d["claim_verifications"][0]["supported"] is True
        assert d["claim_verifications"][1]["claim"] == "Claim Y"
        assert d["claim_verifications"][1]["supported"] is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestFaithfulnessEvaluatorEvaluate:
    """Tests for the convenience evaluate() method that returns a float score."""

    async def test_evaluate_returns_float_score(self):
        """evaluate() should return only the float score from evaluate_detailed()."""
        llm = MockLLM(
            claim_extraction_response='["A", "B"]',
            default_verification=json.dumps(
                {"supported": True, "reasoning": "ok"}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        score = await evaluator.evaluate(
            response="A. B.",
            context="Context.",
        )

        assert isinstance(score, float)
        assert score == 1.0

    async def test_evaluate_passes_through_empty_response(self):
        """evaluate() with empty response returns 1.0."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        score = await evaluator.evaluate(response="", context="Context.")
        assert score == 1.0

    async def test_evaluate_passes_through_empty_context(self):
        """evaluate() with empty context returns 0.0."""
        llm = MockLLM()
        evaluator = FaithfulnessEvaluator(llm=llm)

        score = await evaluator.evaluate(response="Something.", context="")
        assert score == 0.0

    async def test_evaluate_accepts_optional_query(self):
        """evaluate() should accept query parameter for protocol compatibility."""
        llm = MockLLM(
            claim_extraction_response='["Claim"]',
            default_verification=json.dumps(
                {"supported": True, "reasoning": "ok"}
            ),
        )
        evaluator = FaithfulnessEvaluator(llm=llm)

        score = await evaluator.evaluate(
            response="Claim.",
            context="Context.",
            query="What is the claim?",
        )

        assert score == 1.0


# ---------------------------------------------------------------------------
# LLMCallable re-export test
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMCallableReexport:
    """Verify that LLMCallable is re-exported from faithfulness for backwards compat."""

    def test_llmcallable_is_protocol(self):
        """LLMCallable should be the same object as LLMProtocol from rag_protocols."""
        from tldw_Server_API.app.core.RAG.rag_service.rag_protocols import LLMProtocol
        assert LLMCallable is LLMProtocol

    def test_mock_llm_satisfies_protocol(self):
        """Our MockLLM should be recognized as implementing LLMCallable."""
        mock = MockLLM()
        assert isinstance(mock, LLMCallable)
