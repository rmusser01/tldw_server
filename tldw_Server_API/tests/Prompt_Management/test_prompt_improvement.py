import json
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_improvement import (
    MAX_CANDIDATE_CHARS,
    MAX_DRAFT_CHARS,
    MAX_FINDINGS,
    MAX_PROTECTED_TOKEN_CHARS,
    MAX_PROTECTED_TOKEN_OCCURRENCES,
    MAX_PROTECTED_TOKEN_TOTAL_CHARS,
    MAX_PROTECTED_TOKENS,
    META_PROMPT_VERSION,
    PromptImprovementError,
    PromptImprovementInput,
    PromptProtectedToken,
    improve_prompt,
    validate_protected_tokens,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prompt_improvement_cases.json"


def _structured_output(
    improved_text: str,
    *,
    target: str | None = None,
    findings: list[object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "status": "improved",
        "improved_text": improved_text,
        "findings": findings or [],
    }
    if target is not None:
        payload["target"] = target
    return json.dumps(payload, ensure_ascii=False)


async def _improve_candidate(
    draft: str,
    candidate: str,
    *,
    target: str = "user_message",
):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output(candidate)

    return await improve_prompt(
        PromptImprovementInput(target=target, text=draft),
        generate=fake_generate,
    )


@pytest.mark.asyncio
async def test_structured_output_is_auto_apply_eligible_and_messages_are_isolated():
    draft = "summarize {{topic}}"

    async def fake_generate(messages: list[dict[str, str]]) -> str:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert META_PROMPT_VERSION in messages[0]["content"]
        envelope = json.loads(messages[1]["content"])
        assert envelope == {"target": "user_message", "draft": draft}
        serialized = json.dumps(messages)
        for forbidden in (
            "COUNTERPART",
            "HISTORY_SENTINEL",
            "ATTACHMENT_SENTINEL",
            "RAG_SENTINEL",
            "TOOL_SENTINEL",
            "METADATA_SENTINEL",
        ):
            assert forbidden not in serialized
        return _structured_output(
            "Summarize {{topic}} in three bullets.",
            findings=[
                {
                    "category": "clarity",
                    "issue": "The requested form was unclear.",
                    "change": "Specified three bullets.",
                }
            ],
        )

    result = await improve_prompt(
        PromptImprovementInput(target="user_message", text=draft),
        generate=fake_generate,
    )

    assert result.status == "improved"
    assert result.improved_text == "Summarize {{topic}} in three bullets."
    assert result.review_required is False
    assert result.warnings == ()
    assert result.meta_prompt_version == META_PROMPT_VERSION


@pytest.mark.asyncio
async def test_whole_response_json_fence_is_parsed_as_structured_output():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return "```json\n" + _structured_output("Make the answer concise.") + "\n```"

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="be concise"),
        generate=fake_generate,
    )

    assert result.improved_text == "Make the answer concise."
    assert result.review_required is False
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_same_line_whole_response_json_fence_is_parsed_as_structured_output():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return "```json" + _structured_output("Make the answer concise.") + "```"

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="be concise"),
        generate=fake_generate,
    )

    assert result.improved_text == "Make the answer concise."
    assert result.review_required is False


@pytest.mark.asyncio
async def test_plain_text_fallback_always_requires_review():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return "Summarize the source in three concise bullets."

    result = await improve_prompt(
        PromptImprovementInput(target="user_message", text="summarize the source"),
        generate=fake_generate,
    )

    assert result.improved_text == "Summarize the source in three concise bullets."
    assert result.review_required is True
    assert result.warnings == ("unstructured_output",)


@pytest.mark.asyncio
async def test_unchanged_plain_text_still_requires_review_as_unstructured_output():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return "Be helpful."

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="Be helpful."),
        generate=fake_generate,
    )

    assert result.status == "improved"
    assert result.improved_text == "Be helpful."
    assert result.review_required is True
    assert result.warnings == ("unstructured_output",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "I can't help rewrite that prompt.",
        "Here is the result:\n```json\n{\"status\":\"no_change\",\"findings\":[]}\n```",
    ],
)
async def test_refusal_or_mixed_structured_commentary_is_not_a_plain_candidate(raw: str):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return raw

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="Be helpful."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_model_output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"improved","findings":[]}',
        '{"status":"improved","improved_text":"","findings":[]}',
        "  \n\t ",
    ],
)
async def test_missing_empty_or_blank_candidate_is_rejected(raw: str):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return raw

    with pytest.raises(PromptImprovementError, match="usable candidate") as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="Be helpful."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_model_output"


@pytest.mark.asyncio
async def test_oversized_candidate_is_a_hard_preservation_failure():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output("x" * (MAX_CANDIDATE_CHARS + 1))

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="Be helpful."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "preservation_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "candidate", "warning"),
    [
        ("Summarize {{topic}}.", "Summarize the topic clearly.", "placeholder_mismatch"),
        (
            "Use https://example.com/reference as the source.",
            "Use the linked reference as the source.",
            "url_mismatch",
        ),
        (
            "Explain:\n```python\nprint('ok')\n```",
            "Explain print('ok').",
            "code_fence_mismatch",
        ),
        (
            "<role>Be helpful.</role>",
            "Be helpful.",
            "wrapper_mismatch",
        ),
    ],
)
async def test_detectable_literal_loss_preserves_candidate_for_review(
    draft: str,
    candidate: str,
    warning: str,
):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output(candidate)

    result = await improve_prompt(
        PromptImprovementInput(target="system", text=draft),
        generate=fake_generate,
    )

    assert result.improved_text == candidate
    assert result.review_required is True
    assert warning in result.warnings


@pytest.mark.asyncio
async def test_client_protected_token_loss_preserves_candidate_for_review():
    async def fake_generate(messages: list[dict[str, str]]) -> str:
        envelope = json.loads(messages[1]["content"])
        assert envelope == {
            "target": "user_message",
            "draft": "Summarize MACRO_SENTINEL.",
        }
        assert "protected_tokens" not in envelope
        return _structured_output("Summarize the attached material.")

    result = await improve_prompt(
        PromptImprovementInput(
            target="user_message",
            text="Summarize MACRO_SENTINEL.",
            protected_tokens=(
                PromptProtectedToken(
                    kind="macro",
                    value="MACRO_SENTINEL",
                    occurrences=1,
                ),
            ),
        ),
        generate=fake_generate,
    )

    assert result.improved_text == "Summarize the attached material."
    assert result.review_required is True
    assert "protected_token_mismatch" in result.warnings


@pytest.mark.asyncio
async def test_target_mismatch_preserves_candidate_for_review():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output("Be precise and concise.", target="user_message")

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="Be precise."),
        generate=fake_generate,
    )

    assert result.improved_text == "Be precise and concise."
    assert result.review_required is True
    assert "target_mismatch" in result.warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "Be helpful.\n",
        "\r\nBe helpful.\r\n",
    ],
)
async def test_outer_whitespace_or_line_ending_only_change_normalizes_to_no_change(candidate: str):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output(candidate)

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="Be helpful."),
        generate=fake_generate,
    )

    assert result.status == "no_change"
    assert result.improved_text is None
    assert result.review_required is False


@pytest.mark.asyncio
async def test_no_change_with_non_null_candidate_is_rejected():
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "status": "no_change",
                "improved_text": "This provider text must be ignored.",
                "findings": [],
            }
        )

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="user_message", text="Already clear."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_model_output"


@pytest.mark.asyncio
async def test_findings_are_bounded_and_normalized_without_scores_or_hidden_reasoning():
    findings = [
        {
            "category": " CLARITY ",
            "issue": "  Ambiguous request.  ",
            "change": "  Named the output.  ",
        }
    ] + [
        {
            "category": "invented-category",
            "issue": f"Issue {index}",
            "change": f"Change {index}",
        }
        for index in range(MAX_FINDINGS + 2)
    ]

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output("Return a concise answer.", findings=findings)

    result = await improve_prompt(
        PromptImprovementInput(target="user_message", text="return answer"),
        generate=fake_generate,
    )

    assert len(result.findings) == MAX_FINDINGS
    assert result.findings[0].category == "clarity"
    assert result.findings[0].issue == "Ambiguous request."
    assert result.findings[0].change == "Named the output."
    assert result.findings[1].category == "other"
    assert not hasattr(result, "quality_score")
    assert not hasattr(result.findings[0], "reasoning")


@pytest.mark.asyncio
async def test_large_rewrite_requires_review_but_keeps_bounded_candidate():
    draft = " ".join(["Keep each sentence focused on the supplied evidence."] * 20)
    candidate = "Write a poem about an unrelated subject."

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output(candidate)

    result = await improve_prompt(
        PromptImprovementInput(target="system", text=draft),
        generate=fake_generate,
    )

    assert result.improved_text == candidate
    assert result.review_required is True
    assert "large_rewrite" in result.warnings


@pytest.mark.asyncio
async def test_dollar_single_brace_and_unpaired_angle_names_are_not_placeholders():
    draft = "Explain $name, {item}, and <value without treating them as variables."
    candidate = "Explain the shell name, item, and comparison expression clearly."

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return _structured_output(candidate)

    result = await improve_prompt(
        PromptImprovementInput(target="user_message", text=draft),
        generate=fake_generate,
    )

    assert "placeholder_mismatch" not in result.warnings
    assert "wrapper_mismatch" not in result.warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token", "expected_code"),
    [
        (PromptProtectedToken(kind="macro", value="ABSENT", occurrences=1), "invalid_input"),
        (PromptProtectedToken(kind="macro", value="TOKEN", occurrences=2), "invalid_input"),
        (
            PromptProtectedToken(
                kind="macro",
                value="T" * (MAX_PROTECTED_TOKEN_CHARS + 1),
                occurrences=1,
            ),
            "invalid_input",
        ),
    ],
)
async def test_invalid_protected_token_is_rejected_before_generation(
    token: PromptProtectedToken,
    expected_code: str,
):
    called = False

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        nonlocal called
        called = True
        return _structured_output("unused")

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(
                target="user_message",
                text="TOKEN appears once.",
                protected_tokens=(token,),
            ),
            generate=fake_generate,
        )

    assert exc_info.value.code == expected_code
    assert called is False


def test_protected_tokens_are_deduplicated_after_exact_count_validation():
    token = PromptProtectedToken(kind="macro", value="TOKEN", occurrences=2)

    normalized = validate_protected_tokens("TOKEN and TOKEN", (token, token))

    assert normalized == (token,)


@pytest.mark.asyncio
async def test_protected_token_count_limit_is_enforced_before_generation():
    tokens = tuple(
        PromptProtectedToken(kind="macro", value=f"T{index}", occurrences=1)
        for index in range(MAX_PROTECTED_TOKENS + 1)
    )
    text = " ".join(token.value for token in tokens)
    called = False

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        nonlocal called
        called = True
        return _structured_output("unused")

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="user_message", text=text, protected_tokens=tokens),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_input"
    assert called is False


def test_protected_token_total_size_limit_is_enforced():
    token_count = (MAX_PROTECTED_TOKEN_TOTAL_CHARS // MAX_PROTECTED_TOKEN_CHARS) + 1
    values = [
        f"{index:02d}" + (chr(65 + index) * (MAX_PROTECTED_TOKEN_CHARS - 2))
        for index in range(token_count)
    ]

    with pytest.raises(PromptImprovementError) as exc_info:
        validate_protected_tokens(
            " ".join(values),
            tuple(
                PromptProtectedToken(kind="macro", value=value, occurrences=1)
                for value in values
            ),
        )

    assert exc_info.value.code == "invalid_input"


@pytest.mark.asyncio
async def test_invalid_target_and_whitespace_draft_are_rejected_before_generation():
    called = False

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        nonlocal called
        called = True
        return _structured_output("unused")

    for request in (
        PromptImprovementInput(target="assistant", text="Draft"),
        PromptImprovementInput(target="user_message", text=" \t\n"),
    ):
        with pytest.raises(PromptImprovementError) as exc_info:
            await improve_prompt(request, generate=fake_generate)
        assert exc_info.value.code == "invalid_input"

    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "candidate", "expected_mismatch"),
    [
        (
            "Use https://en.wikipedia.org/wiki/Function_(mathematics) as the source.",
            "Use https://en.wikipedia.org/wiki/Function_(mathematics as the source.",
            True,
        ),
        (
            "Search https://example.com/find?q=why? before answering.",
            "Search https://example.com/find?q=why before answering.",
            True,
        ),
        (
            "Search https://example.com/find?q=now! before answering.",
            "Search https://example.com/find?q=now before answering.",
            True,
        ),
        (
            "Read [the docs](https://example.com/a_(b)) before answering.",
            "Read [these docs](https://example.com/a_(b)) before answering.",
            False,
        ),
        (
            "Compare https://example.com/a and https://example.com/a.",
            "Use https://example.com/a for the comparison.",
            True,
        ),
        (
            "Read https://example.com/docs.",
            "Please read https://example.com/docs carefully.",
            False,
        ),
    ],
)
async def test_review_fix_round_1_url_literals_are_delimiter_aware_and_counted(
    draft: str,
    candidate: str,
    expected_mismatch: bool,
):
    result = await _improve_candidate(draft, candidate)

    assert ("url_mismatch" in result.warnings) is expected_mismatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "candidate", "expected_mismatch"),
    [
        (
            "Explain:\r\n```python\r\nprint('ok')\r\n```",
            "Explain print('ok').",
            True,
        ),
        (
            "Explain:\n````python\nprint('ok')\n````",
            "Explain:\n````python\nprint('ok')\n```",
            True,
        ),
        (
            "Explain:\n```python\nprint('ok')\n```",
            "Explain:\n```python\nprint('ok')\n~~~",
            True,
        ),
        (
            "Explain:\n   ```python\nprint('ok')\n   ```",
            "Explain print('ok').",
            True,
        ),
        (
            "Indented code:\n    ```python\n    print('ok')\n    ```",
            "Indented code: print('ok').",
            False,
        ),
        (
            "First:\n```\na\n```\nSecond:\n```\nb\n```",
            "First:\n```\na\n```\nSecond: b",
            True,
        ),
        (
            "Explain:\n~~~python\nprint('ok')\n~~~",
            "Explain print('ok').",
            True,
        ),
        (
            "Explain print('ok').",
            "Explain:\n```python\nprint('ok')\n```",
            True,
        ),
        (
            "Explain print('ok').",
            "Explain:\n```python\nprint('ok')",
            True,
        ),
        (
            "Explain:\n```python\nprint('ok')\n```",
            "Please explain:\n```python\nprint('ok')\n````",
            False,
        ),
        (
            "Explain:\r\n~~~python\r\nprint('ok')\r\n~~~",
            "Please explain:\n~~~python\nprint('ok')\n~~~",
            False,
        ),
    ],
)
async def test_review_fix_round_1_markdown_fences_follow_block_rules(
    draft: str,
    candidate: str,
    expected_mismatch: bool,
):
    result = await _improve_candidate(draft, candidate)

    assert ("code_fence_mismatch" in result.warnings) is expected_mismatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "candidate", "expected_mismatch"),
    [
        (
            "<role>Be helpful.</role>\nKeep literal <value>.",
            "Be helpful.\nKeep literal <value>.",
            True,
        ),
        (
            "Show `<value>` and <span class=\"x\">HTML-looking code</span>.",
            "Show the value and HTML-looking code.",
            False,
        ),
        (
            "<outer><inner>Keep this.</inner></outer>",
            "<outer><inner>Keep this concise.</inner></outer>",
            False,
        ),
        (
            "<outer><inner>Keep this.</inner></outer>",
            "<outer>Keep this concise.</outer>",
            True,
        ),
        (
            "<outer><inner>Keep this.</inner></outer>",
            "<outer><inner>Keep this.</outer></inner>",
            True,
        ),
        (
            "Keep this concise.",
            "<role>Keep this concise.</role>",
            True,
        ),
        (
            "Keep this concise.",
            "Keep this concise. <role>",
            True,
        ),
        (
            "Keep the literal <value> in examples.",
            "Keep the literal value in examples.",
            False,
        ),
    ],
)
async def test_review_fix_round_1_xml_wrappers_are_independently_preserved(
    draft: str,
    candidate: str,
    expected_mismatch: bool,
):
    result = await _improve_candidate(draft, candidate, target="system")

    assert ("wrapper_mismatch" in result.warnings) is expected_mismatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"status": "improved", "improved_text": "Better.", "findings": [], "target": 7},
        {"status": "improved", "improved_text": "Better."},
        {"status": "improved", "improved_text": "Better.", "findings": None},
        {"status": "improved", "improved_text": "Better.", "findings": "none"},
        {"status": "improved", "improved_text": 7, "findings": []},
        {"status": "improved", "improved_text": True, "findings": []},
        {"status": "no_change", "improved_text": "Better.", "findings": []},
        {"status": "unknown", "improved_text": "Better.", "findings": []},
        {"status": True, "improved_text": "Better.", "findings": []},
        {"status": "improved", "improved_text": "Better.", "findings": [], "analysis": "hidden"},
        {"status": "improved", "improved_text": "Better.", "findings": [], "quality_score": 9},
        {"status": "improved", "improved_text": "Better.", "findings": [], "target": "assistant"},
        {"status": "improved", "improved_text": "Better.", "findings": ["bad"]},
        {"status": "improved", "improved_text": "Better.", "findings": [{}]},
        {
            "status": "improved",
            "improved_text": "Better.",
            "findings": [{"category": 7, "issue": "Issue", "change": "Change"}],
        },
        {
            "status": "improved",
            "improved_text": "Better.",
            "findings": [{"category": "clarity", "issue": 7, "change": "Change"}],
        },
        {
            "status": "improved",
            "improved_text": "Better.",
            "findings": [
                {
                    "category": "clarity",
                    "issue": "Issue",
                    "change": "Change",
                    "reasoning": "hidden",
                }
            ],
        },
    ],
)
async def test_review_fix_round_1_invalid_structured_contract_is_rejected(payload: object):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return json.dumps(payload)

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="Be helpful."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_model_output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"status": "no_change", "findings": []},
        {"status": "no_change", "improved_text": None, "findings": []},
    ],
)
async def test_review_fix_round_1_valid_no_change_shapes_are_accepted(payload: object):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return json.dumps(payload)

    result = await improve_prompt(
        PromptImprovementInput(target="system", text="Be helpful."),
        generate=fake_generate,
    )

    assert result.status == "no_change"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"unknown":',
        '[{"status":"improved"}]',
        "Sorry, I can’t help improve that request.",
        "I’m sorry, but I cannot rewrite that prompt.",
        "As an AI, I cannot assist with that request.",
        "I’m unable to comply with that request.",
        "Here’s the improved prompt:\nBe more specific.",
        "Certainly! Here is the revised prompt:\nBe more specific.",
    ],
)
async def test_review_fix_round_1_json_like_refusal_or_commentary_is_rejected(raw: str):
    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        return raw

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="Be helpful."),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_model_output"


@pytest.mark.asyncio
async def test_review_fix_round_1_rewrite_check_is_bounded_for_adversarial_repetition():
    draft = ("ab" * 6_000) + "x"
    candidate = ("ba" * 6_000) + "y"

    started = time.perf_counter()
    result = await _improve_candidate(draft, candidate, target="system")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert "large_rewrite" in result.warnings


@pytest.mark.asyncio
async def test_review_fix_round_1_short_unrelated_rewrite_requires_review():
    result = await _improve_candidate(
        "Summarize the supplied evidence for a medical research audience.",
        "Write a limerick about penguins dancing under moonlight tonight.",
        target="system",
    )

    assert "large_rewrite" in result.warnings


@pytest.mark.parametrize(
    "token",
    [
        {"kind": 7, "value": "TOKEN", "occurrences": 1},
        {"kind": True, "value": "TOKEN", "occurrences": 1},
        {"kind": "macro", "value": 7, "occurrences": 1},
        {"kind": "macro", "value": True, "occurrences": 1},
        {"kind": "macro", "value": "TOKEN", "occurrences": True},
        {"kind": "macro", "value": "TOKEN", "occurrences": "1"},
        {"kind": "macro", "value": "TOKEN", "occurrences": 1.0},
    ],
)
def test_review_fix_round_1_malformed_protected_token_types_use_domain_error(token: object):
    with pytest.raises(PromptImprovementError) as exc_info:
        validate_protected_tokens("TOKEN", (token,))

    assert exc_info.value.code == "invalid_input"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [
        {"kind": [], "value": "TOKEN", "occurrences": 1},
        {"kind": {}, "value": "TOKEN", "occurrences": 1},
        {"kind": "macro", "value": [], "occurrences": 1},
        {"kind": "macro", "value": {}, "occurrences": 1},
        {"kind": "macro", "value": "TOKEN", "occurrences": []},
        {"kind": "macro", "value": "TOKEN", "occurrences": {}},
        {"kind": "macro", "value": "TOKEN", "occurrences": True},
    ],
)
async def test_review_fix_round_2_malformed_token_fields_fail_before_generation(token: object):
    called = False

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        nonlocal called
        called = True
        return _structured_output("unused")

    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(
                target="user_message",
                text="TOKEN",
                protected_tokens=(token,),
            ),
            generate=fake_generate,
        )

    assert exc_info.value.code == "invalid_input"
    assert called is False


def test_review_fix_round_1_protected_token_exact_boundaries_are_accepted():
    kind_boundary = PromptProtectedToken(kind="k" * 50, value="KIND", occurrences=1)
    value_boundary = PromptProtectedToken(
        kind="macro",
        value="V" * MAX_PROTECTED_TOKEN_CHARS,
        occurrences=1,
    )
    occurrence_boundary = PromptProtectedToken(
        kind="macro",
        value="COUNT",
        occurrences=MAX_PROTECTED_TOKEN_OCCURRENCES,
    )
    values = [f"{index:02d}" + (chr(65 + index) * 498) for index in range(8)]
    total_boundary = tuple(
        PromptProtectedToken(kind="macro", value=value, occurrences=1)
        for value in values
    )
    count_boundary = tuple(
        PromptProtectedToken(kind="macro", value=f"T{index:03d}", occurrences=1)
        for index in range(MAX_PROTECTED_TOKENS)
    )

    assert validate_protected_tokens("KIND", (kind_boundary,)) == (kind_boundary,)
    assert validate_protected_tokens(value_boundary.value, (value_boundary,)) == (value_boundary,)
    occurrence_text = " ".join(["COUNT"] * MAX_PROTECTED_TOKEN_OCCURRENCES)
    assert validate_protected_tokens(occurrence_text, (occurrence_boundary,)) == (occurrence_boundary,)
    assert validate_protected_tokens(" ".join(values), total_boundary) == total_boundary
    assert validate_protected_tokens(
        " ".join(token.value for token in count_boundary),
        count_boundary,
    ) == count_boundary


@pytest.mark.parametrize(
    "token",
    [
        PromptProtectedToken(kind="k" * 51, value="TOKEN", occurrences=1),
        PromptProtectedToken(kind="macro", value="TOKEN", occurrences=0),
        PromptProtectedToken(
            kind="macro",
            value="TOKEN",
            occurrences=MAX_PROTECTED_TOKEN_OCCURRENCES + 1,
        ),
    ],
)
def test_review_fix_round_1_protected_token_outside_boundaries_is_rejected(token: object):
    with pytest.raises(PromptImprovementError) as exc_info:
        validate_protected_tokens("TOKEN", (token,))

    assert exc_info.value.code == "invalid_input"


@pytest.mark.asyncio
async def test_review_fix_round_1_draft_size_exact_boundary_and_overflow():
    called = 0

    async def fake_generate(_messages: list[dict[str, str]]) -> str:
        nonlocal called
        called += 1
        return '{"status":"no_change","improved_text":null,"findings":[]}'

    accepted = await improve_prompt(
        PromptImprovementInput(target="system", text="x" * MAX_DRAFT_CHARS),
        generate=fake_generate,
    )
    with pytest.raises(PromptImprovementError) as exc_info:
        await improve_prompt(
            PromptImprovementInput(target="system", text="x" * (MAX_DRAFT_CHARS + 1)),
            generate=fake_generate,
        )

    assert accepted.status == "no_change"
    assert exc_info.value.code == "draft_too_large"
    assert called == 1


@pytest.mark.asyncio
async def test_corpus_cases_preserve_only_the_target_draft_in_generator_messages():
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in cases:
        called = False
        expected_envelope = {"target": case["target"], "draft": case["text"]}
        protected_tokens = tuple(
            PromptProtectedToken(**token)
            for token in case["protected_tokens"]
        )

        async def fake_generate(
            messages: list[dict[str, str]],
            expected_envelope: dict[str, str] = expected_envelope,
            output: dict[str, object] = case["output"],
        ) -> str:
            nonlocal called
            called = True
            assert [message["role"] for message in messages] == ["system", "user"]
            envelope = json.loads(messages[1]["content"])
            assert envelope == expected_envelope
            assert set(envelope) == {"target", "draft"}
            return json.dumps(output, ensure_ascii=False)

        request = PromptImprovementInput(
            target=case["target"],
            text=case["text"],
            protected_tokens=protected_tokens,
        )
        if "expected_error" not in case:
            result = await improve_prompt(request, generate=fake_generate)
            assert result.status == case["expected_status"]
            assert list(result.warnings) == case["expected_warnings"]
            expected_candidate = case["output"].get("improved_text")
            assert result.improved_text == expected_candidate
            assert called is True
        else:
            with pytest.raises(PromptImprovementError) as exc_info:
                await improve_prompt(request, generate=fake_generate)
            assert exc_info.value.code == case["expected_error"]
            assert called is False
