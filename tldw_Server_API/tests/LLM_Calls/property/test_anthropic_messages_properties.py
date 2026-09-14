from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.LLM_Calls.anthropic_messages import anthropic_messages_to_openai

_ASCII_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=20,
)


@st.composite
def _block_strategy(draw):
    block_type = draw(st.sampled_from(["text", "tool_use", "tool_result", "image"]))
    if block_type == "text":
        return {"type": "text", "text": draw(_ASCII_TEXT)}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": draw(_ASCII_TEXT) or "tool_1",
            "name": draw(_ASCII_TEXT) or "tool",
            "input": {"q": draw(_ASCII_TEXT)},
        }
    if block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": draw(_ASCII_TEXT),
            "content": draw(_ASCII_TEXT),
        }
    return {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/img.png"},
    }


@st.composite
def _message_strategy(draw):
    role = draw(st.sampled_from(["user", "assistant"]))
    content = draw(st.one_of(_ASCII_TEXT, st.lists(_block_strategy(), min_size=0, max_size=3)))
    return {"role": role, "content": content}


@given(
    messages=st.lists(_message_strategy(), min_size=1, max_size=5),
    system=st.one_of(st.none(), _ASCII_TEXT, st.lists(_block_strategy(), max_size=2)),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_anthropic_messages_conversion_roles(messages, system):
    openai_messages, system_message = anthropic_messages_to_openai(messages, system)

    assert system_message is None or isinstance(system_message, str)
    for message in openai_messages:
        assert message.get("role") in {"user", "assistant", "tool"}
