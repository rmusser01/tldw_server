from tldw_Server_API.app.core.Chat import chat_service


def _messages(base64_payload: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_payload}"}},
            ],
        }
    ]


def test_token_estimate_redacts_base64_images():
    est_with_payload = chat_service._estimate_tokens_from_messages(_messages("A" * 4000))
    est_small_payload = chat_service._estimate_tokens_from_messages(_messages("A" * 4))

    assert est_with_payload == est_small_payload
