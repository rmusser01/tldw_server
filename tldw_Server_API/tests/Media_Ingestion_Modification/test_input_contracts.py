from tldw_Server_API.app.api.v1.utils import media_input_contracts as contracts_module


def test_normalize_urls_field_handles_legacy_empty_list_sentinel():
    assert contracts_module.normalize_urls_field([""]) is None
    assert contracts_module.normalize_urls_field(["https://example.com"]) == [
        "https://example.com"
    ]


def test_validate_media_inputs_delegates_to_provided_callable():
    calls = []

    def _fake_validator(media_type, urls, files):
        calls.append((media_type, urls, files))

    contracts_module.validate_media_inputs(
        _fake_validator,
        "video",
        ["https://example.com"],
        None,
    )
    assert calls == [("video", ["https://example.com"], None)]

