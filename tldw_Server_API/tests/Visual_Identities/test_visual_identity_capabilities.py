from tldw_Server_API.app.core.Visual_Identities.constraints import (
    MAX_EXPRESSION_ARCHIVE_BYTES,
    MAX_EXPRESSION_ASSET_BYTES,
    MAX_EXPRESSION_FRAME_COUNT,
    MAX_EXPRESSION_IMAGE_DIMENSION,
    build_visual_identity_capabilities,
)


def test_capabilities_include_avif_only_when_runtime_supports_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif",
        lambda: False,
    )

    capabilities = build_visual_identity_capabilities()

    assert "image/avif" not in capabilities["supported_mime_types"]
    assert capabilities["avif_enabled"] is False


def test_capabilities_include_baseline_mime_types_and_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif",
        lambda: False,
    )

    capabilities = build_visual_identity_capabilities()

    assert capabilities["supported_mime_types"] == [
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    assert capabilities["upload_max_bytes"] == MAX_EXPRESSION_ASSET_BYTES
    assert capabilities["archive_max_bytes"] == MAX_EXPRESSION_ARCHIVE_BYTES
    assert capabilities["max_dimension"] == MAX_EXPRESSION_IMAGE_DIMENSION
    assert capabilities["max_frame_count"] == MAX_EXPRESSION_FRAME_COUNT


def test_capabilities_add_avif_when_runtime_supports_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif",
        lambda: True,
    )

    capabilities = build_visual_identity_capabilities()

    assert "image/avif" in capabilities["supported_mime_types"]
    assert capabilities["avif_enabled"] is True
