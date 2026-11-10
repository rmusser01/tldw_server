from tldw_Server_API.app.api.v1.utils import media_deprecation_signals as signals_module


def test_build_media_legacy_signal_includes_standard_headers():
    signal = signals_module.build_media_legacy_signal(
        successor="/api/v1/media/process-videos",
        warning_code="legacy_compat_path",
    )
    assert signal.headers["Deprecation"] == "true"
    assert "Sunset" in signal.headers
    assert (
        signal.headers["Link"]
        == "</api/v1/media/process-videos>; rel=successor-version"
    )
    assert signal.payload["warning"] == "deprecated_endpoint"
    assert signal.payload["code"] == "legacy_compat_path"
    assert signal.payload["successor"] == "/api/v1/media/process-videos"


def test_apply_media_legacy_headers_updates_response_headers():
    signal = signals_module.build_media_legacy_signal(
        successor="/api/v1/media/process-videos",
        warning_code="legacy_compat_path",
    )
    response = type("ResponseStub", (), {"headers": {}})()
    updated = signals_module.apply_media_legacy_headers(response, signal)
    assert updated is response
    assert response.headers.get("Deprecation") == "true"
    assert "Sunset" in response.headers
