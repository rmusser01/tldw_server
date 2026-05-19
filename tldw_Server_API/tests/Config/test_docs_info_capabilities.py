import asyncio
import configparser
import importlib
from pathlib import Path

from tldw_Server_API.app.api.v1.endpoints import config_info
from tldw_Server_API.app.core import config as config_mod


def _write_minimal_config(path: Path, *, stable_only: bool = True) -> None:
    path.write_text(
        "\n".join(
            [
                "[Authentication]",
                "auth_mode = single_user",
                "single_user_api_key = test-key",
                "",
                "[Server]",
                "host = 127.0.0.1",
                "port = 8000",
                "",
                "[API-Routes]",
                f"stable_only = {'true' if stable_only else 'false'}",
            ]
        ),
        encoding="utf-8",
    )


def test_docs_info_persona_capability_enabled_by_default_even_when_stable_only_true(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "true")
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.setitem(config_mod.settings, "PERSONA_ENABLED", True)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["persona"] is True
    assert safe_config["supported_features"]["persona"] is True
    assert safe_config["capabilities"] == safe_config["supported_features"]


def test_docs_info_persona_capability_can_be_disabled_via_route_toggle(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "true")
    monkeypatch.setenv("ROUTES_DISABLE", "persona")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.setitem(config_mod.settings, "PERSONA_ENABLED", True)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["persona"] is False
    assert safe_config["supported_features"]["persona"] is False


def test_docs_info_persona_feature_flag_disables_capability_even_when_route_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "true")
    monkeypatch.setenv("ROUTES_ENABLE", "persona")
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.setitem(config_mod.settings, "PERSONA_ENABLED", False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["persona"] is False
    assert safe_config["supported_features"]["persona"] is False


def test_docs_info_persona_capability_enabled_by_default_when_stable_only_false(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path, stable_only=False)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "false")
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.setitem(config_mod.settings, "PERSONA_ENABLED", True)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["persona"] is True
    assert safe_config["supported_features"]["persona"] is True


def test_docs_info_never_exposes_real_api_key(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["api_key_for_docs"] == ""
    assert safe_config["api_key_configured"] is True


def test_docs_info_exposes_slides_and_presentation_studio_capabilities(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasSlides"] is True
    assert safe_config["supported_features"]["hasSlides"] is True
    assert safe_config["capabilities"]["hasPresentationStudio"] is True
    assert safe_config["supported_features"]["hasPresentationStudio"] is True
    assert safe_config["capabilities"]["hasPresentationRender"] is True
    assert safe_config["supported_features"]["hasPresentationRender"] is True


def test_docs_info_exposes_audio_capabilities_from_audio_and_websocket_routes(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasAudio"] is True
    assert safe_config["capabilities"]["hasStt"] is True
    assert safe_config["capabilities"]["hasTts"] is True
    assert safe_config["capabilities"]["hasVoiceChat"] is True
    assert safe_config["capabilities"]["hasVoiceConversationTransport"] is True
    assert safe_config["supported_features"] == safe_config["capabilities"]


def test_docs_info_exposes_bulk_conference_ingest_capabilities(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()
    caps = safe_config["capabilities"]

    assert caps["hasMediaPlaylistPreflight"] is True
    assert caps["hasMediaIngestJobs"] is True
    assert caps["hasMediaIngestJobEvents"] is True
    assert caps["hasMediaIngestWorker"] is False
    assert caps["hasDurableMediaCollections"] is True
    assert caps["hasKnowledgeQaMediaScope"] is True
    assert safe_config["supported_features"] == caps


def test_docs_info_disables_voice_transport_when_audio_websocket_route_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_DISABLE", "audio-websocket")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasAudio"] is True
    assert safe_config["capabilities"]["hasStt"] is True
    assert safe_config["capabilities"]["hasTts"] is True
    assert safe_config["capabilities"]["hasVoiceChat"] is True
    assert safe_config["capabilities"]["hasVoiceConversationTransport"] is False


def test_docs_info_endpoint_returns_placeholder_api_key(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod._route_toggle_policy.cache_clear()

    payload = asyncio.run(config_info.get_documentation_config())

    assert payload["api_key"] == "YOUR_API_KEY"
    assert payload["api_key_configured"] is True
    assert "test-key" not in payload["examples"]["python"]
    assert "test-key" not in payload["examples"]["curl"]
    assert "test-key" not in payload["examples"]["javascript"]


def test_quickstart_redirect_reads_configured_ui_url(monkeypatch) -> None:
    parser = configparser.ConfigParser()
    parser.read_string("[UI]\nquickstart_url = /docs-static/Getting_Started/README.md\n")

    monkeypatch.delenv("QUICKSTART_URL", raising=False)
    monkeypatch.setattr(config_info.config_mod, "load_comprehensive_config", lambda: parser)

    response = asyncio.run(config_info.get_quickstart_redirect())

    assert response.status_code == 307
    assert response.headers["location"] == "/docs-static/Getting_Started/README.md"


def test_docs_info_persona_capability_stable_across_config_module_reload(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "true")
    monkeypatch.setenv("ROUTES_DISABLE", "persona")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.setitem(config_mod.settings, "PERSONA_ENABLED", True)
    config_mod._route_toggle_policy.cache_clear()

    disabled_caps = config_info.load_safe_config()["capabilities"]
    assert disabled_caps["persona"] is False

    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    reloaded_config_mod = importlib.reload(config_mod)
    monkeypatch.setitem(reloaded_config_mod.settings, "PERSONA_ENABLED", True)
    reloaded_config_mod._route_toggle_policy.cache_clear()

    enabled_caps = config_info.load_safe_config()["capabilities"]
    assert enabled_caps["persona"] is True
