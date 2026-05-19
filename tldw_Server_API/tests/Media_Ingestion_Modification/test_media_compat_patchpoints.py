import inspect
from pathlib import Path
import pytest


pytestmark = pytest.mark.unit


def test_compat_patchpoints_module_removed():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "api"
        / "v1"
        / "endpoints"
        / "media"
        / "compat_patchpoints.py"
    )
    assert not module_path.exists()  # nosec B101


def test_media_module_drops_legacy_patchpoints():
    from tldw_Server_API.app.api.v1.endpoints import media as media_mod

    assert not hasattr(media_mod, "_download_url_async")  # nosec B101
    assert not hasattr(media_mod, "_save_uploaded_files")  # nosec B101


def test_web_scraping_endpoint_resolves_task_without_compat_module():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "api"
        / "v1"
        / "endpoints"
        / "media"
        / "process_web_scraping.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "compat_patchpoints" not in source  # nosec B101
    assert "_resolve_process_web_scraping_task" in source  # nosec B101

    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod

    resolver_source = inspect.getsource(endpoint_mod._resolve_process_web_scraping_task)
    assert "suppress(Exception)" not in resolver_source  # nosec B101


def test_web_scraping_endpoint_resolver_honors_media_shim(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod

    async def shim_process_web_scraping_task(**kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(
        media_mod,
        "process_web_scraping_task",
        shim_process_web_scraping_task,
        raising=True,
    )

    resolved = endpoint_mod._resolve_process_web_scraping_task()
    assert resolved is shim_process_web_scraping_task  # nosec B101


def test_web_scraping_endpoint_resolver_ignores_sync_media_shim(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod

    def shim_process_web_scraping_task(**kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(
        media_mod,
        "process_web_scraping_task",
        shim_process_web_scraping_task,
        raising=True,
    )

    resolved = endpoint_mod._resolve_process_web_scraping_task()
    assert resolved is endpoint_mod.process_web_scraping_task  # nosec B101


def test_web_scraping_endpoint_resolver_is_typed_and_documented():
    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod

    signature = inspect.signature(endpoint_mod._resolve_process_web_scraping_task)
    assert signature.return_annotation is not inspect.Signature.empty  # nosec B101
    assert endpoint_mod._resolve_process_web_scraping_task.__doc__  # nosec B101
