from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _saved_config(models_dir: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "models_dir": str(models_dir),
        "allowed_paths": [],
    }
    values.update(overrides)
    return values


@pytest.mark.unit
@pytest.mark.parametrize("url", ["", "file:///tmp/model.gguf", "ftp://example.com/model.gguf"])
def test_validate_download_request_rejects_empty_and_unsupported_urls(tmp_path: Path, url: str) -> None:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAssetDownloadRequest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = LlamaCppAssetDownloadRequest(url=url, filename="model.gguf")

    with pytest.raises(ServerError):
        llamacpp_acquisition_service.validate_download_request(payload, _saved_config(models_dir))


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/model.gguf",
        "http://127.0.0.1/model.gguf",
        "http://10.0.0.1/model.gguf",
        "http://169.254.1.1/model.gguf",
    ],
)
def test_validate_download_request_rejects_private_network_sources_by_default(tmp_path: Path, url: str) -> None:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAssetDownloadRequest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = LlamaCppAssetDownloadRequest(url=url, filename="model.gguf")

    with pytest.raises(ServerError, match="private|local|loopback|link-local"):
        llamacpp_acquisition_service.validate_download_request(payload, _saved_config(models_dir))


@pytest.mark.unit
def test_validate_download_request_allows_private_network_when_explicitly_enabled(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAssetDownloadRequest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = LlamaCppAssetDownloadRequest(url="http://10.0.0.1/model.gguf", filename="model.gguf")

    validated = llamacpp_acquisition_service.validate_download_request(
        payload,
        _saved_config(models_dir, allow_private_downloads=True),
    )

    assert validated.destination_path == models_dir / "model.gguf"
    assert validated.source_url.startswith("http://10.0.0.1/")


@pytest.mark.unit
def test_redacted_source_label_removes_credentials_and_secret_queries() -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    label = llamacpp_acquisition_service.redacted_source_label(
        "https://user:pass@example.com/models/model.gguf?token=secret&download=1"
    )

    assert "user" not in label
    assert "pass" not in label
    assert "secret" not in label
    assert "token=" not in label
    assert "download=1" in label


@pytest.mark.unit
def test_resolve_download_destination_uses_allowlisted_models_dir_and_blocks_traversal(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAssetDownloadRequest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = LlamaCppAssetDownloadRequest(url="https://example.com/releases/model.gguf")

    destination = llamacpp_acquisition_service.resolve_download_destination(payload, _saved_config(models_dir))

    assert destination == models_dir / "model.gguf"

    traversal = LlamaCppAssetDownloadRequest(url="https://example.com/releases/model.gguf", filename="../model.gguf")
    with pytest.raises(ServerError, match="filename"):
        llamacpp_acquisition_service.resolve_download_destination(traversal, _saved_config(models_dir))


@pytest.mark.unit
def test_resolve_download_destination_rejects_delimiters_and_outside_allowed_paths(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service
    from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAssetDownloadRequest

    models_dir = tmp_path / "models"
    outside_dir = tmp_path / "outside"
    models_dir.mkdir()
    outside_dir.mkdir()

    bad_name = LlamaCppAssetDownloadRequest(url="https://example.com/model.gguf", filename="bad,name.gguf")
    with pytest.raises(ServerError, match="delimiter"):
        llamacpp_acquisition_service.resolve_download_destination(bad_name, _saved_config(models_dir))

    outside = LlamaCppAssetDownloadRequest(
        url="https://example.com/model.gguf",
        destination_dir=str(outside_dir),
        filename="model.gguf",
    )
    with pytest.raises(ServerError, match="allowed"):
        llamacpp_acquisition_service.resolve_download_destination(outside, _saved_config(models_dir))


@pytest.mark.unit
def test_partial_path_and_completed_download_validation(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    final_path.write_bytes(b"model-bytes")
    digest = hashlib.sha256(b"model-bytes").hexdigest()

    partial = llamacpp_acquisition_service.partial_download_path(final_path, "job-123")
    assert partial.parent == final_path.parent
    assert partial.name.endswith(".job-123.partial")

    warnings = llamacpp_acquisition_service.validate_completed_download(final_path, digest, len(b"model-bytes"))
    assert warnings == []

    with pytest.raises(ServerError, match="checksum"):
        llamacpp_acquisition_service.validate_completed_download(final_path, "0" * 64, len(b"model-bytes"))


@pytest.mark.unit
def test_register_completed_download_delegates_to_inventory_registration(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service

    registered: list[Path] = []
    model_path = tmp_path / "models" / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_text("model")

    def fake_register_asset_path(path: Path) -> dict[str, object]:
        registered.append(path)
        return {
            "asset_id": "gguf:registered",
            "kind": "gguf",
            "identity_basis": "resolved_path",
            "path": str(path),
            "resolved_path": str(path),
            "display_name": "model",
            "source": "registered_path",
            "metadata": {},
            "capabilities": ["unknown"],
            "mmproj_asset_ids": [],
            "base_model_asset_ids": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        llamacpp_acquisition_service.llamacpp_inventory_service,
        "register_asset_path",
        fake_register_asset_path,
    )

    asset = llamacpp_acquisition_service.register_completed_download(model_path)

    assert asset.asset_id == "gguf:registered"
    assert registered == [model_path]
