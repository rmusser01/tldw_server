from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import LlamaCppProfile, LlamaCppProfileMode


def _llamacpp_parser(models_dir: Path, **overrides: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    values = {
        "enabled": "true",
        "models_dir": str(models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
        "imported_asset_folders": "",
    }
    values.update(overrides)
    parser["LlamaCpp"] = values
    return parser


def _configure_assets(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(models_dir),
    )


@pytest.mark.unit
def test_chat_profile_resolves_base_model_without_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "chat.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="chat",
        name="Chat",
        mode=LlamaCppProfileMode.CHAT,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    resolved = resolve_profile_launch(profile)

    assert resolved.model_path == base.resolve()
    assert "mmproj" not in resolved.server_args
    assert resolved.capabilities["chat"] is True
    assert resolved.capabilities["vision"] is False
    assert resolved.modalities == {"input": ["text"], "output": ["text"]}


@pytest.mark.unit
def test_vision_profile_requires_mmproj_asset(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "vision.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    with pytest.raises(ServerError, match="mmproj"):
        resolve_profile_launch(profile)


@pytest.mark.unit
def test_vision_profile_injects_resolved_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "llava.gguf"
    projector = models_dir / "mmproj-llava.gguf"
    base.write_text("base")
    projector.write_text("projector")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
        mmproj_model_id=llamacpp_inventory_service.asset_id_for_path(projector, "mmproj"),
        server_args={"ctx_size": 4096},
    )

    resolved = resolve_profile_launch(profile)

    assert resolved.model_path == base.resolve()
    assert resolved.mmproj_path == projector.resolve()
    assert resolved.server_args["ctx_size"] == 4096
    assert resolved.server_args["mmproj"] == str(projector.resolve())
    assert resolved.capabilities["chat"] is True
    assert resolved.capabilities["vision"] is True
    assert resolved.modalities == {"input": ["text", "image"], "output": ["text"]}


@pytest.mark.unit
def test_vision_profile_rejects_conflicting_manual_mmproj(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "llava.gguf"
    projector = models_dir / "mmproj-llava.gguf"
    other_projector = models_dir / "mmproj-other.gguf"
    base.write_text("base")
    projector.write_text("projector")
    other_projector.write_text("other projector")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
        mmproj_model_id=llamacpp_inventory_service.asset_id_for_path(projector, "mmproj"),
        server_args={"mmproj": str(other_projector)},
    )

    with pytest.raises(ServerError, match="conflict"):
        resolve_profile_launch(profile)


@pytest.mark.unit
def test_embedding_profile_derives_text_vector_capability(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "embedding.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="embedding",
        name="Embedding",
        mode=LlamaCppProfileMode.EMBEDDING,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    resolved = resolve_profile_launch(profile)

    assert resolved.capabilities["chat"] is False
    assert resolved.capabilities["embeddings"] is True
    assert resolved.modalities == {"input": ["text"], "output": ["embedding"]}


@pytest.mark.unit
def test_rerank_profile_derives_text_score_capability(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "rerank.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="rerank",
        name="Rerank",
        mode=LlamaCppProfileMode.RERANK,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    resolved = resolve_profile_launch(profile)

    assert resolved.capabilities["chat"] is False
    assert resolved.capabilities["rerank"] is True
    assert resolved.modalities == {"input": ["text"], "output": ["score"]}


@pytest.mark.unit
def test_server_generic_profile_makes_no_specialized_capability_claim(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import resolve_profile_launch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "server.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="server",
        name="Server",
        mode=LlamaCppProfileMode.SERVER_GENERIC,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    resolved = resolve_profile_launch(profile)

    assert resolved.capabilities["chat"] is False
    assert resolved.capabilities["vision"] is False
    assert resolved.capabilities["embeddings"] is False
    assert resolved.capabilities["rerank"] is False
    assert resolved.modalities == {"input": ["text"], "output": ["text"]}


@pytest.mark.unit
def test_profile_capability_metadata_is_bounded(monkeypatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import profile_capability_metadata

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base = models_dir / "chat.gguf"
    base.write_text("base")
    _configure_assets(monkeypatch, models_dir)
    profile = LlamaCppProfile(
        profile_id="chat",
        name="Chat",
        mode=LlamaCppProfileMode.CHAT,
        model_id=llamacpp_inventory_service.asset_id_for_path(base, "gguf"),
    )

    metadata = profile_capability_metadata(profile)

    assert metadata["capabilities"]["chat"] is True
    assert metadata["modalities"] == {"input": ["text"], "output": ["text"]}
    assert str(base.resolve()) not in repr(metadata)
