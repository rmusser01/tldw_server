"""Isolated configuration diagnostics, without HTTP, storage, or provider calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetPackResponse, VNAssetSlotResponse
from tldw_Server_API.app.core.Image_Generation import config as image_config
from tldw_Server_API.app.core.VN_Assets import preflight


@pytest.fixture
def pack() -> VNAssetPackResponse:
    """Build metadata without creating a database."""
    return VNAssetPackResponse(
        id=7,
        owner_user_id=42,
        title="Test",
        primary_character_id=1,
        status="draft",
        content_rating="sfw",
        version=1,
        deleted=False,
    )


@pytest.fixture
def slot() -> VNAssetSlotResponse:
    """Build one planned slot without a service or repository."""
    return VNAssetSlotResponse(
        id=10,
        pack_id=7,
        asset_type="sprite",
        slot_key="neutral",
        variant_count=1,
        requires_review=True,
        required_for_runtime=True,
        status="planned",
    )


@pytest.fixture(autouse=True)
def local_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate catalog, registry, and configuration from the host environment."""
    monkeypatch.setattr(image_config, "_config_cache", None)
    monkeypatch.setattr(image_config, "get_config_section", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        preflight,
        "get_registry",
        lambda: SimpleNamespace(
            resolve_backend=lambda name: None if name == "disabled" else name or "openrouter",
        ),
    )
    monkeypatch.setattr(
        preflight,
        "list_image_models_for_catalog",
        lambda: [
            {"name": "openrouter", "is_configured": True},
            {"name": "swarmui", "is_configured": False},
        ],
    )
    for backend in ("OPENROUTER", "NOVITA", "TOGETHER", "MODELSTUDIO"):
        monkeypatch.delenv(f"{backend}_IMAGE_MODEL", raising=False)
    monkeypatch.setenv("VN_ASSET_JOBS_WORKER_ENABLED", "false")
    monkeypatch.setenv("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED", "false")


@pytest.mark.parametrize(
    ("pack_backend", "slot_backend", "expected_backend", "expected_status"),
    [
        (None, None, "openrouter", "configured"),
        ("swarmui", None, "swarmui", "missing_configuration"),
        ("swarmui", "openrouter", "openrouter", "configured"),
        ("openrouter", "disabled", None, "unavailable"),
        (None, "custom", "custom", "unknown"),
    ],
)
def test_backend_precedence_and_configuration_status(
    pack: VNAssetPackResponse,
    slot: VNAssetSlotResponse,
    pack_backend: str | None,
    slot_backend: str | None,
    expected_backend: str | None,
    expected_status: str,
) -> None:
    """Honor overrides and distinguish missing, disabled, and unknown backends."""
    pack.default_backend = pack_backend
    slot.backend_override = slot_backend
    check = preflight.generation_preflight(pack, [slot]).slots[0]
    assert (check.backend, check.status) == (expected_backend, expected_status)


@pytest.mark.parametrize(
    ("pack_model", "slot_model", "expected"),
    [
        (None, None, image_config.DEFAULT_OPENROUTER_IMAGE_MODEL),
        ("pack-model", None, "pack-model"),
        ("pack-model", "slot-model", "slot-model"),
    ],
)
def test_effective_model_includes_adapter_default(
    pack: VNAssetPackResponse,
    slot: VNAssetSlotResponse,
    pack_model: str | None,
    slot_model: str | None,
    expected: str,
) -> None:
    """Report the same model selection used by generation, including defaults."""
    pack.default_model = pack_model
    slot.model_override = slot_model
    assert preflight.generation_preflight(pack, [slot]).slots[0].model == expected


@pytest.mark.parametrize(("jobs", "generation"), [(False, False), (True, False), (False, True), (True, True)])
def test_worker_flags_never_claim_worker_health(
    pack: VNAssetPackResponse,
    monkeypatch: pytest.MonkeyPatch,
    jobs: bool,
    generation: bool,
) -> None:
    """Flag combinations describe local configuration, not worker liveness."""
    monkeypatch.setenv("VN_ASSET_JOBS_WORKER_ENABLED", str(jobs))
    monkeypatch.setenv("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED", str(generation))
    result = preflight.generation_preflight(pack, [])
    assert result.local_workers_enabled is (jobs and generation)
    assert result.worker_health == "unknown"
    assert "Add asset slots before starting generation." in result.warnings
    assert any("Local generation workers" in warning for warning in result.warnings) is not (jobs and generation)


def test_preflight_does_not_instantiate_a_generation_adapter(
    pack: VNAssetPackResponse,
    slot: VNAssetSlotResponse,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration checks must not load models, call providers, or submit jobs."""
    registry = Mock()
    registry.resolve_backend.return_value = "openrouter"
    monkeypatch.setattr(preflight, "get_registry", lambda: registry)
    preflight.generation_preflight(pack, [slot])
    assert [call[0] for call in registry.mock_calls] == ["resolve_backend"]
