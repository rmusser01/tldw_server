"""Isolated configuration diagnostics, without HTTP, storage, or provider calls."""

from configparser import ConfigParser
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetPackResponse, VNAssetSlotResponse
from tldw_Server_API.app.core import config as server_config
from tldw_Server_API.app.core.Image_Generation import config as image_config
from tldw_Server_API.app.core.Image_Generation.adapter_registry import ImageAdapterRegistry
from tldw_Server_API.app.core.Image_Generation.adapters.openrouter_image_adapter import OpenRouterImageAdapter
from tldw_Server_API.app.core.Jobs.manager import JobManager
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
    monkeypatch.setattr(server_config, "load_comprehensive_config", lambda: ConfigParser())
    monkeypatch.setattr(server_config, "_route_toggle_policy", server_config._route_toggle_policy.__wrapped__)
    for name in ("ROUTES_ENABLE", "ROUTES_DISABLE", "ROUTES_EXPERIMENTAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "true")
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


@pytest.mark.parametrize(
    ("jobs", "generation", "disabled_routes", "expected"),
    [
        (None, None, "", False),
        ("", " ", "", False),
        (None, "true", "", False),
        ("true", None, "", False),
        ("true", "true", "", True),
        ("true", "true", "vn-assets", False),
        ("true", "true", "vn-assets-generation", False),
    ],
)
def test_worker_configuration_matches_active_lifecycle_route_gates(
    pack: VNAssetPackResponse,
    monkeypatch: pytest.MonkeyPatch,
    jobs: str | None,
    generation: str | None,
    disabled_routes: str,
    expected: bool,
) -> None:
    """Active VN lifecycle predicates require explicit flags and enabled routes."""
    for name, value in (
        ("VN_ASSET_JOBS_WORKER_ENABLED", jobs),
        ("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED", generation),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("ROUTES_DISABLE", disabled_routes)
    result = preflight.generation_preflight(pack, [])
    assert result.local_workers_enabled is expected
    assert result.worker_health == "unknown"
    assert any("Local generation workers" in warning for warning in result.warnings) is not expected


@pytest.mark.parametrize("runtime_flag", ["TEST_MODE", "TLDW_TEST_MODE", "TLDW_WORKERS_SIDECAR_MODE"])
def test_worker_configuration_preserves_vn_lifecycle_runtime_flag_behavior(
    pack: VNAssetPackResponse,
    monkeypatch: pytest.MonkeyPatch,
    runtime_flag: str,
) -> None:
    """VN lifecycle predicates do not suppress enabled workers in these modes."""
    monkeypatch.setenv("VN_ASSET_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setenv("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setenv(runtime_flag, "true")
    result = preflight.generation_preflight(pack, [])
    assert result.local_workers_enabled is True
    assert result.worker_health == "unknown"


def test_preflight_does_not_generate_images_or_submit_jobs(
    pack: VNAssetPackResponse,
    slot: VNAssetSlotResponse,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration checks must not load models, call providers, or submit jobs."""
    registry = ImageAdapterRegistry(
        config_override={"default_backend": "openrouter", "enabled_backends": ["openrouter"]}
    )
    get_adapter = Mock(side_effect=AssertionError("Preflight must not instantiate generation adapters"))
    generate = Mock(side_effect=AssertionError("Preflight must not generate images"))
    create_job = Mock(side_effect=AssertionError("Preflight must not submit jobs"))
    monkeypatch.setattr(ImageAdapterRegistry, "get_adapter", get_adapter)
    monkeypatch.setattr(OpenRouterImageAdapter, "generate", generate)
    monkeypatch.setattr(JobManager, "create_job", create_job)
    monkeypatch.setattr(preflight, "get_registry", lambda: registry)
    result = preflight.generation_preflight(pack, [slot])
    assert len(result.slots) == 1
    check = result.slots[0]
    assert (check.slot_id, check.backend, check.status) == (slot.id, "openrouter", "configured")
    assert check.model == image_config.DEFAULT_OPENROUTER_IMAGE_MODEL
    assert result.worker_health == "unknown"
    get_adapter.assert_not_called()
    generate.assert_not_called()
    create_job.assert_not_called()
