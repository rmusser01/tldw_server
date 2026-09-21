"""Read-only generation configuration checks using the shared image catalog."""

from collections.abc import Sequence

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetGenerationPreflightResponse,
    VNAssetPackResponse,
    VNAssetSlotPreflight,
    VNAssetSlotResponse,
)
from tldw_Server_API.app.core.config import route_enabled
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.config import get_image_generation_config, resolve_image_generation_model
from tldw_Server_API.app.core.Image_Generation.listing import list_image_models_for_catalog
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.worker_startup_policy import worker_route_default


def generation_preflight(
    pack: VNAssetPackResponse,
    slots: Sequence[VNAssetSlotResponse],
) -> VNAssetGenerationPreflightResponse:
    """Inspect local configuration without generating images or probing workers."""
    registry = get_registry()
    config = get_image_generation_config()
    catalog = {entry["name"]: entry for entry in list_image_models_for_catalog()}
    # Match the active VN lifecycle predicates: explicit flags plus route gates.
    local_workers_enabled = (
        env_flag_enabled("VN_ASSET_JOBS_WORKER_ENABLED")
        and worker_route_default("vn-assets", default_stable=True, route_enabled=route_enabled)
        and env_flag_enabled("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED")
        and worker_route_default("vn-assets-generation", default_stable=True, route_enabled=route_enabled)
    )
    warnings = ["Worker health has not been checked. Separately deployed workers may use different configuration."]
    if not local_workers_enabled:
        warnings.append(
            "Local generation workers are not both enabled. Enable them or confirm that separate workers are running."
        )
    if not slots:
        warnings.append("Add asset slots before starting generation.")
    checks = []
    for slot in slots:
        requested = (slot.backend_override or "").strip() or (pack.default_backend or "").strip() or None
        backend = registry.resolve_backend(requested)
        entry = catalog.get(backend)
        if backend is None:
            status = "unavailable"
            message = "Enable the selected image backend or choose an enabled backend in the pack or slot settings."
        elif entry is None:
            status = "unknown"
            message = "This backend has no configuration check. Confirm its setup on the generation worker."
        elif not entry.get("is_configured"):
            status = "missing_configuration"
            message = "Complete the image backend configuration on the generation worker before retrying."
        else:
            status = "configured"
            message = None
        checks.append(
            VNAssetSlotPreflight(
                slot_id=slot.id,
                backend=backend,
                model=resolve_image_generation_model(
                    backend,
                    (slot.model_override or "").strip() or (pack.default_model or "").strip() or None,
                    config,
                ),
                status=status,
                message=message,
            )
        )
    return VNAssetGenerationPreflightResponse(
        local_workers_enabled=local_workers_enabled,
        warnings=warnings,
        slots=checks,
    )
