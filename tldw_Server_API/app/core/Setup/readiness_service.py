"""Read-only setup readiness preview helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Setup import install_manager
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan
from tldw_Server_API.app.core.Setup.readiness_models import (
    LANE_CHAT,
    LANE_EMBEDDINGS_RAG,
    LANE_IDS,
    LANE_SPEECH,
    build_lane_summary,
)

_CHAT_MODEL_KEYS = {
    "anthropic": ("API", "anthropic_model"),
    "cohere": ("API", "cohere_model"),
    "deepseek": ("API", "deepseek_model"),
    "google": ("API", "google_model"),
    "groq": ("API", "groq_model"),
    "huggingface": ("API", "huggingface_model"),
    "mistral": ("API", "mistral_model"),
    "openai": ("API", "openai_model"),
    "openrouter": ("API", "openrouter_model"),
    "qwen": ("API", "qwen_model"),
}

_LOCAL_PROVIDER_MODEL_KEYS = {
    "custom_openai": ("API", "custom_openai_api_model"),
}

_LOCAL_PROVIDER_ENDPOINT_KEYS = {
    "custom_openai": ("API", "custom_openai_api_ip"),
    "kobold": ("Local-API", "kobold_openai_api_IP"),
    "llama": ("Local-API", "llama_api_IP"),
    "ooba": ("Local-API", "ooba_api_IP"),
    "tabby": ("Local-API", "tabby_api_IP"),
}


def _payload_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return dict(payload)


def _nested_payload_dict(payload: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in (_payload_dict(payload).items()):
        if isinstance(value, dict):
            result[str(key)] = dict(value)
    return result


def _text(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_config_update(
    config_updates: dict[str, dict[str, Any]],
    section: str,
    key: str,
    value: Any,
) -> None:
    text_value = _text(value)
    if not text_value:
        return
    config_updates.setdefault(section, {})[key] = text_value


def _append_unique(values: list[str], value: Any) -> None:
    text_value = _text(value)
    if text_value and text_value not in values:
        values.append(text_value)


def _merge_install_plan(target: dict[str, Any], source: InstallPlan) -> None:
    source_payload = _model_dump(source)
    target["stt"].extend(source_payload.get("stt", []))
    target["tts"].extend(source_payload.get("tts", []))
    source_embeddings = source_payload.get("embeddings", {})
    for key in ("huggingface", "custom", "onnx"):
        for model in source_embeddings.get(key, []):
            _append_unique(target["embeddings"][key], model)


def _model_dump(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    model_dict = getattr(model, "dict", None)
    if callable(model_dict):
        return model_dict()
    return dict(model)


def _empty_install_plan_payload() -> dict[str, Any]:
    return {
        "stt": [],
        "tts": [],
        "embeddings": {
            "huggingface": [],
            "custom": [],
            "onnx": [],
        },
    }


def _install_plan_has_work(install_plan: dict[str, Any]) -> bool:
    embeddings = install_plan["embeddings"]
    return bool(
        install_plan["stt"]
        or install_plan["tts"]
        or embeddings["huggingface"]
        or embeddings["custom"]
        or embeddings["onnx"]
    )


def _preview_chat_lane(
    lane: dict[str, Any],
    config_updates: dict[str, dict[str, Any]],
    secret_fields: list[dict[str, str]],
) -> dict[str, Any]:
    mode = _text(lane.get("mode"))
    provider = _text(lane.get("provider"))
    model = _text(lane.get("model"))

    if not lane:
        return build_lane_summary(LANE_CHAT)
    if mode == "skip":
        return build_lane_summary(
            LANE_CHAT,
            status="skipped",
            consequences=["Chat will remain limited until a provider or local endpoint is configured."],
        )
    if mode == "hosted":
        if not provider:
            return build_lane_summary(LANE_CHAT, status="blocked", blockers=["Hosted chat provider is required."])
        _merge_config_update(config_updates, "API", "default_api", provider)
        model_key = _CHAT_MODEL_KEYS.get(provider)
        if model_key and model:
            _merge_config_update(config_updates, model_key[0], model_key[1], model)
        if lane.get("api_key"):
            secret_fields.append(
                {
                    "section": "API",
                    "key": f"{provider}_api_key",
                    "provider": provider,
                    "state": "submitted",
                }
            )
        return build_lane_summary(
            LANE_CHAT,
            status="previewed",
            selection={
                "mode": mode,
                "provider": provider,
                "model": model,
                "endpoint": _text(lane.get("endpoint")),
            },
        )
    if mode == "local":
        provider = provider or "custom_openai"
        _merge_config_update(config_updates, "API", "default_api", provider)
        model_key = _LOCAL_PROVIDER_MODEL_KEYS.get(provider)
        endpoint_key = _LOCAL_PROVIDER_ENDPOINT_KEYS.get(provider)
        if model_key:
            _merge_config_update(config_updates, model_key[0], model_key[1], model)
        if endpoint_key:
            _merge_config_update(config_updates, endpoint_key[0], endpoint_key[1], lane.get("endpoint"))
        return build_lane_summary(
            LANE_CHAT,
            status="previewed",
            selection={"mode": mode, "provider": provider, "model": model},
        )
    return build_lane_summary(LANE_CHAT, status="blocked", blockers=["Unsupported chat setup mode."])


def _preview_embeddings_lane(
    lane: dict[str, Any],
    config_updates: dict[str, dict[str, Any]],
    install_plan: dict[str, Any],
) -> dict[str, Any]:
    mode = _text(lane.get("mode"))
    provider = _text(lane.get("provider"))
    model = _text(lane.get("model"))

    if not lane:
        return build_lane_summary(LANE_EMBEDDINGS_RAG)
    if mode == "skip":
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            status="skipped",
            consequences=["RAG search will use non-vector search until embeddings are configured."],
        )
    if not provider or not model:
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            status="blocked",
            blockers=["Embedding provider and model are required."],
        )

    if _truthy(lane.get("trusted_custom_model")) and not _truthy(
        lane.get("trusted_custom_model_acknowledged")
    ):
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            status="blocked",
            selection={"provider": provider, "model": model},
            blockers=["trusted custom Hugging Face model acknowledgement is required"],
        )

    _merge_config_update(config_updates, "Embeddings", "embedding_provider", provider)
    _merge_config_update(config_updates, "Embeddings", "embedding_model", model)
    if provider == "huggingface":
        target = "custom" if _truthy(lane.get("trusted_custom_model")) else "huggingface"
        _append_unique(install_plan["embeddings"][target], model)
    elif provider == "onnx":
        _append_unique(install_plan["embeddings"]["onnx"], model)

    return build_lane_summary(
        LANE_EMBEDDINGS_RAG,
        status="previewed",
        selection={"provider": provider, "model": model},
    )


def _preview_speech_lane(lane: dict[str, Any], install_plan: dict[str, Any]) -> dict[str, Any]:
    if not lane:
        return build_lane_summary(LANE_SPEECH)
    if _text(lane.get("mode")) == "skip":
        return build_lane_summary(
            LANE_SPEECH,
            status="skipped",
            consequences=["Transcription and local speech remain unavailable until configured."],
        )

    bundle_id = _text(lane.get("bundle_id"))
    resource_profile = _text(lane.get("resource_profile")) or "balanced"
    if not bundle_id:
        return build_lane_summary(LANE_SPEECH, status="blocked", blockers=["Speech bundle is required."])

    try:
        bundle_plan = install_manager.build_install_plan_from_bundle(
            bundle_id,
            resource_profile,
            tts_choice=_text(lane.get("tts_choice")) or None,
        )
    except (KeyError, ValueError) as exc:
        return build_lane_summary(LANE_SPEECH, status="blocked", blockers=[str(exc)])

    _merge_install_plan(install_plan, bundle_plan)
    return build_lane_summary(
        LANE_SPEECH,
        status="previewed",
        selection={
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "tts_choice": _text(lane.get("tts_choice")) or None,
        },
    )


def preview_readiness_selection(selection: Any) -> dict[str, Any]:
    """Return a sanitized readiness preview without writes, downloads, or verification calls."""

    payload = _payload_dict(selection)
    lane_inputs = _nested_payload_dict(payload.get("lanes"))
    config_updates: dict[str, dict[str, Any]] = {}
    secret_fields: list[dict[str, str]] = []
    install_plan = _empty_install_plan_payload()

    lanes = {
        LANE_CHAT: _preview_chat_lane(lane_inputs.get(LANE_CHAT, {}), config_updates, secret_fields),
        LANE_EMBEDDINGS_RAG: _preview_embeddings_lane(
            lane_inputs.get(LANE_EMBEDDINGS_RAG, {}),
            config_updates,
            install_plan,
        ),
        LANE_SPEECH: _preview_speech_lane(lane_inputs.get(LANE_SPEECH, {}), install_plan),
    }

    overlays: list[str] = []
    if config_updates:
        overlays.append("restart_required")

    operation_required = bool(config_updates or secret_fields or _install_plan_has_work(install_plan))

    return {
        "profile_id": _text(payload.get("profile_id")) or None,
        "lane_ids": list(LANE_IDS),
        "lanes": lanes,
        "overlays": overlays,
        "config_updates": config_updates,
        "secret_fields": secret_fields,
        "install_plan": install_plan,
        "operation_required": operation_required,
    }


def _lane_input(payload: dict[str, Any], lane_id: str) -> dict[str, Any]:
    lane = _nested_payload_dict(payload.get("lanes")).get(lane_id, {})
    if not lane:
        return {}
    selection = lane.get("selection")
    if isinstance(selection, dict):
        merged = dict(selection)
        for key, value in lane.items():
            if key not in {"selection", "label", "primary_capability", "secondary_capabilities"}:
                merged.setdefault(key, value)
        return merged
    return lane


def _health_ready(health: Any) -> bool:
    if not isinstance(health, dict):
        return False
    status_value = _text(health.get("status")).lower()
    return status_value in {"ready", "healthy", "ok", "available"}


def _verification_status(lanes: dict[str, dict[str, Any]]) -> str:
    statuses = {lane.get("status") for lane in lanes.values()}
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses:
        return "blocked"
    if "ready_with_warnings" in statuses:
        return "ready_with_warnings"
    if "provisioning" in statuses:
        return "provisioning"
    if "not_configured" in statuses:
        return "ready_with_warnings"
    return "ready"


def _verify_chat_lane(lane: dict[str, Any]) -> dict[str, Any]:
    mode = _text(lane.get("mode"))
    provider = _text(lane.get("provider"))
    endpoint = _text(lane.get("endpoint"))
    model = _text(lane.get("model"))

    if not lane:
        return build_lane_summary(LANE_CHAT)
    if mode == "skip" or lane.get("status") == "skipped":
        return build_lane_summary(
            LANE_CHAT,
            status="skipped",
            consequences=["Chat will remain limited until a provider or local endpoint is configured."],
        )
    if mode == "local":
        if not endpoint and provider not in {"custom_openai"}:
            return build_lane_summary(
                LANE_CHAT,
                status="ready_with_warnings",
                selection={"mode": mode, "provider": provider, "model": model},
                warnings=["Local chat endpoint reachability was not checked during setup verification."],
            )
        if endpoint and not endpoint.startswith(("http://", "https://")):
            return build_lane_summary(
                LANE_CHAT,
                status="blocked",
                selection={"mode": mode, "provider": provider, "model": model},
                blockers=["Local chat endpoint must be an HTTP URL."],
            )
        return build_lane_summary(
            LANE_CHAT,
            status="ready_with_warnings",
            selection={"mode": mode, "provider": provider, "model": model},
            warnings=["Local chat endpoint reachability was not checked during setup verification."],
        )
    if mode == "hosted":
        if not provider:
            return build_lane_summary(LANE_CHAT, status="blocked", blockers=["Hosted chat provider is required."])
        return build_lane_summary(
            LANE_CHAT,
            status="ready_with_warnings",
            selection={"mode": mode, "provider": provider, "model": model},
            warnings=["Hosted chat provider was not contacted during setup verification."],
        )
    return build_lane_summary(LANE_CHAT, status="blocked", blockers=["Unsupported chat setup mode."])


def _verify_embeddings_lane(lane: dict[str, Any]) -> dict[str, Any]:
    mode = _text(lane.get("mode"))
    provider = _text(lane.get("provider"))
    model = _text(lane.get("model"))

    if not lane:
        return build_lane_summary(LANE_EMBEDDINGS_RAG)
    if mode == "skip" or lane.get("status") == "skipped":
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            status="skipped",
            consequences=["RAG search will use non-vector search until embeddings are configured."],
        )
    if not provider or not model:
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            status="blocked",
            blockers=["Embedding provider and model are required."],
        )
    return build_lane_summary(
        LANE_EMBEDDINGS_RAG,
        status="ready_with_warnings",
        selection={"provider": provider, "model": model},
        warnings=["Embedding model availability will be confirmed during provisioning or first use."],
    )


async def _verify_speech_lane(lane: dict[str, Any]) -> dict[str, Any]:
    if not lane:
        return build_lane_summary(LANE_SPEECH)
    if _text(lane.get("mode")) == "skip" or lane.get("status") == "skipped":
        return build_lane_summary(
            LANE_SPEECH,
            status="skipped",
            consequences=["Transcription and local speech remain unavailable until configured."],
        )

    bundle_id = _text(lane.get("bundle_id"))
    resource_profile = _text(lane.get("resource_profile")) or "balanced"
    tts_choice = _text(lane.get("tts_choice")) or None
    if not bundle_id:
        return build_lane_summary(LANE_SPEECH, status="blocked", blockers=["Speech bundle is required."])

    verification = await install_manager.verify_audio_bundle_async(
        bundle_id,
        resource_profile,
        tts_choice=tts_choice,
    )
    stt_ready = _health_ready(verification.get("stt_health"))
    tts_ready = _health_ready(verification.get("tts_health"))
    if stt_ready and tts_ready:
        status = "ready"
        warnings: list[str] = []
    elif stt_ready or tts_ready or verification.get("status") == "ready":
        status = "ready_with_warnings"
        warnings = ["One or more secondary speech capabilities need attention."]
    else:
        status = "failed"
        warnings = []

    return build_lane_summary(
        LANE_SPEECH,
        status=status,
        selection={
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "tts_choice": tts_choice,
        },
        warnings=warnings,
    ) | {"verification": verification}


async def verify_readiness_lanes(selection: Any) -> dict[str, Any]:
    """Verify selected readiness lanes without implicit hosted calls or downloads."""

    payload = _payload_dict(selection)
    lanes = {
        LANE_CHAT: _verify_chat_lane(_lane_input(payload, LANE_CHAT)),
        LANE_EMBEDDINGS_RAG: _verify_embeddings_lane(_lane_input(payload, LANE_EMBEDDINGS_RAG)),
        LANE_SPEECH: await _verify_speech_lane(_lane_input(payload, LANE_SPEECH)),
    }

    return {
        "profile_id": _text(payload.get("profile_id")) or None,
        "lane_ids": list(LANE_IDS),
        "lanes": lanes,
        "overlays": [],
        "status": _verification_status(lanes),
        "verified_at": _utc_now(),
    }


__all__ = [
    "preview_readiness_selection",
    "verify_readiness_lanes",
]
