# llamacpp.py
# Description: This file contains the API endpoints for managing Llama.cpp server operations in tldw_Server_API.
#
# Imports
import inspect
from pathlib import Path
from typing import Any, Optional

#
# Thid-party Libraries
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, RequireRole, User
from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppConfigResponse,
    LlamaCppConfigUpdateRequest,
    LlamaCppHardwareSnapshotResponse,
    LlamaCppInventoryItem,
    LlamaCppInventoryResponse,
    LlamaCppLifecycleActionResponse,
    LlamaCppLogTailResponse,
    LlamaCppProfileCreateRequest,
    LlamaCppProfileDeleteResponse,
    LlamaCppProfileListResponse,
    LlamaCppProfileResponse,
    LlamaCppProfileUpdateRequest,
    LlamaCppRegisterModelPathRequest,
    LlamaCppRuntimeListResponse,
    LlamaCppRuntimeResponse,
    LlamaCppStartByModelRequest,
    LlamaCppStartByModelResponse,
    LlamaCppUseInChatResponse,
    LlamaCppValidationRequest,
    LlamaCppValidationResponse,
)

from tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler import LlamaCppHandler

#
# Local Imports
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError, ModelNotFoundError, ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Manager import LLMInferenceManager
from tldw_Server_API.app.core.Local_LLM import (
    http_utils,
    llamacpp_config_service,
    llamacpp_hardware_service,
    llamacpp_inventory_service,
    llamacpp_provider_service,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import DEFAULT_PROFILE_ID
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppProfile,
    LlamaCppProfileConflictError,
    LlamaCppProfileNotFoundError,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_supervisor_service import LlamaCppSupervisor

#
########################################################################################################################
#
# Functions:

router = APIRouter()

#     LlamaCppConfig: Defines paths and default arguments for llama.cpp/server.
#
#     LlamaCpp_Handler:
#
#         Manages a single llama.cpp/server process (_active_server_process).
#
#         start_server(): This is your model swap function. If an existing server is running (managed by this handler), it calls stop_server() first, then starts a new server with the new model_filename and server_args.
#
#         stop_server(): Terminates the managed server process, handling process groups for robust cleanup.
#
#         inference(): Sends requests to the Llama.cpp server's OpenAI-compatible API (e.g., /v1/chat/completions).
#
#         list_models(): Scans the models_dir for .gguf files.
#
#         get_server_status(): Reports the current state of the managed server.
#
#         _cleanup_managed_server_sync(): Ensures server is stopped on application exit.
#
#         Optional logging of llama.cpp/server output to a file.
#
#     LLM_Inference_Manager Updates:
#
#         Initializes and provides access to LlamaCppHandler.
#
#         Delegates relevant calls (start_server, stop_server, run_inference, list_local_models) to the LlamaCppHandler.
#
#     API Endpoints: Provide HTTP interfaces to list models, start/swap the server with a specific model, stop it, get status, and run inference.

# Assuming 'llm_manager' is available, e.g., initialized in main.py and stored
# on app.state.llm_manager. Tests may still patch the module-level llm_manager
# for compatibility, so we fall back to the global when state is missing.


def _resolve_llm_manager(request: Request) -> LLMInferenceManager:
    mgr = getattr(request.app.state, "llm_manager", None)
    if mgr is None:
        mgr = globals().get("llm_manager")
    if mgr is None:
        raise HTTPException(status_code=503, detail="LLM manager not initialized.")
    return mgr  # type: ignore[return-value]


def _llamacpp_unavailable(detail: Optional[str] = None) -> HTTPException:
    base = "Managed llama.cpp backend is not configured."
    guidance = "Enable [LlamaCpp] enabled=true in Config_Files/config.txt and restart the server."
    safe_detail = "backend unavailable" if detail else None
    message = f"{base} ({safe_detail}) {guidance}" if safe_detail else f"{base} {guidance}"
    return HTTPException(status_code=503, detail=message)


def _resolve_llamacpp_target(llm_manager: LLMInferenceManager, required: tuple[str, ...]):
    """
    Return an object (handler or manager) that supports the required llama.cpp methods.
    Falls back to the manager for compatibility with tests that monkeypatch llm_manager directly.
    """
    handler = getattr(llm_manager, "llamacpp", None)
    candidates = [handler, llm_manager]
    for cand in candidates:
        if cand and all(hasattr(cand, name) for name in required):
            return cand
    raise _llamacpp_unavailable()


def _resolve_llamacpp_supervisor(llm_manager: LLMInferenceManager) -> LlamaCppSupervisor:
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is None:
        raise _llamacpp_unavailable()
    return supervisor


def _log_sanitized_manager_error(llm_manager: LLMInferenceManager, message: str) -> None:
    """Log a sanitized manager fallback without attaching exception details."""
    llm_manager.logger.error(message)


def _profile_response(profile: LlamaCppProfile) -> LlamaCppProfileResponse:
    return LlamaCppProfileResponse.model_validate(profile.model_dump(mode="python"))


def _runtime_response(runtime: LlamaCppRuntime) -> LlamaCppRuntimeResponse:
    return LlamaCppRuntimeResponse.model_validate(runtime.model_dump(mode="python"))


def _get_profile_or_404(supervisor: LlamaCppSupervisor, profile_id: str) -> LlamaCppProfile:
    for profile in supervisor.list_profiles():
        if profile.profile_id == profile_id:
            return profile
    raise HTTPException(status_code=404, detail=f"Llama.cpp profile '{profile_id}' was not found.")


def _supervisor_error_to_http(exc: Exception, llm_manager: LLMInferenceManager, log_message: str) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, LlamaCppProfileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, LlamaCppProfileConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (ValidationError, ValueError, ModelNotFoundError, ServerError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, InferenceError):
        return _llamacpp_unavailable(str(exc))
    _log_sanitized_manager_error(llm_manager, log_message)
    return HTTPException(status_code=500, detail="An unexpected error occurred.")


def _lifecycle_response(profile_id: str, action: str, runtime: LlamaCppRuntime) -> LlamaCppLifecycleActionResponse:
    return LlamaCppLifecycleActionResponse(
        profile_id=profile_id,
        action=action,
        state=runtime.state,
        accepted=True,
        message=runtime.message,
    )


def _start_by_model_response(runtime: LlamaCppRuntime, model_id: str) -> dict[str, object]:
    return {
        "status": "running" if runtime.state == LlamaCppRuntimeState.RUNNING else runtime.state.value,
        "backend": "llamacpp",
        "model_id": model_id,
        "model": runtime.model_path,
        "path": runtime.model_path,
        "host": runtime.host,
        "port": runtime.port,
        "pid": runtime.pid,
        "endpoint": runtime.endpoint,
        "message": runtime.message,
    }


def _start_by_path_response(runtime: LlamaCppRuntime, model_filename: str) -> dict[str, object]:
    return {
        "status": "running" if runtime.state == LlamaCppRuntimeState.RUNNING else runtime.state.value,
        "backend": "llamacpp",
        "model": runtime.model_path or model_filename,
        "path": runtime.model_path,
        "host": runtime.host,
        "port": runtime.port,
        "pid": runtime.pid,
        "endpoint": runtime.endpoint,
        "message": runtime.message,
    }


def _runtime_base_url(runtime: LlamaCppRuntime) -> str:
    if runtime.state != LlamaCppRuntimeState.RUNNING or runtime.port is None:
        raise llamacpp_provider_service.ManagedServerNotRunningError("Managed llama.cpp server is not running.")
    return llamacpp_provider_service.normalize_managed_base_url(runtime.host, runtime.port)


async def _use_runtime_in_chat(runtime: LlamaCppRuntime) -> dict[str, object]:
    endpoint = _runtime_base_url(runtime)
    try:
        with llamacpp_provider_service.llamacpp_config_write_lock():
            llamacpp_provider_service.setup_manager.update_config(
                {
                    llamacpp_provider_service.PROVIDER_SECTION: {
                        llamacpp_provider_service.PROVIDER_ENDPOINT_FIELD: endpoint
                    }
                }
            )
            llamacpp_provider_service.refresh_config_cache()
    except Exception as exc:
        raise llamacpp_provider_service.ProviderConfigWriteError(
            "Failed to update llama.cpp chat provider endpoint."
        ) from exc

    warnings: list[str] = []
    effective = True
    env_override = llamacpp_provider_service.get_provider_endpoint_env_override()
    if env_override:
        effective = False
        warnings.append(
            f"{env_override} is set, so updating "
            f"{llamacpp_provider_service.PROVIDER_SECTION}."
            f"{llamacpp_provider_service.PROVIDER_ENDPOINT_FIELD} may not affect chat."
        )
    return {
        "provider": llamacpp_provider_service.PROVIDER_NAME,
        "endpoint": endpoint,
        "updated": True,
        "effective": effective,
        "warnings": warnings,
    }


async def _use_default_runtime_in_chat(supervisor: LlamaCppSupervisor) -> dict[str, object]:
    try:
        runtime = supervisor.get_runtime(DEFAULT_PROFILE_ID)
    except LlamaCppProfileNotFoundError as exc:
        raise llamacpp_provider_service.ManagedServerNotRunningError(
            "Managed llama.cpp server is not running."
        ) from exc
    return await _use_runtime_in_chat(runtime)


def _messages_to_prompt(messages: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _is_chat_endpoint(api_endpoint: str) -> bool:
    return "chat/completions" in api_endpoint


async def _post_supervisor_runtime_inference(
    runtime: LlamaCppRuntime,
    payload: "LlamaCppInferenceRequest",
) -> dict[str, Any]:
    base_url = _runtime_base_url(runtime)
    request_payload = payload.to_kwargs()
    api_endpoint = str(request_payload.pop("api_endpoint", "/v1/chat/completions") or "/v1/chat/completions")
    timeout = request_payload.pop("timeout", None)
    request_payload.pop("stream", None)
    prompt_value = request_payload.pop("prompt", None)
    messages_value = request_payload.pop("messages", None)
    if _is_chat_endpoint(api_endpoint):
        if messages_value:
            request_payload["messages"] = messages_value
        elif prompt_value is not None:
            request_payload["messages"] = [{"role": "user", "content": prompt_value}]
        else:
            raise InferenceError("Either 'prompt' or 'messages' must be provided for inference.")
    else:
        if prompt_value is None and messages_value:
            prompt_value = _messages_to_prompt(messages_value)
        if prompt_value is None:
            raise InferenceError("Prompt is required for completion endpoint inference.")
        request_payload["prompt"] = prompt_value
    request_payload["stream"] = False
    target_url = f"{base_url}/{api_endpoint.lstrip('/')}"
    async with http_utils.create_async_client(timeout=timeout) as client:
        try:
            return await http_utils.request_json(
                client,
                "POST",
                target_url,
                json=request_payload,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            status = http_utils.get_http_status_from_exception(exc)
            if status is not None:
                error_text = http_utils.get_http_error_text(exc)
                raise HTTPException(status_code=status, detail=error_text or "Llama.cpp API request failed.") from exc
            if http_utils.is_network_error(exc):
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not communicate with managed llama.cpp server at {target_url}: {exc}",
                ) from exc
            error_text = http_utils.get_http_error_text(exc)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during Llama.cpp inference: {error_text}",
            ) from exc


# --- Llama.cpp Specific Endpoints ---
@router.get(
    "/llamacpp/config",
    summary="Get llama.cpp Admin Config State",
    response_model=LlamaCppConfigResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_config_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppConfigResponse:
    return llamacpp_config_service.get_config_state(llm_manager)


@router.put(
    "/llamacpp/config",
    summary="Update llama.cpp Admin Config",
    response_model=LlamaCppConfigResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def update_llamacpp_config_endpoint(
    payload: LlamaCppConfigUpdateRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppConfigResponse:
    return llamacpp_config_service.update_config_state(payload, llm_manager)


@router.post(
    "/llamacpp/validate",
    summary="Validate llama.cpp Binary",
    response_model=LlamaCppValidationResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def validate_llamacpp_binary_endpoint(
    payload: LlamaCppValidationRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppValidationResponse:
    return await run_in_threadpool(
        llamacpp_config_service.validate_binary,
        payload.binary_path,
        payload.timeout_seconds,
        llm_manager=llm_manager,
        run_probe=payload.run_probe,
    )


@router.get(
    "/llamacpp/inventory",
    summary="List llama.cpp Model Inventory",
    response_model=LlamaCppInventoryResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_inventory_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppInventoryResponse:
    config_state = llamacpp_config_service.get_config_state(llm_manager)
    return llamacpp_inventory_service.scan_inventory(config_state)


@router.post(
    "/llamacpp/models/register-path",
    summary="Register a llama.cpp Model Path",
    response_model=LlamaCppInventoryItem,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def register_llamacpp_model_path_endpoint(
    payload: LlamaCppRegisterModelPathRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppInventoryItem:
    _ = llm_manager
    try:
        return llamacpp_inventory_service.register_model_path(Path(payload.path))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/llamacpp/profiles",
    summary="List llama.cpp Runtime Profiles",
    response_model=LlamaCppProfileListResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def list_llamacpp_profiles_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppProfileListResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    return LlamaCppProfileListResponse(profiles=[_profile_response(profile) for profile in supervisor.list_profiles()])


@router.post(
    "/llamacpp/profiles",
    summary="Create llama.cpp Runtime Profile",
    response_model=LlamaCppProfileResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def create_llamacpp_profile_endpoint(
    payload: LlamaCppProfileCreateRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppProfileResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _profile_response(await supervisor.create_profile(payload))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error creating Llama.cpp profile") from e


@router.get(
    "/llamacpp/profiles/{profile_id}",
    summary="Get llama.cpp Runtime Profile",
    response_model=LlamaCppProfileResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppProfileResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    return _profile_response(_get_profile_or_404(supervisor, profile_id))


@router.put(
    "/llamacpp/profiles/{profile_id}",
    summary="Update llama.cpp Runtime Profile",
    response_model=LlamaCppProfileResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def update_llamacpp_profile_endpoint(
    profile_id: str,
    payload: LlamaCppProfileUpdateRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppProfileResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _profile_response(await supervisor.update_profile(profile_id, payload))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error updating Llama.cpp profile") from e


@router.delete(
    "/llamacpp/profiles/{profile_id}",
    summary="Delete llama.cpp Runtime Profile",
    response_model=LlamaCppProfileDeleteResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def delete_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppProfileDeleteResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        deleted = await supervisor.delete_profile(profile_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Llama.cpp profile '{profile_id}' was not found.")
        return LlamaCppProfileDeleteResponse(profile_id=profile_id, deleted=True)
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error deleting Llama.cpp profile") from e


@router.post(
    "/llamacpp/profiles/{profile_id}/start",
    summary="Start llama.cpp Runtime Profile",
    response_model=LlamaCppLifecycleActionResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def start_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLifecycleActionResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _lifecycle_response(profile_id, "start", await supervisor.start_profile(profile_id))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error starting Llama.cpp profile") from e


@router.post(
    "/llamacpp/profiles/{profile_id}/stop",
    summary="Stop llama.cpp Runtime Profile",
    response_model=LlamaCppLifecycleActionResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def stop_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLifecycleActionResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _lifecycle_response(profile_id, "stop", await supervisor.stop_profile(profile_id))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error stopping Llama.cpp profile") from e


@router.post(
    "/llamacpp/profiles/{profile_id}/pause",
    summary="Pause llama.cpp Runtime Profile",
    response_model=LlamaCppLifecycleActionResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def pause_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLifecycleActionResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _lifecycle_response(profile_id, "pause", await supervisor.pause_profile(profile_id))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error pausing Llama.cpp profile") from e


@router.post(
    "/llamacpp/profiles/{profile_id}/resume",
    summary="Resume llama.cpp Runtime Profile",
    response_model=LlamaCppLifecycleActionResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def resume_llamacpp_profile_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLifecycleActionResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _lifecycle_response(profile_id, "resume", await supervisor.resume_profile(profile_id))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error resuming Llama.cpp profile") from e


@router.post(
    "/llamacpp/profiles/{profile_id}/use-in-chat",
    summary="Use llama.cpp Runtime Profile in Chat",
    response_model=LlamaCppUseInChatResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def use_llamacpp_profile_in_chat_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppUseInChatResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return await _use_runtime_in_chat(supervisor.get_runtime(profile_id))
    except llamacpp_provider_service.ManagedServerNotRunningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except llamacpp_provider_service.ProviderConfigWriteError as e:
        raise HTTPException(status_code=500, detail="Failed to update llama.cpp chat provider endpoint.") from e
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error wiring Llama.cpp profile endpoint") from e


@router.get(
    "/llamacpp/instances",
    summary="List llama.cpp Runtime Instances",
    response_model=LlamaCppRuntimeListResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def list_llamacpp_instances_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppRuntimeListResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    return LlamaCppRuntimeListResponse(runtimes=[_runtime_response(runtime) for runtime in supervisor.list_runtimes()])


@router.get(
    "/llamacpp/instances/{profile_id}",
    summary="Get llama.cpp Runtime Instance",
    response_model=LlamaCppRuntimeResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_instance_endpoint(
    profile_id: str,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppRuntimeResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return _runtime_response(supervisor.get_runtime(profile_id))
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error getting Llama.cpp instance") from e


@router.get(
    "/llamacpp/instances/{profile_id}/logs/tail",
    summary="Tail llama.cpp Runtime Instance Logs",
    response_model=LlamaCppLogTailResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def tail_llamacpp_instance_logs_endpoint(
    profile_id: str,
    lines: int = Query(default=200, ge=1, le=1000),
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLogTailResponse:
    supervisor = _resolve_llamacpp_supervisor(llm_manager)
    try:
        return await run_in_threadpool(supervisor.tail_logs, profile_id, lines)
    except Exception as e:
        raise _supervisor_error_to_http(e, llm_manager, "Unexpected error tailing Llama.cpp instance logs") from e


@router.post(
    "/llamacpp/start-by-model",
    summary="Start llama.cpp Server by Inventory Model ID",
    response_model=LlamaCppStartByModelResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def start_llamacpp_by_model_endpoint(
    payload: LlamaCppStartByModelRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppStartByModelResponse:
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is not None:
        try:
            runtime = await supervisor.start_default_by_model(payload.model_id, payload.server_args)
            return _start_by_model_response(runtime, payload.model_id)
        except Exception as e:
            raise _supervisor_error_to_http(e, llm_manager, "Unexpected error starting Llama.cpp default profile") from e

    try:
        target = _resolve_llamacpp_target(llm_manager, ("start_server_by_path",))
        model_path = llamacpp_inventory_service.resolve_model_id(payload.model_id)
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except (ModelNotFoundError, ServerError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error resolving Llama.cpp model ID")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e

    try:
        result = await target.start_server_by_path(
            model_path,
            model_label=model_path.name,
            server_args=payload.server_args,
        )
        if isinstance(result, dict):
            result.setdefault("status", "started")
            result["backend"] = "llamacpp"
            result["model_id"] = payload.model_id
        return result
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except ServerError as e:
        _log_sanitized_manager_error(llm_manager, "Failed to start Llama.cpp server by model ID")
        raise HTTPException(status_code=400, detail="Failed to start llama.cpp server for the selected model.") from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error starting Llama.cpp server by model ID")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.post(
    "/llamacpp/use-in-chat",
    summary="Use Managed llama.cpp Server in Chat",
    response_model=LlamaCppUseInChatResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def use_llamacpp_in_chat_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppUseInChatResponse:
    try:
        supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
        if supervisor is not None:
            return await _use_default_runtime_in_chat(supervisor)
        return await llamacpp_provider_service.use_managed_server_in_chat(llm_manager)
    except llamacpp_provider_service.ManagedServerNotRunningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except llamacpp_provider_service.ProviderConfigWriteError as e:
        raise HTTPException(status_code=500, detail="Failed to update llama.cpp chat provider endpoint.") from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error wiring Llama.cpp provider endpoint")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.post(
    "/llamacpp/start_server",
    summary="Start or Swap Llama.cpp Server Model",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def start_llamacpp_server_endpoint(
    model_filename: str = Body(
        ..., embed=True, description="Filename of the GGUF model to load (e.g., 'mistral-7b-v0.1.Q4_K_M.gguf')"
    ),
    server_args: Optional[dict[str, Any]] = Body(
        {}, embed=True, description="Optional Llama.cpp server arguments (e.g., port, n_gpu_layers)"
    ),
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
):
    """
    Starts the Llama.cpp server with the specified model.
    If a server is already running, it will be stopped and restarted with the new model (model swap).
    """
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is not None:
        try:
            requested = Path(model_filename)
            if requested.is_absolute() or ".." in requested.parts:
                raise ServerError("Model filename must be a relative filename under the configured models directory.")
            model_path = Path(supervisor.config.models_dir) / requested
            runtime = await supervisor.start_default_by_path(
                model_path,
                dict(server_args or {}),
                model_label=model_filename,
            )
            return _start_by_path_response(runtime, model_filename)
        except Exception as e:
            raise _supervisor_error_to_http(e, llm_manager, "Unexpected error starting Llama.cpp default profile") from e

    try:
        target = _resolve_llamacpp_target(llm_manager, ("start_server",))
        # Prefer handler.start_server if available, else manager.start_server
        if isinstance(target, LlamaCppHandler):
            result = await target.start_server(model_filename=model_filename, server_args=server_args)
        else:
            result = await target.start_server(backend="llamacpp", model_name=model_filename, server_args=server_args)
        if isinstance(result, dict):
            result.setdefault("status", "started")
            result.setdefault("backend", "llamacpp")
        return result
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except (ModelNotFoundError, ServerError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error starting Llama.cpp server")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.post(
    "/llamacpp/stop_server",
    summary="Stop Llama.cpp Server",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def stop_llamacpp_server_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is not None:
        try:
            runtime = await supervisor.stop_default()
            return {"status": "stopped", "message": runtime.message or "Stopped", "backend": "llamacpp"}
        except LlamaCppProfileNotFoundError:
            return {
                "status": "stopped",
                "message": "No managed llama.cpp server is currently running.",
                "backend": "llamacpp",
            }
        except Exception as e:
            raise _supervisor_error_to_http(e, llm_manager, "Unexpected error stopping Llama.cpp default profile") from e

    try:
        target = _resolve_llamacpp_target(llm_manager, ("stop_server",))
        if isinstance(target, LlamaCppHandler):
            result = await target.stop_server()
        else:
            result = await target.stop_server(backend="llamacpp")
        return {"status": "stopped", "message": result, "backend": "llamacpp"}
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error stopping Llama.cpp server")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/status",
    summary="Get Llama.cpp Server Status",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_status_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is not None:
        try:
            return supervisor.default_status_compat()
        except Exception as e:
            raise _supervisor_error_to_http(e, llm_manager, "Unexpected error getting Llama.cpp default status") from e

    try:
        target = _resolve_llamacpp_target(llm_manager, ("get_server_status",))
        if isinstance(target, LlamaCppHandler):
            status = await target.get_server_status()
        else:
            status = await target.get_server_status(backend="llamacpp")  # Via manager
        if isinstance(status, dict):
            status.setdefault("backend", "llamacpp")
        return status
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llama.cpp server status")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/logs/tail",
    summary="Tail Managed llama.cpp Logs",
    response_model=LlamaCppLogTailResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def tail_llamacpp_logs_endpoint(
    lines: int = Query(default=200, ge=1, le=1000),
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppLogTailResponse:
    supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
    if supervisor is not None:
        try:
            return await run_in_threadpool(supervisor.tail_logs, DEFAULT_PROFILE_ID, lines)
        except LlamaCppProfileNotFoundError as e:
            raise HTTPException(status_code=409, detail="Managed llama.cpp server is not running.") from e
        except Exception as e:
            raise _supervisor_error_to_http(e, llm_manager, "Unexpected error tailing Llama.cpp default logs") from e

    try:
        return await llamacpp_provider_service.tail_managed_log(llm_manager, requested_lines=lines)
    except llamacpp_provider_service.ManagedServerNotRunningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error tailing Llama.cpp logs")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/hardware",
    summary="Get llama.cpp Hardware Snapshot",
    response_model=LlamaCppHardwareSnapshotResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_hardware_endpoint(
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
) -> LlamaCppHardwareSnapshotResponse:
    _ = llm_manager
    return llamacpp_hardware_service.get_hardware_snapshot()


@router.get(
    "/llamacpp/metrics",
    summary="Get Llama.cpp Metrics",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_metrics_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        handler = _resolve_llamacpp_target(llm_manager, ("get_metrics",))
        metrics = handler.get_metrics()
        if inspect.isawaitable(metrics):
            metrics = await metrics
        if isinstance(metrics, dict):
            metrics.setdefault("backend", "llamacpp")
        return metrics
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llama.cpp metrics")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get("/llamafile/metrics", summary="Get Llamafile Metrics")
async def get_llamafile_metrics_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        if not getattr(llm_manager, "llamafile", None):
            raise HTTPException(status_code=400, detail="Llamafile backend is not enabled or configured.")
        handler = llm_manager.llamafile
        if hasattr(handler, "get_metrics"):
            return handler.get_metrics()  # type: ignore[attr-defined]
        return {"message": "metrics not available"}
    except HTTPException:
        raise
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llamafile metrics")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/models",
    summary="List available Llama.cpp models",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def list_llamacpp_models_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        handler = getattr(llm_manager, "llamacpp", None)
        if handler is not None and hasattr(handler, "list_models"):
            models = await handler.list_models()
        elif hasattr(llm_manager, "list_local_models"):
            models = await llm_manager.list_local_models(backend="llamacpp")
        else:
            raise _llamacpp_unavailable()
        return {"available_models": models, "backend": "llamacpp"}
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error listing Llama.cpp models")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


from tldw_Server_API.app.api.v1.schemas.llamacpp_schemas import LlamaCppInferenceRequest


@router.post("/llamacpp/inference", summary="Run inference with Llama.cpp")
async def run_llamacpp_inference_endpoint(
    payload: LlamaCppInferenceRequest, llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)
):
    """
    Runs inference using the currently loaded Llama.cpp model.
    Payload should be OpenAI compatible (e.g., include 'messages' list).
    Example: {"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.7}
    """
    try:
        supervisor = getattr(llm_manager, "llamacpp_supervisor", None)
        if supervisor is not None:
            try:
                runtime = supervisor.get_runtime(DEFAULT_PROFILE_ID)
            except LlamaCppProfileNotFoundError:
                runtime = None
            if runtime is not None and runtime.state == LlamaCppRuntimeState.RUNNING:
                result = await _post_supervisor_runtime_inference(runtime, payload)
                result.setdefault("model", runtime.model_path or runtime.model_id or "unknown_active_model")
                result.setdefault("backend", "llamacpp")
                return result

        handler = getattr(llm_manager, "llamacpp", None)
        # Prefer handler methods when available; fallback to manager for compatibility with tests
        if handler and hasattr(handler, "get_server_status") and hasattr(handler, "inference"):
            status = await handler.get_server_status()
            current_model = status.get("model", "unknown_active_model")
            result = await handler.inference(
                prompt=None,  # Assuming payload contains 'messages'
                messages=payload.messages,
                **payload.to_kwargs(),
            )
            # Align response model naming with manager-style return
            result.setdefault("model", current_model)
        elif hasattr(llm_manager, "get_server_status") and hasattr(llm_manager, "run_inference"):
            status = await llm_manager.get_server_status(backend="llamacpp")
            current_model = status.get("model", "unknown_active_model")
            result = await llm_manager.run_inference(
                backend="llamacpp",
                model_name_or_path=current_model,  # Contextual
                prompt=None,  # Assuming payload contains 'messages'
                **payload.to_kwargs(),  # Pass validated payload as kwargs (extras allowed)
            )
        else:
            raise _llamacpp_unavailable()
        if isinstance(result, dict):
            result.setdefault("backend", "llamacpp")
        return result
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error during Llama.cpp inference")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


# --- Llama.cpp Reranker (GGUF embeddings) ---


class LlamaCppRerankItem(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional identifier for the passage")
    text: str = Field(..., min_length=1, description="Passage text to score")


class LlamaCppRerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query to rank against passages")
    passages: list[LlamaCppRerankItem] = Field(..., min_length=1, description="Candidate passages to rerank")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="Top-K results to return (defaults to len(passages))"
    )
    # Optional overrides for llama.cpp and model selection
    model: Optional[str] = Field(default=None, description="GGUF model path (overrides config)")
    binary: Optional[str] = Field(default=None, description="llama-embedding binary name or path")
    ngl: Optional[int] = Field(default=None, ge=0, description="n_gpu_layers (-ngl)")
    separator: Optional[str] = Field(default=None, description="Text separator between query and passages")
    output_format: Optional[str] = Field(default=None, description="Embedding output format (e.g., json+)")
    pooling: Optional[str] = Field(default=None, description="Embedding pooling method (e.g., last)")
    normalize: Optional[int] = Field(default=None, description="Embedding normalize flag (-1, 0, 1)")
    max_doc_chars: Optional[int] = Field(default=None, ge=0, description="Max chars per passage (truncation)")
    # OpenAPI example
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What do llamas eat?",
                    "passages": [
                        {"id": "a", "text": "Llamas eat bananas"},
                        {"id": "b", "text": "Llamas in pyjamas"},
                        {"id": "c", "text": "A bowl of fruit salad"},
                    ],
                    "top_k": 2,
                    "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
                    "ngl": 99,
                    "separator": "<#sep#>",
                    "output_format": "json+",
                    "pooling": "last",
                    "normalize": -1,
                }
            ]
        }
    )


class LlamaCppRerankResult(BaseModel):
    id: Optional[str] = Field(default=None)
    index: int = Field(..., description="Index of the passage in input list")
    score: float = Field(..., ge=0.0, le=1.0)
    text: Optional[str] = Field(default=None, description="Original passage text (truncated)")


class LlamaCppRerankResponse(BaseModel):
    results: list[LlamaCppRerankResult]


@router.post(
    "/llamacpp/reranking",
    summary="Rerank passages with llama.cpp embeddings (GGUF)",
    response_model=LlamaCppRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
@router.post(
    "/llamacpp/rerank",
    summary="Rerank passages with llama.cpp embeddings (GGUF)",
    response_model=LlamaCppRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def llamacpp_reranker_endpoint(payload: LlamaCppRerankRequest, current_user: User = Depends(get_request_user)):
    """
    Rerank passages using the llama.cpp embeddings binary (llama-embedding) with a GGUF embedding model
    such as Qwen3-Embedding-0.6B. Scores are cosine(query, passage) normalized to [0,1].
    """
    try:
        # Lazy imports to avoid extra startup cost
        from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
            RerankingConfig,
            RerankingStrategy,
            create_reranker,
        )
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to initialize reranking") from e

    # Build documents from passages
    documents: list[Document] = []
    for i, item in enumerate(payload.passages):
        documents.append(
            Document(
                id=item.id or str(i),
                content=item.text,
                metadata={"source": "llamacpp_reranker"},
                source=DataSource.MEDIA_DB,
                score=0.0,
            )
        )

    # Config and overrides
    cfg = RerankingConfig(
        strategy=RerankingStrategy.LLAMA_CPP,
        top_k=min(payload.top_k or len(documents), len(documents)) if documents else 0,
        model_name=payload.model,
    )
    if payload.binary is not None:
        cfg.llama_binary = payload.binary
    if payload.ngl is not None:
        cfg.llama_ngl = payload.ngl
    if payload.separator is not None:
        cfg.llama_embd_separator = payload.separator
    if payload.output_format is not None:
        cfg.llama_embd_output_format = payload.output_format
    if payload.pooling is not None:
        cfg.llama_pooling = payload.pooling
    if payload.normalize is not None:
        cfg.llama_normalize = payload.normalize
    if payload.max_doc_chars is not None:
        cfg.llama_max_doc_chars = payload.max_doc_chars

    reranker = create_reranker(RerankingStrategy.LLAMA_CPP, cfg)

    # Execute reranking
    try:
        # Support both async and sync reranker implementations
        rerank_fn = getattr(reranker, "rerank", None)
        if rerank_fn is None:
            raise RuntimeError("Invalid reranker: missing rerank() method")
        scored = rerank_fn(payload.query, documents)
        if hasattr(scored, "__await__"):
            scored = await scored
        # Enforce top_k even if underlying reranker returns more
        if isinstance(scored, list) and cfg.top_k:
            scored = scored[: cfg.top_k]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rerank passages") from e

    # Convert results
    # Map back to original order indices
    id_to_index = {(p.id or str(i)): i for i, p in enumerate(payload.passages)}
    results: list[LlamaCppRerankResult] = []
    for sd in scored:
        pid = getattr(sd.document, "id", None)
        idx = id_to_index.get(pid, 0)
        results.append(
            LlamaCppRerankResult(
                id=pid,
                index=idx,
                score=float(getattr(sd, "rerank_score", 0.0)),
                text=getattr(sd.document, "content", None),
            )
        )

    return LlamaCppRerankResponse(results=results)


# Public aliases matching common reranker API shapes
public_router = APIRouter()


class PublicRerankRequest(BaseModel):
    model: Optional[str] = Field(
        default=None, description="Reranker model id/path (GGUF for llama.cpp or HF id for transformers)"
    )
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1, description="Documents (plain text) to rank")
    top_n: Optional[int] = Field(default=None, ge=1, le=100)
    backend: Optional[str] = Field(default="auto", description="Reranker backend: auto|llamacpp|transformers")
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
                    "query": "What is panda?",
                    "top_n": 3,
                    "documents": [
                        "hi",
                        "it is a bear",
                        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear ...",
                    ],
                }
            ]
        }
    )


class PublicRerankResponse(BaseModel):
    results: list[LlamaCppRerankResult]


async def _run_public_rerank(
    query: str, docs: list[str], model_override: Optional[str], top_k: Optional[int], backend: str
) -> list[LlamaCppRerankResult]:
    from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
        RerankingConfig,
        RerankingStrategy,
        create_reranker,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    documents: list[Document] = [
        Document(id=str(i), content=txt, metadata={"source": "public_reranking"}, source=DataSource.MEDIA_DB)
        for i, txt in enumerate(docs)
    ]

    # Select backend
    strategy = RerankingStrategy.FLASHRANK
    model_name = model_override
    b = (backend or "auto").lower()
    is_gguf = bool(model_override and model_override.strip().lower().endswith(".gguf"))
    looks_hf_id = bool(model_override and "/" in model_override and not is_gguf)
    if b == "llamacpp" or is_gguf:
        strategy = RerankingStrategy.LLAMA_CPP
    elif b == "transformers" or looks_hf_id:
        strategy = RerankingStrategy.CROSS_ENCODER
    else:
        # Auto fallback: prefer transformers if it looks like HF id, else llama if gguf
        strategy = (
            RerankingStrategy.LLAMA_CPP
            if is_gguf
            else (RerankingStrategy.CROSS_ENCODER if looks_hf_id else RerankingStrategy.FLASHRANK)
        )

    cfg = RerankingConfig(
        strategy=strategy,
        top_k=min(top_k or len(documents), len(documents)) if documents else 0,
        model_name=model_name,
    )
    reranker = create_reranker(strategy, cfg)
    # Support both async and sync reranker implementations
    rerank_fn = getattr(reranker, "rerank", None)
    if rerank_fn is None:
        raise HTTPException(status_code=500, detail="Invalid reranker: missing rerank() method")
    scored = rerank_fn(query, documents)
    if hasattr(scored, "__await__"):
        scored = await scored
    # Enforce top_k even if underlying reranker returns more
    if isinstance(scored, list) and cfg.top_k:
        scored = scored[: cfg.top_k]
    out: list[LlamaCppRerankResult] = []
    for sd in scored:
        idx = int(getattr(sd.document, "id", "0")) if str(getattr(sd.document, "id", "0")).isdigit() else 0
        out.append(
            LlamaCppRerankResult(
                id=getattr(sd.document, "id", None),
                index=idx,
                score=float(getattr(sd, "rerank_score", 0.0)),
                text=getattr(sd.document, "content", None),
            )
        )
    return out


@public_router.post(
    "/v1/reranking",
    summary="Rerank documents according to a query",
    response_model=PublicRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
@public_router.post(
    "/v1/rerank",
    summary="Rerank documents according to a query",
    response_model=PublicRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def public_reranking_endpoint(payload: PublicRerankRequest, current_user: User = Depends(get_request_user)):
    try:
        results = await _run_public_rerank(
            query=payload.query,
            docs=payload.documents,
            model_override=payload.model,
            top_k=payload.top_n,
            backend=(payload.backend or "auto"),
        )
        return PublicRerankResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rerank documents") from e


#
# End of llamacpp.py
##########################################################################################################################
