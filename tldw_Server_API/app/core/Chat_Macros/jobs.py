"""Jobs integration for chat macro execution."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.Chat.command_router import reserved_core_command_names
from tldw_Server_API.app.core.Chat_Macros.branch_runner import ChatMacroLLMBranchRunner
from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError
from tldw_Server_API.app.core.Chat_Macros.executor import ChatMacroExecutor, MacroExecutorSettings
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chat_Macros.service import ChatMacrosService
from tldw_Server_API.app.core.Chat_Macros.storage import ChatMacroStorage
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager

CHAT_MACROS_DOMAIN = "chat_macros"
CHAT_MACROS_JOB_TYPE = "chat_macro_run"
_POST_BACK_SCAN_LIMIT = 500


def chat_macro_jobs_queue() -> str:
    """Return the configured chat macro Jobs queue."""

    return (os.getenv("CHAT_MACROS_JOBS_QUEUE") or "default").strip() or "default"


def enqueue_chat_macro_run_job(
    *,
    macro_run_id: str,
    user_id: str,
    macro_digest: str | None,
    normalized_args: dict[str, Any],
    job_manager: JobManager | None = None,
    priority: int = 5,
) -> dict[str, Any]:
    """Enqueue a background job for one stored chat macro run."""

    jm = job_manager or JobManager()
    run_id = str(macro_run_id)
    owner_user_id = str(user_id)
    payload = {
        "macro_run_id": run_id,
        "user_id": owner_user_id,
        "macro_digest": macro_digest,
        "normalized_args": dict(normalized_args or {}),
    }
    return jm.create_job(
        domain=CHAT_MACROS_DOMAIN,
        queue=chat_macro_jobs_queue(),
        job_type=CHAT_MACROS_JOB_TYPE,
        payload=payload,
        owner_user_id=owner_user_id,
        priority=priority,
        max_retries=1,
        idempotency_key=f"chat_macro_run:{run_id}",
    )


async def handle_chat_macro_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one acquired chat macro job."""

    payload = _validated_payload(job)
    run_id = str(payload["macro_run_id"])
    user_id = str(payload["user_id"])
    user_id_int = int(user_id)
    chat_db = None
    try:
        chat_db = await get_chacha_db_for_user_id(
            user_id_int,
            client_id=f"chat-macro-worker-{user_id_int}",
        )
        executor = build_chat_macro_executor(chat_db=chat_db, user_id=user_id)
        run = await executor.execute_run(run_id)
        return {"macro_run_id": run.run_id, "status": run.status}
    finally:
        _close_worker_database(chat_db)


async def should_cancel_chat_macro_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
    repository: ChatMacroRepository | None = None,
) -> bool:
    """Return true when a macro job should stop and mirror cancellation to the run."""

    jm = job_manager or JobManager()
    job_id = int(job["id"])
    current = jm.get_job(job_id)
    if not current:
        return False

    status = str(current.get("status") or "").strip().lower()
    cancel_requested = bool(current.get("cancel_requested_at")) or status == "cancelled"
    if not cancel_requested:
        return False

    reason = str(current.get("cancellation_reason") or "requested")
    payload = job.get("payload") or current.get("payload") or {}
    run_id = str(payload.get("macro_run_id") or "").strip()
    if run_id:
        await _mark_macro_run_cancelled(run_id, payload=payload, repository=repository)

    if current.get("cancel_requested_at"):
        try:
            jm.finalize_cancelled(job_id, reason=reason)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Chat macro job cancellation finalize skipped for job {}: {}", job_id, exc)
    return True


def build_chat_macro_executor(
    *,
    chat_db: Any,
    user_id: str,
    branch_runner: Any | None = None,
    settings: MacroExecutorSettings | None = None,
) -> ChatMacroExecutor:
    """Build the executor used by the Jobs worker."""

    repository = ChatMacroRepository(chat_db)
    repository.ensure_ready()
    service = ChatMacrosService(
        user_id=str(user_id),
        storage=ChatMacroStorage(DatabasePaths.get_user_base_directory(user_id)),
        repository=repository,
        core_commands=reserved_core_command_names(),
    )
    output_settings = settings or MacroExecutorSettings(
        output_profiles=_output_profiles_from_service(service),
    )
    runner = branch_runner or ChatMacroLLMBranchRunner()

    def _macro_loader(run: Any) -> Any:
        return _load_macro_definition_for_run(service, run)

    def _post_back(*, run_id: str, final_output: str, post_idempotency_key: str) -> str:
        return post_chat_macro_final_output(
            chat_db=chat_db,
            repository=repository,
            run_id=run_id,
            final_output=final_output,
            post_idempotency_key=post_idempotency_key,
        )

    return ChatMacroExecutor(
        repository=repository,
        macro_loader=_macro_loader,
        branch_runner=runner,
        settings=output_settings,
        post_back=_post_back,
    )


def post_chat_macro_final_output(
    *,
    chat_db: Any,
    repository: ChatMacroRepository,
    run_id: str,
    final_output: str,
    post_idempotency_key: str,
) -> str:
    """Persist a final macro output as a visible assistant message, idempotently."""

    run = repository.get_run(run_id)
    if run is None:
        raise MacroStorageError(f"macro run not found: {run_id}")
    if not run.conversation_id:
        return ""

    existing = _find_existing_macro_post(
        chat_db,
        conversation_id=run.conversation_id,
        post_idempotency_key=post_idempotency_key,
    )
    if existing:
        return existing

    content = final_output if final_output else "Macro completed with no output."
    message_id = chat_db.add_message(
        {
            "conversation_id": run.conversation_id,
            "sender": "assistant",
            "content": content,
            "client_id": f"chat_macro:{run.user_id}",
        }
    )
    if not message_id:
        raise MacroStorageError("failed to persist chat macro final message")

    metadata = {
        "chat_macro": {
            "run_id": run.run_id,
            "name": run.macro_name,
            "command": run.macro_command,
            "status": "completed",
            "detail_url": f"/api/v1/chat/macros/runs/{run.run_id}",
            "output_profile": run.output_profile,
            "post_idempotency_key": post_idempotency_key,
        }
    }
    if not chat_db.add_message_metadata(str(message_id), extra=metadata):
        raise MacroStorageError(f"failed to persist chat macro metadata for message {message_id}")
    return str(message_id)


def _validated_payload(job: dict[str, Any]) -> dict[str, Any]:
    domain = str(job.get("domain") or CHAT_MACROS_DOMAIN).strip()
    job_type = str(job.get("job_type") or CHAT_MACROS_JOB_TYPE).strip()
    if domain != CHAT_MACROS_DOMAIN or job_type != CHAT_MACROS_JOB_TYPE:
        raise ValueError("unsupported_chat_macro_job")

    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("invalid_chat_macro_job_payload")

    run_id = str(payload.get("macro_run_id") or "").strip()
    user_id = str(payload.get("user_id") or job.get("owner_user_id") or "").strip()
    if not run_id or not user_id:
        raise ValueError("invalid_chat_macro_job_payload")
    normalized_args = payload.get("normalized_args") or {}
    if not isinstance(normalized_args, dict):
        raise ValueError("invalid_chat_macro_job_payload")
    return {
        "macro_run_id": run_id,
        "user_id": user_id,
        "macro_digest": payload.get("macro_digest"),
        "normalized_args": normalized_args,
    }


async def _mark_macro_run_cancelled(
    run_id: str,
    *,
    payload: dict[str, Any],
    repository: ChatMacroRepository | None,
) -> None:
    owned_db = None
    repo = repository
    try:
        if repo is None:
            raw_user_id = str(payload.get("user_id") or "").strip()
            if not raw_user_id:
                logger.debug("Cannot mirror chat macro cancellation for {} without user_id.", run_id)
                return
            user_id = int(raw_user_id)
            owned_db = await get_chacha_db_for_user_id(
                user_id,
                client_id=f"chat-macro-cancel-{user_id}",
            )
            repo = ChatMacroRepository(owned_db)
            repo.ensure_ready()
        with suppress(MacroStorageError):
            repo.request_cancel(run_id)
        with suppress(MacroStorageError):
            repo.update_run_status(
                run_id,
                status="cancelled",
                error_code="cancelled",
                error_message="Macro job was cancelled before execution completed.",
            )
    finally:
        _close_worker_database(owned_db)


def _load_macro_definition_for_run(service: ChatMacrosService, run: Any) -> Any:
    digest = str(getattr(run, "macro_digest", "") or "")
    for item in service.list_macros():
        if digest and item.digest == digest:
            return item.definition
    name = str(getattr(run, "macro_name", "") or "")
    if name:
        item = service.get_macro(name)
        if not digest or item.digest == digest:
            return item.definition
    raise MacroStorageError("macro definition for run is no longer available")


def _output_profiles_from_service(service: ChatMacrosService) -> dict[str, Any]:
    settings = service.get_settings()
    raw_profiles = settings.get("output_profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        return {"default": service.resolve_output_profile("default")}
    profiles: dict[str, Any] = {}
    for name in raw_profiles:
        try:
            profiles[str(name)] = service.resolve_output_profile(str(name))
        except Exception as exc:  # noqa: BLE001 - invalid user profiles should not kill worker startup
            logger.debug("Skipping invalid chat macro output profile {}: {}", name, exc)
    profiles.setdefault("default", service.resolve_output_profile("default"))
    return profiles


def _find_existing_macro_post(
    chat_db: Any,
    *,
    conversation_id: str,
    post_idempotency_key: str,
) -> str | None:
    messages = chat_db.get_messages_for_conversation(
        conversation_id,
        limit=_POST_BACK_SCAN_LIMIT,
        offset=0,
        order_by_timestamp="DESC",
    )
    assistant_ids = [str(message["id"]) for message in messages if message.get("sender") == "assistant"]
    metadata_by_id = chat_db.get_message_metadata_map(assistant_ids)
    for message_id in assistant_ids:
        metadata = metadata_by_id.get(message_id) or {}
        extra = metadata.get("extra") if isinstance(metadata, dict) else None
        chat_macro = extra.get("chat_macro") if isinstance(extra, dict) else None
        if isinstance(chat_macro, dict) and chat_macro.get("post_idempotency_key") == post_idempotency_key:
            return message_id
    return None


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    try:
        if hasattr(db, "release_context_connection"):
            db.release_context_connection()
            return
        if hasattr(db, "close_connection"):
            db.close_connection()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        logger.debug("Chat macro worker DB cleanup skipped for {}.", type(db).__name__)


__all__ = [
    "CHAT_MACROS_DOMAIN",
    "CHAT_MACROS_JOB_TYPE",
    "ChatMacroLLMBranchRunner",
    "build_chat_macro_executor",
    "chat_macro_jobs_queue",
    "enqueue_chat_macro_run_job",
    "handle_chat_macro_job",
    "post_chat_macro_final_output",
    "should_cancel_chat_macro_job",
]
