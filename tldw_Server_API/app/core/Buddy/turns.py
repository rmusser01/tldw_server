"""Bounded process-owned FIFO turns through the ordinary authenticated Chat API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from tldw_Server_API.app.api.v1.schemas.buddy_turns import BuddyTurnCreate
from tldw_Server_API.app.core.Buddy.publication import BuddyPublication, current_buddy_publication
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.DB_Management.Buddy_DB import BuddyConflictError, BuddyNotFoundError
from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import (
    BuddyPublicationRevokedError,
    BuddyRuntimeBusyError,
    BuddyTurnRepository,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError


class BuddyQueueFullError(RuntimeError):
    """The bounded process queue cannot accept another turn."""


@dataclass
class _PendingTurn:
    repository: BuddyTurnRepository
    row: dict[str, Any]
    payload: dict[str, Any]
    headers: dict[str, str]
    client: tuple[str, int] | None


def _resolve_payload(service: BuddyService, request: BuddyTurnCreate) -> dict[str, Any]:
    wrapped = service.db.get_conversation_settings(request.conversation_id) or {}
    settings = wrapped.get("settings") or {}
    resume = service.db.get_roleplay_resume_state(request.conversation_id, owner_client_id=service.user_id)
    effective = resume.get("effective_completion") or {}
    # This is a configured target, not a catalog or server-default guess.
    provider = request.provider or effective.get("provider") or settings.get("provider")
    model = request.model or effective.get("model") or settings.get("model")
    if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
        raise ValueError("Choose a Chat provider and model before sending")
    payload = {
        "api_provider": provider.strip(),
        "model": model.strip(),
        "conversation_id": request.conversation_id,
        "messages": [{"role": "user", "content": request.text}],
        "stream": False,
        "save_to_db": True,
        "tool_choice": "none",
    }
    sampling = effective.get("sampling") or {}
    for key in ("temperature", "top_p", "repetition_penalty", "stop", "max_tokens"):
        if key in sampling:
            payload[key] = sampling[key]
    return payload


class BuddyTurnRuntime:
    """One owner per principal, with independent bounded conversation queues.

    Bodies and credentials exist only in these retained tasks. A SQL lease fences
    competing processes; an expired owner fails terminal rather than replaying
    provider effects. Other workers can read and Stop work, but new acceptance
    needs owner affinity while this process has the lease.
    """

    def __init__(self, app: Any, *, capacity: int = 64, per_user_capacity: int = 16, timeout: float = 120) -> None:
        self.app = app
        self.owner_id = uuid.uuid4().hex
        self.capacity = capacity
        self.per_user_capacity = per_user_capacity
        self.timeout = timeout
        self.closed = False
        self._queues: dict[tuple[str, str], deque[_PendingTurn]] = {}
        self._workers: dict[tuple[str, str], asyncio.Task] = {}
        self._admissions: set[asyncio.Task] = set()
        self._inflight: dict[str, asyncio.Task] = {}
        self._heartbeats: dict[str, asyncio.Task] = {}
        self._repositories: dict[str, BuddyTurnRepository] = {}
        self._admission_lock = asyncio.Lock()

    async def accept(
        self,
        service: BuddyService,
        client_slot: str,
        request: BuddyTurnCreate,
        headers: dict[str, str],
        client: tuple[str, int] | None,
    ) -> dict[str, Any]:
        """Acceptance owns its task before yielding to a disconnectable caller."""
        if self.closed or len(self._admissions) >= self.capacity:
            raise BuddyQueueFullError("Buddy queue is full")
        task = asyncio.create_task(self._accept(service, client_slot, request, headers, client))
        self._admissions.add(task)
        task.add_done_callback(self._admission_done)
        return await asyncio.shield(task)

    def _admission_done(self, task: asyncio.Task) -> None:
        self._admissions.discard(task)
        if not task.cancelled():
            task.exception()  # Observe failures even if the caller disconnected.

    async def _accept(
        self,
        service: BuddyService,
        client_slot: str,
        request: BuddyTurnCreate,
        headers: dict[str, str],
        client: tuple[str, int] | None,
    ) -> dict[str, Any]:
        async with self._admission_lock:
            repository = BuddyTurnRepository(service.db, service.user_id)
            digest = hashlib.sha256(
                json.dumps({"client_slot": client_slot, **request.model_dump()}, sort_keys=True).encode()
            ).hexdigest()
            existing = await asyncio.to_thread(repository.by_key, request.client_request_id)
            if existing is not None:
                if existing["request_digest"] != digest:
                    raise BuddyConflictError("Request key was used with different input")
                await asyncio.to_thread(repository.expire_interrupted)
                return await asyncio.to_thread(repository.get, existing["id"])
            count = sum(len(queue) for queue in self._queues.values())
            user_count = sum(len(queue) for (owner, _), queue in self._queues.items() if owner == service.user_id)
            if self.closed or count >= self.capacity or user_count >= self.per_user_capacity:
                raise BuddyQueueFullError("Buddy queue is full")

            def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
                attachment = service.attachment(client_slot)
                if attachment["version"] != request.expected_attachment_version:
                    raise BuddyConflictError("Attachment version changed")
                target = attachment["attachment"]
                if target is None:
                    raise BuddyNotFoundError("Attach a Buddy to an available target first")
                resolved = service.resolve_target("conversation", request.conversation_id)
                if (target["scope_type"] == "conversation" and target["scope_id"] != request.conversation_id) or (
                    target["scope_type"] == "workspace" and target["scope_id"] != resolved["workspace_id"]
                ):
                    raise BuddyNotFoundError("Conversation is outside the attached target")
                conversation = service.db.get_conversation_by_id(request.conversation_id)
                payload = _resolve_payload(service, request)
                repository.claim(self.owner_id)
                row = repository.create(
                    {
                        "id": uuid.uuid4().hex,
                        "owner_id": self.owner_id,
                        "client_slot": client_slot,
                        "client_request_id": request.client_request_id,
                        "request_digest": digest,
                        "conversation_id": request.conversation_id,
                        "conversation_title": resolved["title"],
                        "conversation_version": conversation["version"],
                        "workspace_id": resolved["workspace_id"],
                        "attachment_version": attachment["version"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    check_attachment=True,
                )
                return row, payload

            row, payload = await asyncio.to_thread(prepare)
            # No await between the committed admission and registration of its owner.
            key = (service.user_id, request.conversation_id)
            queue = self._queues.setdefault(key, deque())
            queue.append(_PendingTurn(repository, row, payload, headers, client))
            self._repositories[service.user_id] = repository
            if service.user_id not in self._heartbeats:
                self._heartbeats[service.user_id] = asyncio.create_task(self._heartbeat(service.user_id))
            if key not in self._workers:
                self._workers[key] = asyncio.create_task(self._drain(key))
            return row

    async def _heartbeat(self, user_id: str) -> None:
        try:
            while not self.closed and any(owner == user_id for owner, _ in self._queues):
                await asyncio.sleep(5)
                await asyncio.to_thread(self._repositories[user_id].claim, self.owner_id)
        except (BuddyRuntimeBusyError, asyncio.CancelledError):
            # Publication requires the still-current lease; losing it is terminal.
            pass
        finally:
            self._heartbeats.pop(user_id, None)
            if not any(owner == user_id for owner, _ in self._queues):
                self._repositories.pop(user_id, None)

    async def _drain(self, key: tuple[str, str]) -> None:
        queue = self._queues[key]
        try:
            while queue and not self.closed:
                item = queue[0]
                try:
                    await self._execute(item)
                finally:
                    # Clear credentials/body promptly even if the provider failed.
                    item.headers.clear()
                    item.payload.clear()
                    queue.popleft()
        finally:
            self._workers.pop(key, None)
            self._queues.pop(key, None)

    async def _execute(self, item: _PendingTurn) -> None:
        repository, row = item.repository, item.row

        def check_target() -> None:
            with repository.db.transaction() as conn:
                repository.assert_publication(conn, row, row["conversation_id"])

        try:
            current = await asyncio.to_thread(repository.get, row["id"])
            if current["status"] != "queued":
                return
            await asyncio.to_thread(repository.transition, row["id"], "running")
            await asyncio.to_thread(check_target)
            token = current_buddy_publication.set(BuddyPublication(repository, row))
            try:
                transport = httpx.ASGITransport(
                    app=self.app, client=item.client or ("127.0.0.1", 0), raise_app_exceptions=False
                )
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver", headers=item.headers
                ) as internal:
                    dispatch = asyncio.create_task(internal.post("/api/v1/chat/completions", json=item.payload))
                    self._inflight[row["id"]] = dispatch
                    try:
                        response = await asyncio.wait_for(dispatch, self.timeout)
                    finally:
                        self._inflight.pop(row["id"], None)
            finally:
                current_buddy_publication.reset(token)
            if not response.is_success:
                await asyncio.to_thread(check_target)
                await asyncio.to_thread(
                    repository.transition, row["id"], "failed", error_code=f"chat_http_{response.status_code}"
                )
                return
            value = response.json()
            message_id = value.get("tldw_message_id")
            if value.get("tldw_conversation_id") != row["conversation_id"] or not message_id:
                raise BuddyPublicationRevokedError("Chat did not return the exact persisted result")
            await asyncio.to_thread(check_target)
            current = await asyncio.to_thread(repository.get, row["id"])
            if current["result_message_id"] != message_id:
                raise BuddyPublicationRevokedError("Chat result identity did not match the committed message")
            await asyncio.to_thread(repository.transition, row["id"], "completed")
        except BuddyPublicationRevokedError:
            await asyncio.to_thread(repository.transition, row["id"], "failed", error_code="stale_target")
        except asyncio.CancelledError:
            current = await asyncio.to_thread(repository.get, row["id"])
            if current["status"] == "stopped" and not self.closed:
                return
            await asyncio.to_thread(repository.transition, row["id"], "failed", error_code="interrupted_unknown")
            raise
        except (
            TimeoutError,
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            OSError,
            CharactersRAGDBError,
        ):
            await asyncio.to_thread(repository.transition, row["id"], "failed", error_code="completion_unknown")

    def cancel_dispatch(self, turn_id: str) -> None:
        """Called only after SQL revocation; provider cleanup retains its owner."""
        dispatch = self._inflight.get(turn_id)
        if dispatch is not None:
            dispatch.cancel()

    async def close(self) -> None:
        """Revoke work before application shutdown closes shared content databases."""
        self.closed = True
        await asyncio.gather(*tuple(self._admissions), return_exceptions=True)
        for queue in tuple(self._queues.values()):
            for item in tuple(queue):
                await asyncio.to_thread(
                    item.repository.transition, item.row["id"], "failed", error_code="interrupted_unknown"
                )
        tasks = [*self._workers.values(), *self._heartbeats.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
