"""Bounded transfer observation over an acquisition's existing CDP connection."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .article_models import ArticleFailure


class ChildSession:
    """The small CDPSession port used by existing request and transfer handlers."""

    def __init__(self, owner: ChildSessions, identifier: str, parent: Any) -> None:
        self.owner, self.identifier, self.parent = owner, identifier, parent
        self.target: str | None = None
        self.handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> None:
        self.handlers.setdefault(event, []).append(handler)

    async def send(self, method: str, params: dict | None = None) -> Any:
        return await self.owner.command(self.identifier, method, params or {})

    async def detach(self) -> None:
        self.owner.disposed(self.identifier)
        await self.parent.send("Target.detachFromTarget", {"sessionId": self.identifier})


class ChildSessions:
    """No connection/endpoint creation; commands retain the existing browser lease."""

    def __init__(self, root: Any, lease: Any, fail: Callable[[], None]) -> None:
        self.root, self.lease, self.fail = root, lease, fail
        self.sessions: dict[str, ChildSession] = {}
        self.pending: dict[int, tuple[str, asyncio.Future]] = {}
        self.sequence = 0
        root.on("Target.receivedMessageFromTarget", self.received)
        root.on("Target.detachedFromTarget", lambda event: self.disposed(event["sessionId"]))

    async def attach(self, target: str) -> ChildSession:
        if len(self.sessions) >= 63:
            raise ArticleFailure("browser_error", "capacity")
        result = await self.root.send("Target.attachToTarget", {"targetId": target, "flatten": False})
        session = self.adopt(result["sessionId"], self.root)
        session.target = target
        return session

    def adopt(self, identifier: str, parent: Any) -> ChildSession:
        if len(self.sessions) >= 63 or len(identifier) > 1024:
            raise ArticleFailure("browser_error", "capacity")
        session = ChildSession(self, identifier, parent)
        self.sessions[session.identifier] = session
        return session

    async def command(self, identifier: str, method: str, params: dict) -> Any:
        task = await self.forward(identifier, method, params)
        return await asyncio.shield(task)

    async def forward(self, identifier: str, method: str, params: dict) -> asyncio.Task:
        if len(self.pending) >= self.lease._callback_capacity:
            raise ArticleFailure("browser_error", "capacity")
        self.sequence += 1
        ident = self.sequence
        message = json.dumps({"id": ident, "method": method, "params": params})
        if len(message.encode("utf-8")) > 262_144:
            raise ArticleFailure("browser_error", "capacity")
        response = asyncio.get_running_loop().create_future()
        response.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self.pending[ident] = identifier, response

        async def wait_for_reply() -> Any:
            try:
                return await response
            finally:
                self.pending.pop(ident, None)

        task, retention = self.lease.start_cleanup(wait_for_reply(), name="article-browser-child-command")
        if task is None:
            self.pending.pop(ident, None)
            response.cancel()
            raise ArticleFailure("browser_error", "callback")
        if retention == "unobserved":
            task.cancel()
            self.pending.pop(ident, None)
            response.cancel()
            raise ArticleFailure("browser_error", "callback")
        try:
            # Forward before network continuation; the native reply may need navigation.
            await self.sessions[identifier].parent.send(
                "Target.sendMessageToTarget",
                {
                    "sessionId": identifier,
                    "message": message,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - native send failure settles the owned reply without payloads
            if not response.done():
                response.set_exception(ArticleFailure("browser_error", "callback"))
            raise
        return task

    def received(self, event: dict) -> None:
        try:
            session = self.sessions.get(event["sessionId"])
            if session is None:
                return
            raw = event["message"]
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > 262_144:
                raise ValueError("oversized browser event")
            message = json.loads(raw)
            if "id" in message:
                entry = self.pending.get(message["id"])
                if entry is not None and entry[0] == session.identifier and not entry[1].done():
                    if "error" in message:
                        entry[1].set_exception(ArticleFailure("browser_error", "capability"))
                    else:
                        entry[1].set_result(message.get("result", {}))
            elif "method" in message:
                for handler in session.handlers.get(message["method"], ()):
                    handler(message.get("params", {}))
        except Exception:  # noqa: BLE001 - malformed native envelopes fail closed without echoing payloads
            self.fail()

    def disposed(self, identifier: str) -> None:
        session = self.sessions.pop(identifier, None)
        if session is not None:
            session.handlers.clear()
        for session, future in self.pending.values():
            if session == identifier and not future.done():
                future.set_exception(ArticleFailure("browser_error", "callback"))


class TransferSessions:
    """Observe only this context's targets; never coordinate renderer startup."""

    def __init__(self, root: Any, lease: Any, ledger: Any, fail: Callable[[], None]) -> None:
        self.root, self.ledger, self.fail = root, ledger, fail
        self.ports = ChildSessions(root, lease, fail)
        self.context_id: str | None = None
        self.main_target: str | None = None
        self.targets: dict[str, asyncio.Future] = {}
        self.closed = False

    async def track(self) -> None:
        if self.closed or self.context_id is None:
            raise ArticleFailure("browser_error", "callback")
        targets = (await self.root.send("Target.getTargets"))["targetInfos"]
        if not isinstance(targets, list) or len(targets) > 64:
            raise ArticleFailure("browser_error", "capacity")
        for target in targets:
            await self.created({"targetInfo": target})
        for ready in tuple(self.targets.values()):
            # The setup result is an owned native-ack task, not its acknowledgement.
            await asyncio.shield(ready)

    async def created(self, event: dict) -> None:
        if self.closed or self.context_id is None:
            return
        target = event["targetInfo"]
        if target.get("browserContextId") != self.context_id or target.get("type") not in {
            "page",
            "iframe",
            "worker",
            "shared_worker",
        }:
            return
        identifier = target["targetId"]
        if identifier != self.main_target and identifier not in self.targets:
            await self._attach(identifier)

    async def _attach(self, identifier: str) -> None:
        if not isinstance(identifier, str) or len(identifier) > 1024 or len(self.targets) >= 63:
            raise ArticleFailure("browser_error", "capacity")
        ready = asyncio.get_running_loop().create_future()
        ready.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self.targets[identifier] = ready
        try:
            session = await self.ports.attach(identifier)
            session.on("Network.dataReceived", lambda event: self.ledger.on_request_data(session.identifier, event))
            session.on(
                "Network.loadingFinished", lambda event: self.ledger.on_request_finished(session.identifier, event)
            )
            session.on("Network.loadingFailed", lambda event: self.ledger.on_request_failed(session.identifier, event))
            session.on("Network.webSocketFrameReceived", self.ledger.on_websocket_frame)
            session.on("Network.webSocketFrameSent", self.ledger.on_websocket_frame)
            acknowledgement = await self.ports.forward(session.identifier, "Network.enable", {})
            acknowledgement.add_done_callback(self._acknowledged)
            ready.set_result(acknowledgement)
        except BaseException as error:  # noqa: BLE001 - settle shared setup, never detach its native work
            if not ready.done():
                ready.cancel() if isinstance(error, asyncio.CancelledError) else ready.set_exception(error)
            raise

    def _acknowledged(self, task: asyncio.Task) -> None:
        if not self.closed and (task.cancelled() or task.exception() is not None):
            self.fail()

    async def acknowledged(self, timeout: float) -> None:
        tasks = [ready.result() for ready in self.targets.values() if ready.done() and not ready.cancelled()]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending or any(task.cancelled() or task.exception() is not None for task in done):
                raise ArticleFailure("browser_error", "capability")

    async def detach(self) -> None:
        self.closed = True
        active = None
        if self.ports.sessions:
            targets = (await self.root.send("Target.getTargets"))["targetInfos"]
            if not isinstance(targets, list) or len(targets) > 64:
                raise ArticleFailure("browser_error", "capacity")
            active = {target["targetId"] for target in targets}
        for session in tuple(self.ports.sessions.values()):
            if session.target is not None and session.target not in active:
                # Context close can precede delivery of detachedFromTarget.
                self.ports.disposed(session.identifier)
            elif session.identifier in self.ports.sessions:
                await session.detach()
        self.targets.clear()

    def retire(self, *, native_browser_disconnected: bool) -> None:
        """Settle local native replies only behind the acquisition's owner-exit fence."""
        if native_browser_disconnected is not True:
            raise ArticleFailure("browser_error", "cleanup")
        self.closed = True
        for identifier in tuple(self.ports.sessions):
            self.ports.disposed(identifier)
        self.targets.clear()
