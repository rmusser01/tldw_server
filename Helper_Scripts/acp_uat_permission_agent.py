"""Controlled ACP UAT worker; all file effects go through the real runner tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

OUTPUT = "C05 approved disposable output\n"
COMMANDS = {"read", "write", "delayed-write", "boundary", "repeat", "fail"}


class Worker:
    """Serve a closed set of harmless recipes, without direct filesystem effects."""

    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.session = "c05-" + uuid.uuid4().hex
        self.pending = {}
        self.sequence = 0
        self.cancelled = asyncio.Event()
        self.active = False

    def send(self, payload):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **payload}) + "\n")
        sys.stdout.flush()

    async def tool(self, method, **params):
        self.sequence += 1
        key = "tool-" + str(self.sequence)
        future = asyncio.get_running_loop().create_future()
        self.pending[key] = future
        stopped = asyncio.create_task(self.cancelled.wait())
        try:
            if self.cancelled.is_set():
                raise asyncio.CancelledError
            self.send({"id": key, "method": method, "params": {"sessionId": self.session, **params}})
            done, _ = await asyncio.wait((future, stopped), timeout=60, return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                raise asyncio.CancelledError
            if future not in done:
                raise TimeoutError("Controlled tool response deadline")
            return future.result()
        finally:
            self.pending.pop(key, None)
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
            if not future.done():
                future.cancel()

    async def prompt(self, params):
        blocks = params.get("prompt", [])
        text = "".join(str(x.get("text", x.get("content", ""))) for x in blocks if isinstance(x, dict))
        command = text.removeprefix("C05 ")
        if not text.startswith("C05 ") or command not in COMMANDS:
            raise ValueError("Use a declared C05 command")
        if command == "fail":
            raise ValueError("Controlled downstream failure")
        if command == "delayed-write":
            try:
                await asyncio.wait_for(self.cancelled.wait(), 10)
                raise asyncio.CancelledError
            except TimeoutError:
                pass
        outcomes = []
        if command == "read":
            outcomes.append(await self.tool("fs/read_text_file", path=str(self.root / "README.md")))
        elif command == "boundary":
            for target in (self.root.parent / "sentinel.txt", self.root / "outside-link", self.root / ".env"):
                outcomes.append(await self.tool("fs/read_text_file", path=str(target)))
                outcomes.append(await self.tool("fs/write_text_file", path=str(target), content=OUTPUT))
        else:
            for _ in range(2 if command == "repeat" else 1):
                outcomes.append(
                    await self.tool("fs/write_text_file", path=str(self.root / "output.txt"), content=OUTPUT)
                )
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        summary = json.dumps({"command": command, "tools": outcomes}, separators=(",", ":"))
        self.send(
            {
                "method": "session/update",
                "params": {
                    "sessionId": self.session,
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": summary}},
                },
            }
        )
        return {"stopReason": "end_turn"}

    async def handle(self, message):
        key, method, params = message.get("id"), message.get("method"), message.get("params") or {}
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": 1,
                    "agentInfo": {"name": "c05-controlled", "version": "1.0.0"},
                    "agentCapabilities": {"loadSession": False},
                    "authMethods": [],
                }
            elif method == "session/new":
                if Path(params.get("cwd", "")).resolve() != self.root:
                    raise ValueError("Controlled workspace mismatch")
                result = {"sessionId": self.session}
            else:
                if params.get("sessionId") != self.session:
                    raise ValueError("Controlled session mismatch")
                if method in {"session/cancel", "session/close", "_tldw/session/close"}:
                    self.cancelled.set()
                    result = None
                elif method == "session/prompt":
                    if self.active:
                        raise ValueError("Controlled prompt already active")
                    self.active = True
                    self.cancelled.clear()
                    try:
                        result = await self.prompt(params)
                    except asyncio.CancelledError:
                        result = {"stopReason": "cancelled"}
                    finally:
                        self.active = False
                else:
                    raise ValueError("Unsupported controlled method")
            if key is not None:
                self.send({"id": key, "result": result})
        except (ValueError, TimeoutError) as exc:
            if key is not None:
                self.send({"id": key, "error": {"code": -32602, "message": str(exc)}})

    async def run(self):
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )
        tasks = set()
        try:
            # TaskGroup propagates a failed handler and cancels the pipe reader.
            async with asyncio.TaskGroup() as group:
                try:
                    while line := await reader.readline():
                        message = json.loads(line)
                        if "method" not in message:
                            future = self.pending.get(message.get("id"))
                            if future is not None and not future.done():
                                future.set_result(message)
                            continue
                        task = group.create_task(self.handle(message))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                finally:
                    self.cancelled.set()
                    for task in tasks:
                        task.cancel()
        finally:
            transport.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(Worker(parser.parse_args().root).run())
