from pathlib import Path

from mcp_unified.lsp import LspRuntimeConfig
from mcp_unified.lsp.sessions import LspSessionManager


class _FakeSessionBackend:
    def __init__(self, *, fail_close: bool = False):
        self.fail_close = fail_close
        self.closed = False

    async def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")
        self.closed = True


async def test_session_manager_reuses_workspace_backend_session(tmp_path: Path):
    created: list[tuple[str, Path]] = []

    async def factory(backend_id: str, workspace_root: Path, config: LspRuntimeConfig) -> _FakeSessionBackend:
        created.append((backend_id, workspace_root))
        return _FakeSessionBackend()

    manager = LspSessionManager(config=LspRuntimeConfig(idle_ttl_seconds=300), backend_factory=factory)

    first = await manager.get_session("ruff", workspace_root=tmp_path)
    second = await manager.get_session("ruff", workspace_root=tmp_path)

    assert first is second
    assert manager.active_session_count == 1
    assert created == [("ruff", tmp_path.resolve())]


async def test_session_manager_stop_all_is_exception_safe(tmp_path: Path):
    backends: dict[str, _FakeSessionBackend] = {}

    async def factory(backend_id: str, workspace_root: Path, config: LspRuntimeConfig) -> _FakeSessionBackend:
        backend = _FakeSessionBackend(fail_close=backend_id == "ruff")
        backends[backend_id] = backend
        return backend

    manager = LspSessionManager(config=LspRuntimeConfig(), backend_factory=factory)
    await manager.get_session("ruff", workspace_root=tmp_path)
    await manager.get_session("pylsp", workspace_root=tmp_path)

    errors = await manager.stop_all()

    assert manager.active_session_count == 0
    assert errors["ruff"] == "RuntimeError"
    assert backends["pylsp"].closed is True


async def test_session_manager_evicts_idle_sessions(tmp_path: Path):
    async def factory(backend_id: str, workspace_root: Path, config: LspRuntimeConfig) -> _FakeSessionBackend:
        return _FakeSessionBackend()

    manager = LspSessionManager(config=LspRuntimeConfig(idle_ttl_seconds=1), backend_factory=factory)
    session = await manager.get_session("ruff", workspace_root=tmp_path)

    await manager.evict_idle_sessions(now=session.last_used_monotonic + 2)

    assert manager.active_session_count == 0
    assert session.backend.closed is True
