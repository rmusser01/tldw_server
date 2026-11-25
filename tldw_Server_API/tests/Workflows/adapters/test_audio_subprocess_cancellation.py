"""A cancelled tts or multi_voice_tts step kills the ffmpeg it started (TASK-13887)."""

import ast
import asyncio
import contextlib
import sys
from pathlib import Path

import psutil
import pytest

from tldw_Server_API.app.core.Workflows import subprocess_utils
from tldw_Server_API.app.core.Workflows.adapters.audio import multi_voice_tts
from tldw_Server_API.app.core.Workflows.adapters.audio import tts as tts_adapter

pytestmark = pytest.mark.unit

_ADAPTERS_DIR = Path(tts_adapter.__file__).resolve().parents[1]


@pytest.fixture
def slow_ffmpeg(tmp_path, monkeypatch):
    """An "ffmpeg" that writes its pid, then runs until it is killed."""
    pid_file = tmp_path / "ffmpeg.pid"
    script = tmp_path / "ffmpeg"
    script.write_text(
        "import os, pathlib, sys, time\npathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    children: list[psutil.Process] = []

    async def run_ffmpeg(cmd, **kwargs):
        # Only substitute the command; production still owns cancellation and reaping.
        return await subprocess_utils.run_checked_async(
            [sys.executable, str(script), str(pid_file), *cmd[1:]], **kwargs
        )

    monkeypatch.setattr("shutil.which", lambda name: str(script) if name == "ffmpeg" else None)
    monkeypatch.setattr(multi_voice_tts, "run_checked_async", run_ffmpeg)
    monkeypatch.setattr(tts_adapter, "run_checked_async", run_ffmpeg)
    yield pid_file, children
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()  # A regression can leave the child running.
            child.wait(timeout=5)


async def _cancel_once_ffmpeg_runs(step, ffmpeg: tuple[Path, list[psutil.Process]]) -> psutil.Process:
    """Cancel after the real child starts, retaining its identity for safe liveness checks."""
    pid_file, children = ffmpeg
    task = asyncio.create_task(step)
    try:
        for _ in range(500):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            await asyncio.sleep(0.01)
        child = psutil.Process(int(pid_file.read_text()))
        children.append(child)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return child
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_a_cancelled_multi_voice_tts_step_kills_its_ffmpeg(slow_ffmpeg, tmp_path) -> None:
    """On dev the cancel left ffmpeg running to its end, past the step's timeout."""
    child = await _cancel_once_ffmpeg_runs(
        multi_voice_tts._normalize_audio(tmp_path / "in.mp3", tmp_path / "out.mp3"), slow_ffmpeg
    )

    assert not child.is_running()


async def test_a_cancelled_tts_step_kills_its_ffmpeg(slow_ffmpeg, tmp_path, monkeypatch) -> None:
    """On dev the cancel during post-process normalization left ffmpeg running."""
    monkeypatch.chdir(tmp_path)

    class _Speech:
        async def generate_speech(self, _request, provider=None, **_kwargs):  # noqa: ARG002
            yield b"audio"

    async def _service():
        return _Speech()

    monkeypatch.setattr(tts_adapter, "get_tts_service_v2", _service)
    config = {
        "input": "Hello",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "post_process": {"normalize": True},
    }

    child = await _cancel_once_ffmpeg_runs(tts_adapter.run_tts_adapter(config, {}), slow_ffmpeg)

    assert not child.is_running()


_SPAWNS = {"create_subprocess_exec", "create_subprocess_shell", "subprocess_exec", "subprocess_shell"}


def _spawns_directly(node: ast.AST) -> bool:
    """``asyncio.create_subprocess_exec(...)``, ``loop.subprocess_exec(...)``, or ``from asyncio import create_subprocess_exec``."""
    if isinstance(node, ast.Attribute):
        return node.attr in _SPAWNS
    return isinstance(node, ast.ImportFrom) and any(alias.name in _SPAWNS for alias in node.names)


def test_workflow_adapters_start_commands_only_through_subprocess_utils() -> None:
    """run_checked_async kills and reaps its child on cancel; a direct asyncio spawn does not (dev: 8 sites)."""
    direct = [
        f"{path.relative_to(_ADAPTERS_DIR)}:{node.lineno}"
        for path in sorted(_ADAPTERS_DIR.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if _spawns_directly(node)
    ]

    assert direct == []
