"""Tests for the nonblocking yt-dlp version diagnostic."""

import builtins
import importlib
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from _pytest.monkeypatch import MonkeyPatch

from tldw_Server_API.app.core.Ingestion_Media_Processing import yt_dlp_support


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Iterator[None]:
    with yt_dlp_support._check_lock:
        yt_dlp_support._checked = False
    yield
    with yt_dlp_support._check_lock:
        yt_dlp_support._checked = False


@pytest.mark.unit
def test_stale_yt_dlp_warns_once_without_request_data(monkeypatch: MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(yt_dlp_support.metadata, "version", lambda _name: "2025.8.11")
    monkeypatch.setattr(
        yt_dlp_support,
        "logger",
        SimpleNamespace(warning=lambda message: warnings.append(message)),
    )

    yt_dlp_support.warn_if_yt_dlp_is_stale()
    yt_dlp_support.warn_if_yt_dlp_is_stale()

    assert yt_dlp_support.MINIMUM_YT_DLP_VERSION == "2026.7.4"
    assert len(warnings) == 1
    assert "2025.8.11" in warnings[0]
    assert "minimum 2026.7.4" in warnings[0]
    assert 'pip install -U "yt-dlp>=2026.7.4"' in warnings[0]
    assert "http" not in warnings[0]
    assert "token" not in warnings[0]
    assert "password" not in warnings[0]
    assert "secret" not in warnings[0]


@pytest.mark.unit
def test_concurrent_stale_checks_perform_one_lookup_and_warning(
    monkeypatch: MonkeyPatch,
) -> None:
    worker_count = 16
    start = Barrier(worker_count)
    lookup_lock = Lock()
    lookup_count = 0
    warnings: list[str] = []

    def _version(_name: str) -> str:
        nonlocal lookup_count
        with lookup_lock:
            lookup_count += 1
        return "2025.8.11"

    def _check_version() -> None:
        start.wait()
        yt_dlp_support.warn_if_yt_dlp_is_stale()

    monkeypatch.setattr(yt_dlp_support.metadata, "version", _version)
    monkeypatch.setattr(
        yt_dlp_support,
        "logger",
        SimpleNamespace(warning=lambda message: warnings.append(message)),
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_check_version) for _ in range(worker_count)]
        for future in futures:
            future.result()

    assert lookup_count == 1
    assert len(warnings) == 1


@pytest.mark.unit
@pytest.mark.parametrize("installed_version", ["2026.7.4", "2027.1.1"])
def test_current_or_newer_yt_dlp_does_not_warn(
    monkeypatch: MonkeyPatch,
    installed_version: str,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        yt_dlp_support.metadata,
        "version",
        lambda _name: installed_version,
    )
    monkeypatch.setattr(
        yt_dlp_support,
        "logger",
        SimpleNamespace(warning=lambda message: warnings.append(message)),
    )

    yt_dlp_support.warn_if_yt_dlp_is_stale()

    assert warnings == []


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["malformed", "missing", "lookup", "version"])
def test_version_lookup_failures_do_not_raise_or_warn(
    monkeypatch: MonkeyPatch,
    failure: str,
) -> None:
    lookups = 0
    warnings: list[str] = []
    monkeypatch.setattr(
        yt_dlp_support,
        "logger",
        SimpleNamespace(warning=lambda message: warnings.append(message)),
    )

    def _version(_name: str) -> str:
        nonlocal lookups
        lookups += 1
        if failure == "malformed":
            return "not-a-version"
        if failure == "missing":
            raise PackageNotFoundError("yt-dlp")
        if failure == "lookup":
            raise RuntimeError("metadata unavailable")
        return "2025.8.11"

    monkeypatch.setattr(yt_dlp_support.metadata, "version", _version)
    if failure == "version":
        def _version_error(_value: str) -> object:
            raise RuntimeError("version parser unavailable")

        original_import = builtins.__import__

        def _import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "packaging.version":
                return SimpleNamespace(Version=_version_error)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _import)

    yt_dlp_support.warn_if_yt_dlp_is_stale()
    yt_dlp_support.warn_if_yt_dlp_is_stale()

    assert lookups == 1
    assert warnings == []


@pytest.mark.unit
def test_logging_failure_does_not_raise(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(yt_dlp_support.metadata, "version", lambda _name: "2025.8.11")
    logging_attempts = 0

    def _logging_error(_message: str) -> None:
        nonlocal logging_attempts
        logging_attempts += 1
        raise RuntimeError("logging unavailable")

    monkeypatch.setattr(
        yt_dlp_support,
        "logger",
        SimpleNamespace(warning=_logging_error),
    )

    yt_dlp_support.warn_if_yt_dlp_is_stale()
    yt_dlp_support.warn_if_yt_dlp_is_stale()

    assert logging_attempts == 1


@pytest.mark.unit
def test_video_import_and_diagnostic_survive_unavailable_packaging(
    monkeypatch: MonkeyPatch,
) -> None:
    support_module_name = yt_dlp_support.__name__
    video_module_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video."
        "Video_DL_Ingestion_Lib"
    )
    ingestion_package = importlib.import_module(
        "tldw_Server_API.app.core.Ingestion_Media_Processing"
    )
    video_package = importlib.import_module(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video"
    )
    previous_video_module = sys.modules.get(video_module_name)
    if previous_video_module is not None:
        monkeypatch.setattr(
            video_package,
            "Video_DL_Ingestion_Lib",
            previous_video_module,
            raising=False,
        )
    monkeypatch.setattr(
        ingestion_package,
        "yt_dlp_support",
        yt_dlp_support,
        raising=False,
    )
    monkeypatch.delitem(sys.modules, support_module_name)
    monkeypatch.delitem(sys.modules, video_module_name, raising=False)

    original_import = builtins.__import__

    def _import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "packaging.version":
            raise ModuleNotFoundError("packaging is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    video_lib = importlib.import_module(video_module_name)
    fresh_support = sys.modules[support_module_name]
    warnings: list[str] = []
    monkeypatch.setattr(
        fresh_support,
        "logger",
        SimpleNamespace(warning=lambda message: warnings.append(message)),
    )

    video_lib.warn_if_yt_dlp_is_stale()

    assert warnings == []


class _BoundaryReached(BaseException):
    """Stop a boundary test immediately after YoutubeDL construction."""


@pytest.mark.unit
@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "blocked"])
@pytest.mark.parametrize(
    "helper_name,args",
    [
        ("get_video_info", ("https://example.com/video",)),
        ("get_youtube", ("https://example.com/video",)),
        ("get_playlist_videos", ("https://example.com/playlist",)),
        (
            "download_video",
            ("https://example.com/video", "/unused", {}, False),
        ),
        ("extract_video_info", ("https://example.com/video",)),
        ("get_youtube_playlist_urls", ("PL123",)),
        ("extract_metadata", ("https://example.com/video",)),
    ],
)
def test_video_boundaries_check_version_after_url_validation(
    monkeypatch: MonkeyPatch,
    allowed: bool,
    helper_name: str,
    args: tuple[object, ...],
) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import (
        Video_DL_Ingestion_Lib as video_lib,
    )

    events: list[str] = []

    def _evaluate_url(*_args: object, **_kwargs: object) -> SimpleNamespace:
        events.append("validate")
        return SimpleNamespace(allowed=allowed, reason=None if allowed else "blocked")

    class _YoutubeDL:
        def __init__(self, _options: dict[str, object]) -> None:
            events.append("construct")
            raise _BoundaryReached

    monkeypatch.setattr(video_lib, "evaluate_url_policy", _evaluate_url)
    monkeypatch.setattr(
        video_lib,
        "warn_if_yt_dlp_is_stale",
        lambda: events.append("diagnostic"),
    )
    monkeypatch.setattr(video_lib.yt_dlp, "YoutubeDL", _YoutubeDL)
    if helper_name == "download_video":
        monkeypatch.setattr(video_lib.Path, "mkdir", lambda *_args, **_kwargs: None)

    if allowed:
        with pytest.raises(_BoundaryReached):
            getattr(video_lib, helper_name)(*args)
        assert events == ["validate", "diagnostic", "construct"]
    else:
        with pytest.raises(ValueError, match="blocked"):
            getattr(video_lib, helper_name)(*args)
        assert events == ["validate"]
