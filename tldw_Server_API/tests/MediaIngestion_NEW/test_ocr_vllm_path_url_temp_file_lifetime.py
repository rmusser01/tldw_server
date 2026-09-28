"""Regression guard for TASK-13303.

`dots_ocr` and `hunyuan_ocr` build a path-based `image_url` from a
`NamedTemporaryFile(delete=True)` and then issue the POST **after** the `with` block
has closed -- and unlinked -- the file. The OCR server is handed a path that no longer
exists, so every page comes back empty and `_ocr_pdf_pages` counts zero OCR'd pages
while the run still completes.

`nemotron_parse._ocr_via_vllm` was the correct sibling for *that* defect: `delete=False`,
the path kept in `tmp_path`, and a `try/finally` that unlinks after the request. It is
parametrised throughout as the control, and it passed the lifetime cases before the fix.

It did not escape unscathed, though. Review of PR #2982 found that the same shape leaks
on a *failed write*: `tmp_path` was assigned only after `write()` and `flush()`, so an
OSError from a full filesystem left a file that `delete=False` no longer removes
automatically. All three backends now record the path before writing, and all three
route removal through `runtime_support.discard_staged_page_image`, which reports a failed
unlink instead of suppressing it silently.

The lifetime assertion is made *inside* the stubbed `fetch_json`, which is the only
moment that matters: the file has to exist when the request is made, not merely at some
point during the call.

These tests call `_ocr_via_vllm` directly. That is deliberate: it is the unit whose
temp-file contract is under test, and the public `ocr_image` wraps it in a CLI or
transformers fallback that swallows the failure and would mask every assertion here.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import pytest

pytestmark = pytest.mark.unit

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

# (module path, env prefix, extra positional args for _ocr_via_vllm)
_BACKENDS = [
    ("dots_ocr", "DOTS_VLLM", ()),
    ("hunyuan_ocr", "HUNYUAN_VLLM", ()),
    ("nemotron_parse", "NEMOTRON_VLLM", (False,)),
]


def _load(module_name: str):
    """Import one OCR backend module by its short name."""
    import importlib

    return importlib.import_module(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends."
        f"{module_name}"
    )


def _run_path_url_request(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
    on_request: Callable[[str], None],
) -> None:
    """Drive `_ocr_via_vllm` down its path-URL branch with a stubbed transport."""
    module = _load(module_name)
    monkeypatch.setenv(f"{env_prefix}_URL", "http://ocr.invalid/v1/chat/completions")
    monkeypatch.setenv(f"{env_prefix}_USE_DATA_URL", "false")
    # nemotron reads its flag through _env_bool, which accepts the same spelling.

    def _fake_fetch_json(*, method: str, url: str, json: dict, timeout: int) -> dict:
        content = json["messages"][0]["content"]
        image_url = next(
            part["image_url"]["url"] for part in content if part["type"] == "image_url"
        )
        on_request(image_url)
        return {"choices": [{"message": {"content": "page text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json", _fake_fetch_json
    )

    module._ocr_via_vllm(_PNG, "prompt", *extra_args)


@pytest.mark.parametrize(("module_name", "env_prefix", "extra_args"), _BACKENDS)
def test_temp_image_still_exists_when_the_request_is_made(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
) -> None:
    seen: dict[str, Any] = {}

    def _on_request(image_url: str) -> None:
        seen["url"] = image_url
        seen["exists"] = os.path.exists(image_url)
        seen["size"] = os.path.getsize(image_url) if seen["exists"] else 0

    _run_path_url_request(
        monkeypatch, module_name, env_prefix, extra_args, _on_request
    )

    assert seen["exists"], (
        f"{module_name} sent {seen['url']!r} to the OCR server, but the file was "
        "already unlinked -- NamedTemporaryFile(delete=True) closed before the POST. "
        "The server reads nothing, every page returns empty, and the run still reports "
        "success with zero extracted text."
    )
    assert seen["size"] == len(_PNG), (
        f"{module_name} sent a path whose contents are not the page image"
    )


@pytest.mark.parametrize(("module_name", "env_prefix", "extra_args"), _BACKENDS)
def test_temp_image_is_removed_after_the_request(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
) -> None:
    """Keeping the file alive must not turn into leaking one per page."""
    seen: dict[str, str] = {}

    _run_path_url_request(
        monkeypatch,
        module_name,
        env_prefix,
        extra_args,
        lambda image_url: seen.__setitem__("url", image_url),
    )

    assert not os.path.exists(seen["url"]), (
        f"{module_name} leaked {seen['url']!r}; a multi-page PDF would leave one "
        "temp image per page behind"
    )


@pytest.mark.parametrize(("module_name", "env_prefix", "extra_args"), _BACKENDS)
def test_temp_image_is_removed_when_the_request_fails(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
) -> None:
    """The cleanup has to be in a finally, not after the call."""
    seen: dict[str, str] = {}

    class _Boom(RuntimeError):
        pass

    def _on_request(image_url: str) -> None:
        seen["url"] = image_url
        raise _Boom("upstream refused")

    with pytest.raises(_Boom):
        _run_path_url_request(
            monkeypatch, module_name, env_prefix, extra_args, _on_request
        )

    assert not os.path.exists(seen["url"]), (
        f"{module_name} leaked {seen['url']!r} when the request raised"
    )


@pytest.mark.parametrize(("module_name", "env_prefix", "extra_args"), _BACKENDS)
def test_a_failed_write_does_not_leak_the_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
) -> None:
    """The `delete=False` needed for the fix makes a failed write leak unless handled.

    Raised by Qodo on PR #2982. `delete=True` removed the file on close even when the
    write failed; `delete=False` does not, so the path has to be recorded before the
    write and cleaned up if it raises -- otherwise a full or read-only filesystem leaves
    one partial image per page attempt.
    """
    module = _load(module_name)
    monkeypatch.setenv(f"{env_prefix}_URL", "http://ocr.invalid/v1/chat/completions")
    monkeypatch.setenv(f"{env_prefix}_USE_DATA_URL", "false")

    created: list[str] = []
    # The backends stage page images through runtime_support.image_payload, which owns
    # the temp file, so the write failure is injected there.
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support

    real_named_temp_file = runtime_support.tempfile.NamedTemporaryFile

    class _WriteFails:
        """A NamedTemporaryFile whose write raises, recording the path it created."""

        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self.name = handle.name

        def __enter__(self) -> _WriteFails:
            self._handle.__enter__()
            return self

        def __exit__(self, *exc_info: Any) -> Any:
            return self._handle.__exit__(*exc_info)

        def write(self, _data: bytes) -> int:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:  # pragma: no cover - never reached
            self._handle.flush()

    def _failing_temp_file(*args: Any, **kwargs: Any) -> _WriteFails:
        handle = real_named_temp_file(*args, **kwargs)
        created.append(handle.name)
        return _WriteFails(handle)

    monkeypatch.setattr(runtime_support.tempfile, "NamedTemporaryFile", _failing_temp_file)

    with pytest.raises(OSError):
        module._ocr_via_vllm(_PNG, "prompt", *extra_args)

    assert created, f"{module_name} never created a temp file"
    assert not os.path.exists(created[0]), (
        f"{module_name} leaked {created[0]!r} when the write failed; delete=False means "
        "nothing removes it automatically"
    )


@pytest.mark.parametrize(("module_name", "env_prefix", "extra_args"), _BACKENDS)
def test_data_url_mode_writes_no_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    env_prefix: str,
    extra_args: tuple[Any, ...],
) -> None:
    """Control: the default branch is unaffected by the fix."""
    module = _load(module_name)
    monkeypatch.setenv(f"{env_prefix}_URL", "http://ocr.invalid/v1/chat/completions")
    monkeypatch.setenv(f"{env_prefix}_USE_DATA_URL", "true")
    seen: dict[str, str] = {}

    def _fake_fetch_json(*, method: str, url: str, json: dict, timeout: int) -> dict:
        content = json["messages"][0]["content"]
        seen["url"] = next(
            part["image_url"]["url"] for part in content if part["type"] == "image_url"
        )
        return {"choices": [{"message": {"content": "page text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json", _fake_fetch_json
    )
    module._ocr_via_vllm(_PNG, "prompt", *extra_args)

    assert seen["url"].startswith("data:image/png;base64,"), module_name
