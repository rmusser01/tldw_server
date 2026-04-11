from __future__ import annotations

import socket

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
    CLIOCRProfile,
    ManagedOCRProfile,
    RemoteOCRProfile,
    clear_managed_process,
    effective_page_concurrency,
    get_managed_process,
    is_profile_available,
    load_ocr_runtime_profiles,
    register_managed_process,
    render_argv_template,
    reset_managed_process_registry,
)


@pytest.mark.unit
def test_remote_availability_depends_on_config_only(monkeypatch):
    profile = RemoteOCRProfile(
        mode="remote",
        host="127.0.0.1",
        port=9999,
        allow_managed_start=False,
        max_page_concurrency=4,
        argv=("{model_path}", "{image_path}"),
    )

    def _fail(*args, **kwargs):
        raise AssertionError("reachability probes are not allowed")

    monkeypatch.setattr(socket, "create_connection", _fail)
    monkeypatch.setattr("subprocess.run", _fail)

    assert is_profile_available(profile) is True  # nosec B101


@pytest.mark.unit
def test_is_profile_available_uses_managed_registry_when_backend_matches():
    reset_managed_process_registry()
    profile = ManagedOCRProfile(
        mode="managed",
        allow_managed_start=False,
        max_page_concurrency=2,
        argv=(),
    )
    process = object()
    register_managed_process("llamacpp", process)

    assert is_profile_available(profile) is False  # nosec B101
    assert is_profile_available(profile, backend_name="llamacpp") is True  # nosec B101
    assert is_profile_available(profile, backend_name="chatllm") is False  # nosec B101


@pytest.mark.unit
def test_effective_page_concurrency_uses_global_and_backend_caps():
    assert effective_page_concurrency(8, 3) == 3  # nosec B101
    assert effective_page_concurrency(3, 8) == 3  # nosec B101
    assert effective_page_concurrency(None, 5) == 5  # nosec B101
    assert effective_page_concurrency(5, None) == 5  # nosec B101
    assert effective_page_concurrency(None, None) == 1  # nosec B101


@pytest.mark.unit
def test_render_argv_template_replaces_placeholders_without_shell_execution():
    rendered = render_argv_template(
        [
            "ocr-bin",
            "--model",
            "{model_path}",
            "--image",
            "{image_path}",
            "--prompt",
            "{prompt}",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ],
        model_path="/opt/models/ocr model.bin",
        image_path="/var/run/ocr/input 01.png;rm -rf /",
        prompt='read literally $(echo pwned) & "quoted"',
        host="127.0.0.1",
        port=8123,
    )

    expected = [
        "ocr-bin",
        "--model",
        "/opt/models/ocr model.bin",
        "--image",
        "/var/run/ocr/input 01.png;rm -rf /",
        "--prompt",
        'read literally $(echo pwned) & "quoted"',
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
    ]
    assert rendered == expected  # nosec B101


@pytest.mark.unit
def test_render_argv_template_treats_replacement_values_literally():
    rendered = render_argv_template(
        ["ocr-bin", "--prompt", "{prompt}", "--host", "{host}", "--port", "{port}"],
        prompt="mention {host} and {port} literally",
        host="alpha",
        port=8123,
    )

    assert rendered == [
        "ocr-bin",
        "--prompt",
        "mention {host} and {port} literally",
        "--host",
        "alpha",
        "--port",
        "8123",
    ]  # nosec B101


@pytest.mark.unit
def test_load_ocr_runtime_profiles_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_MAX_PAGE_CONCURRENCY", "12")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["ocr-cli", "{image_path}", "{prompt}"]')

    profiles = load_ocr_runtime_profiles("LLAMACPP")

    assert isinstance(profiles.active, ManagedOCRProfile)  # nosec B101
    assert profiles.active.allow_managed_start is True  # nosec B101
    assert profiles.active.max_page_concurrency == 12  # nosec B101
    assert profiles.active.argv == ("ocr-cli", "{image_path}", "{prompt}")  # nosec B101


@pytest.mark.unit
def test_load_ocr_runtime_profiles_parses_env_inputs():
    env = {
        "LLAMACPP_OCR_MODE": "managed",
        "LLAMACPP_OCR_ALLOW_MANAGED_START": "true",
        "LLAMACPP_OCR_MAX_PAGE_CONCURRENCY": "12",
        "LLAMACPP_OCR_ARGV": '["ocr-cli", "{image_path}", "{prompt}"]',
    }

    profiles = load_ocr_runtime_profiles("LLAMACPP", env=env)

    assert isinstance(profiles.active, ManagedOCRProfile)  # nosec B101
    assert profiles.active.allow_managed_start is True  # nosec B101
    assert profiles.active.max_page_concurrency == 12  # nosec B101
    assert profiles.active.argv == ("ocr-cli", "{image_path}", "{prompt}")  # nosec B101


@pytest.mark.unit
def test_load_ocr_runtime_profiles_defaults_page_concurrency_to_one():
    profiles = load_ocr_runtime_profiles(
        "LLAMACPP",
        env={"LLAMACPP_OCR_MODE": "remote"},
    )

    assert profiles.active.max_page_concurrency == 1  # nosec B101


@pytest.mark.unit
def test_managed_process_registry_is_keyed_by_backend_name():
    reset_managed_process_registry()
    process_a = object()
    process_b = object()

    register_managed_process("llamacpp", process_a)
    register_managed_process("chatllm", process_b)

    assert get_managed_process("llamacpp") is process_a  # nosec B101
    assert get_managed_process("chatllm") is process_b  # nosec B101

    assert clear_managed_process("llamacpp") is process_a  # nosec B101
    assert get_managed_process("llamacpp") is None  # nosec B101
    assert get_managed_process("chatllm") is process_b  # nosec B101


@pytest.mark.unit
def test_wait_for_managed_http_ready_uses_https_connection_when_requested(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod

    captured: dict[str, object] = {"https_calls": 0, "http_calls": 0}

    class _Response:
        status = 200

        def read(self):
            return b""

    class _HTTPSConnection:
        def __init__(self, host, port, timeout):
            captured["https_calls"] = captured.get("https_calls", 0) + 1
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(self, method, path):
            captured["method"] = method
            captured["path"] = path

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    class _HTTPConnection:
        def __init__(self, host, port, timeout):
            captured["http_calls"] = captured.get("http_calls", 0) + 1

        def request(self, method, path):
            raise AssertionError("HTTPConnection should not be used for https probes")

        def getresponse(self):
            raise AssertionError("HTTPConnection should not be used for https probes")

        def close(self):
            return None

    monkeypatch.setattr(runtime_mod.http.client, "HTTPSConnection", _HTTPSConnection)
    monkeypatch.setattr(runtime_mod.http.client, "HTTPConnection", _HTTPConnection)

    assert runtime_mod.wait_for_managed_http_ready(
        host="chatllm.local",
        port=9443,
        scheme="https",
        timeout_total=0.5,
        interval=0.1,
        paths=("/ready",),
    ) is True  # nosec B101
    assert captured["https_calls"] == 1  # nosec B101
    assert captured["http_calls"] == 0  # nosec B101
    assert captured["path"] == "/ready"  # nosec B101
