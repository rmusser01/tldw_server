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


@pytest.fixture
def ocr_probe(monkeypatch):
    """Run the readiness probe under the production egress policy, without a network.

    Traffic goes through the real core.http_client helpers over an httpx.MockTransport.
    Returns ``(seen, connects, dns, responses)``: ``(url, request)`` for each request the
    transport received, the URL as sent (http_client restores the logical URL on the request
    afterwards), every attempted socket connect (a probe that bypasses the helpers reaches
    one), a mutable DNS table, and queued responses (default 200).
    """
    import httpx

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod
    from tldw_Server_API.app.core.Security import egress

    seen: list[tuple[str, httpx.Request]] = []
    connects: list[object] = []
    dns: dict[str, str] = {}
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request))
        return responses.pop(0) if responses else httpx.Response(200)

    def no_connect(address, *_args, **_kwargs):
        connects.append(address)
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket, "create_connection", no_connect)
    monkeypatch.setattr(
        runtime_mod,
        "create_client",
        lambda **kwargs: http_client.create_client(transport=httpx.MockTransport(handler), **kwargs),
        raising=False,
    )
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda host, **_kwargs: [dns[host]] if host in dns else [])
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    return seen, connects, dns, responses


@pytest.mark.unit
@pytest.mark.parametrize(("host", "authority"), [("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")])
def test_wait_for_managed_http_ready_reaches_the_configured_loopback_server(ocr_probe, host, authority):
    import httpx

    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod

    seen, connects, _dns, responses = ocr_probe
    responses.append(httpx.Response(503))

    assert runtime_mod.wait_for_managed_http_ready(host=host, port=8090, timeout_total=5, interval=0.25)

    assert [(request.method, url) for url, request in seen] == [
        ("GET", f"http://{authority}:8090/health"),
        ("GET", f"http://{authority}:8090/v1/models"),
    ]
    assert seen[0][1].extensions["timeout"] == {"connect": 0.25, "read": 0.25, "write": 0.25, "pool": 0.25}
    assert connects == []


@pytest.mark.unit
def test_wait_for_managed_http_ready_uses_https_when_requested(ocr_probe):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod

    seen, connects, dns, _responses = ocr_probe
    dns["chatllm.local"] = "192.168.1.20"

    assert runtime_mod.wait_for_managed_http_ready(
        host="chatllm.local",
        port=9443,
        scheme="https",
        timeout_total=0.5,
        interval=0.1,
        paths=("/ready?probe=1",),
    ) is True  # nosec B101

    [(url, request)] = seen
    # The connection is pinned to the resolved address; Host carries the configured origin.
    assert url == "https://192.168.1.20:9443/ready?probe=1"  # nosec B101
    assert request.headers["host"] == "chatllm.local:9443"  # nosec B101
    assert connects == []  # nosec B101


@pytest.mark.unit
def test_wait_for_managed_http_ready_refuses_a_cloud_metadata_answer_without_waiting(ocr_probe):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod

    seen, connects, dns, _responses = ocr_probe
    dns["ocr.internal"] = "169.254.169.254"

    assert runtime_mod.wait_for_managed_http_ready(host="ocr.internal", port=80, timeout_total=0.3, interval=0.1) is False

    assert seen == []
    assert connects == []


@pytest.mark.unit
def test_wait_for_managed_http_ready_does_not_follow_a_redirect_to_another_private_address(ocr_probe):
    import httpx

    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR import runtime_support as runtime_mod

    seen, connects, _dns, responses = ocr_probe
    responses.extend(httpx.Response(302, headers={"location": "http://10.0.0.9:8090/health"}) for _ in range(50))

    ready = runtime_mod.wait_for_managed_http_ready(
        host="127.0.0.1", port=8090, timeout_total=0.2, interval=0.05, paths=("/health",)
    )

    assert ready is False
    assert seen
    assert {httpx.URL(url).host for url, _request in seen} == {"127.0.0.1"}
    assert connects == []
