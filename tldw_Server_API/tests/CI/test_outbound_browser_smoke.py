"""Offline contracts for the browser smoke's private attempt and result evidence."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from Helper_Scripts.ci.outbound_browser_smoke import check_attempts, validate_browser_result


def test_check_attempts_requires_exact_endpoints_and_rejects_unix_services(tmp_path: Path):
    trace = tmp_path / "browser.12"
    trace.write_text(
        'connect(9, {sa_family=AF_INET, sin_port=htons(18082), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
        'connect(10, {sa_family=AF_INET, sin_port=htons(18085), sin_addr=inet_addr("127.0.0.1")}, 16) = -1 ECONNREFUSED\n'
        'sendto(11, ""..., 13, MSG_NOSIGNAL, {sa_family=AF_INET6, sin6_port=htons(53), '
        'inet_pton(AF_INET6, "2001:db8::5", &sin6_addr)}, 28) = -1 ENETUNREACH\n'
        'connect(12, {sa_family=AF_UNIX, sun_path="/run/private-fixture.sock"}, 31) = -1 ENOENT\n'
        "connect(13, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=0}, 12) = 0\n"
        "+++ exited with 0 +++\n",
        encoding="utf-8",
    )
    assert check_attempts(tmp_path / "browser", {18082}, required_port=18082) == {
        "unapproved_socket",
        "unapproved_unix_socket",
    }


def test_check_attempts_does_not_infer_coverage_for_unfinished_or_malformed_trace(tmp_path: Path):
    trace = tmp_path / "browser.12"
    trace.write_text(
        "connect(9, {sa_family=AF_INET, sin_port=htons(18082)}, 16) = 0\n",
        encoding="utf-8",
    )
    assert check_attempts(tmp_path / "browser", {18082}, required_port=18082) == {
        "trace_incomplete",
        "trace_unclassified",
    }


def test_check_attempts_accepts_ipv4_mapped_fixture_address(tmp_path: Path):
    trace = tmp_path / "browser.12"
    trace.write_text(
        "connect(9, {sa_family=AF_INET6, sin6_port=htons(18083), "
        'inet_pton(AF_INET6, "::ffff:127.0.0.1", &sin6_addr)}, 28) = -1 EINPROGRESS\n'
        "+++ exited with 0 +++\n",
        encoding="utf-8",
    )
    assert check_attempts(tmp_path / "browser", {18083}, required_port=18083) == set()


def _write_netlog(path: Path, *urls: str) -> None:
    path.write_text(json.dumps({
        "constants": {
            "logEventTypes": {"REQUEST_ALIVE": 1, "URL_REQUEST_START_JOB": 2},
            "logEventPhase": {"PHASE_BEGIN": 1},
        },
        "events": [
            {"type": 1, "phase": 1, "params": {"url": url}} for url in urls
        ] + [{"type": 1, "phase": 2}],
        "polledData": [],
    }))


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ws://127.0.0.1:18083/allowed", set()),
        ("wss://opaque-destination.invalid/private?token=synthetic", {"unapproved_extension_websocket"}),
        ("ws://127.0.0.1:18082/wrong-port", {"unapproved_extension_websocket"}),
        ("wss://127.0.0.1:18083/wrong-protocol", {"unapproved_extension_websocket"}),
        ("https://unrelated-browser-request.invalid", set()),
    ],
)
def test_extension_observes_websocket_attempts_before_a_socket_connect(tmp_path, url, reason):
    from Helper_Scripts.ci import outbound_browser_smoke as smoke

    log = tmp_path / "network.json"
    _write_netlog(log, url)
    assert smoke.check_extension_websockets(log, 18083) == reason


def test_control_socket_must_appear_in_its_own_profile(tmp_path):
    from Helper_Scripts.ci import outbound_browser_smoke as smoke

    log, control = tmp_path / "network.json", tmp_path / "control.json"
    _write_netlog(log, "http://127.0.0.1:18083/")
    _write_netlog(control, "ws://127.0.0.1:18083/")
    assert smoke.check_extension_websockets(log, 18083, control=control) == {
        "browser_netlog_missing_websocket_control"
    }

    _write_netlog(control, smoke.WEBSOCKET_CONTROL_URL)
    assert smoke.check_extension_websockets(log, 18083, control=control) == set()

    _write_netlog(log, smoke.WEBSOCKET_CONTROL_URL)
    assert smoke.check_extension_websockets(log, 18083, control=control) == {
        "unapproved_extension_websocket"
    }


@pytest.mark.parametrize("invalid", [b"", b"{", b" " * (4 * 1024 * 1024 + 1)], ids=["empty", "partial", "oversized"])
def test_extension_netlog_must_be_complete_and_bounded(tmp_path, invalid):
    from Helper_Scripts.ci import outbound_browser_smoke as smoke

    log = tmp_path / "network.json"
    log.write_bytes(invalid)
    assert smoke.check_extension_websockets(log, 18083) == {"browser_netlog_incomplete"}


def test_browser_result_requires_one_expected_pass_and_worker_evidence(tmp_path: Path):
    report, observer = tmp_path / "report", tmp_path / "observer"
    report.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "specs": [
                            {"title": "expected", "tests": [{"status": "expected", "results": [{"status": "passed"}]}]}
                        ],
                    }
                ]
            }
        )
    )
    observer.write_text(
        json.dumps(
            {
                "allowedRequests": 1,
                "workerRequests": 0,
                "unexpectedRequests": 0,
                "tripwirePassed": False,
            }
        )
    )
    assert validate_browser_result(report, observer, "expected", require_worker=True) == {
        "browser_observer_missing_worker"
    }
    assert validate_browser_result(report, observer, "expected") == set()
    report.write_text(json.dumps({"suites": []}))
    assert validate_browser_result(report, observer, "expected") == {"playwright_evidence_missing"}


def test_browser_result_rejects_blocked_requests_even_when_playwright_passes(tmp_path: Path):
    report, observer = tmp_path / "report", tmp_path / "observer"
    report.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "specs": [
                            {"title": "expected", "tests": [{"status": "expected", "results": [{"status": "passed"}]}]}
                        ],
                    }
                ]
            }
        )
    )
    observer.write_text(
        json.dumps(
            {
                "allowedRequests": 1,
                "workerRequests": 0,
                "unexpectedRequests": 1,
                "tripwirePassed": False,
            }
        )
    )
    assert validate_browser_result(report, observer, "expected", require_tripwire=True) == {
        "unapproved_browser_request",
        "browser_observer_missing_tripwire",
    }


@pytest.mark.parametrize(("stopped", "exit_status"), [(True, 0), (False, 1)])
def test_next_requires_a_live_process_and_clean_shutdown(tmp_path, monkeypatch, stopped, exit_status):
    """An early exit or failed shutdown cannot count as a successful Next trace."""
    from Helper_Scripts.ci import outbound_browser_smoke as smoke

    server = tmp_path / "apps/tldw-frontend/.next/standalone/apps/tldw-frontend/server.js"
    server.parent.mkdir(parents=True)
    server.touch()
    (tmp_path / "next.pid").write_text("241\n", encoding="ascii")
    tracer = Mock()
    tracer.__enter__ = Mock(return_value=tracer)
    tracer.__exit__ = Mock(return_value=False)
    tracer.poll.return_value = exit_status if stopped else None
    tracer.wait.return_value = exit_status
    monkeypatch.setattr(smoke.shutil, "which", lambda _: "/fixture/node")
    monkeypatch.setattr(smoke, "_trace_command", lambda *_: [])
    monkeypatch.setattr(smoke, "_wait_next", lambda *_: None)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: tracer)
    monkeypatch.setattr(smoke.os, "kill", lambda *args: None)

    with pytest.raises(smoke.BrowserSmokeError):
        with smoke._next_server(tmp_path, tmp_path, {}, "http://127.0.0.1:18081", "fixture-key"):
            pass


@pytest.mark.parametrize("change", ["match", "different-origin", "removed-capability"])
def test_pristine_manifest_agrees_with_reviewed_inventory(tmp_path, change):
    """Changes to the reviewed capability must be reflected by the packaged manifest."""
    from Helper_Scripts.ci import outbound_browser_smoke as smoke

    original = Path(smoke.__file__).with_name("outbound_inventory.json")
    payload = json.loads(original.read_text(encoding="utf-8"))
    entry = next(item for item in payload["capabilities"] if item["id"] == "extension.github-api")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"host_permissions": ["https://api.github.com/*"]}))
    if change == "different-origin":
        entry["origins"] = ["https://different.invalid"]
    if change == "removed-capability":
        payload["capabilities"].remove(entry)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload))

    if change == "match":
        assert smoke.require_extension_host_permission(inventory, manifest) == "https://api.github.com/*"
    else:
        with pytest.raises(smoke.BrowserSmokeError):
            smoke.require_extension_host_permission(inventory, manifest)


@pytest.mark.parametrize(
    ("chromium", "opcode", "family", "type_label", "host", "port", "result", "reasons"),
    [
        (True, "connect", "AF_INET6", "UDPv6", "2001:4860:4860::8888", 443, "-1 ENETUNREACH", set()),
        (False, "connect", "AF_INET6", "UDPv6", "2001:4860:4860::8888", 443, "-1 ENETUNREACH", {"unapproved_socket"}),
        (True, "connect", "AF_INET6", "TCPv6", "2001:4860:4860::8888", 443, "-1 ENETUNREACH", {"unapproved_socket"}),
        (True, "connect", "AF_INET6", "UDPv6", "2001:4860:4860::8888", 443, "0", {"unapproved_socket"}),
        (True, "connect", "AF_INET6", "UDPv6", "2001:4860:4860::8888", 53, "-1 ENETUNREACH", {"unapproved_socket"}),
        (True, "connect", "AF_INET6", "UDPv6", "2001:db8::5", 443, "-1 ENETUNREACH", {"unapproved_socket"}),
        (True, "sendto", "AF_INET6", "UDPv6", "2001:4860:4860::8888", 443, "-1 ENETUNREACH", {"unapproved_socket"}),
    ],
)
def test_only_failed_udp_route_query_from_traced_chromium_is_classified(
    tmp_path, chromium, opcode, family, type_label, host, port, result, reasons
):
    trace = tmp_path / "browser.12"
    trace.write_text(
        f"{opcode}(32<{type_label}:[96255]>, {{sa_family={family}, sin6_port=htons({port}), "
        f'inet_pton(AF_INET6, "{host}", &sin6_addr)}}, 28) = {result}\n'
        "+++ exited with 0 +++\n",
        encoding="utf-8",
    )
    assert check_attempts(tmp_path / "browser", set(), chromium=chromium) == reasons


def test_trace_ignores_the_next_pid_coordination_file(tmp_path: Path):
    (tmp_path / "next.pid").write_text("241\n", encoding="ascii")
    trace = tmp_path / "next.241"
    trace.write_text(
        'connect(9, {sa_family=AF_INET, sin_port=htons(18081), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
        "+++ exited with 0 +++\n",
        encoding="utf-8",
    )
    assert check_attempts(tmp_path / "next", {18081}, required_port=18081) == set()
