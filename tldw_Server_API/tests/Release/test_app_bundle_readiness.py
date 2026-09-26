"""Readiness must traverse the gateway and revoke only its fresh session."""

import copy
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tldw_Server_API.scripts import app_bundle_readiness as readiness


@pytest.fixture
def config():
    return SimpleNamespace(
        project_id="tldw_fixture",
        public_port=18080,
        api_key="private-master",
        gateway_hop_secret="private-hop",
        session_cookie_name="session_fixture",
        csrf_cookie_name="csrf_fixture",
        images={role: f"registry/{role}@sha256:" + "a" * 64 for role in ("backend", "webui", "gateway")},
    )


@pytest.fixture
def inspection(config):
    network = {
        "Id": "n" * 64,
        "Name": "tldw_fixture_private",
        "Driver": "bridge",
        "Internal": False,
        "Labels": {"com.docker.compose.project": "tldw_fixture", "com.docker.compose.network": "private"},
    }
    records = [network]
    shared = {
        "SINGLE_USER_API_KEY": config.api_key,
        "SINGLE_USER_SESSION_COOKIE_NAME": config.session_cookie_name,
        "CSRF_COOKIE_NAME": config.csrf_cookie_name,
        "AUTH_MODE": "single_user",
        "TLDW_GATEWAY_HOP_SECRET": config.gateway_hop_secret,
    }
    envs = {
        "app": {
            **shared,
            "TLDW_MANAGED_GATEWAY": "1",
            "TLDW_MANAGED_PUBLIC_ORIGIN": "http://127.0.0.1:18080",
            "SESSION_COOKIE_SECURE": "0",
            "PORT": "8000",
            "HOST": "0.0.0.0",
        },
        "webui": {
            **shared,
            "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE": "managed",
            "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH": "1",
            "TLDW_INTERNAL_API_ORIGIN": "http://app:8000",
        },
        "gateway": {
            "TLDW_GATEWAY_HOP_SECRET": config.gateway_hop_secret,
            "TLDW_PUBLIC_HOST": "127.0.0.1",
            "TLDW_PUBLIC_PORT": "18080",
            "TLDW_GATEWAY_LISTEN_HOST": "0.0.0.0",
            "TLDW_GATEWAY_LISTEN_PORT": "8080",
            "TLDW_INTERNAL_API_ORIGIN": "http://app:8000",
            "TLDW_INTERNAL_WEBUI_ORIGIN": "http://webui:3000",
        },
    }
    for i, (service, role) in enumerate((("app", "backend"), ("webui", "webui"), ("gateway", "gateway"))):
        ports = {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}]} if service == "gateway" else {}
        records.append(
            {
                "Id": str(i) * 64,
                "Driver": "overlayfs",
                "State": {"Running": True, "Health": {"Status": "healthy"}},
                "Config": {
                    "Image": config.images[role],
                    "Labels": {
                        "com.docker.compose.project": "tldw_fixture",
                        "com.docker.compose.service": service,
                        "org.opencontainers.image.revision": "b" * 40,
                    },
                    "Env": [f"{k}={v}" for k, v in envs[service].items()],
                },
                "HostConfig": {"NetworkMode": "tldw_fixture_private", "PortBindings": ports},
                "NetworkSettings": {
                    "Ports": ports,
                    "Networks": {"tldw_fixture_private": {"NetworkID": "n" * 64, "IPAddress": f"172.23.0.{i+2}"}},
                },
            }
        )
    return records


def test_runtime_identity_selects_only_verified_gateway(config, inspection):
    assert readiness.validate_runtime(inspection, config, "b" * 40) == "172.23.0.4"


@pytest.mark.parametrize(
    "damage",
    [
        "image",
        "revision",
        "project",
        "network",
        "public-backend",
        "public-gateway",
        "credentials",
        "private-target",
        "unhealthy",
        "missing",
        "duplicate",
        "missing-network",
        "mixed-network",
        "malformed",
    ],
)
def test_runtime_identity_refuses_mismatched_stack(config, inspection, damage):
    records = copy.deepcopy(inspection)
    if damage == "image":
        records[1]["Config"]["Image"] = "unsigned:latest"
    if damage == "revision":
        records[1]["Config"]["Labels"]["org.opencontainers.image.revision"] = "c" * 40
    if damage == "project":
        records[1]["Config"]["Labels"]["com.docker.compose.project"] = "other"
    if damage == "network":
        records[0]["Labels"]["com.docker.compose.project"] = "other"
    if damage == "public-backend":
        records[1]["HostConfig"]["PortBindings"] = {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8000"}]}
    if damage == "public-gateway":
        records[3]["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
    if damage == "credentials":
        records[2]["Config"]["Env"] = [
            v for v in records[2]["Config"]["Env"] if not v.startswith("SINGLE_USER_API_KEY=")
        ]
    if damage == "private-target":
        records[3]["Config"]["Env"].append("TLDW_INTERNAL_API_ORIGIN=http://unrelated:8000")
    if damage == "unhealthy":
        records[1]["State"]["Health"]["Status"] = "unhealthy"
    if damage == "missing":
        records.pop()
    if damage == "duplicate":
        records.append(records[-1])
    if damage == "missing-network":
        records.pop(0)
    if damage == "mixed-network":
        records[0]["Config"] = records[1]["Config"]
    if damage == "malformed":
        records[0] = {}
    with pytest.raises(readiness.ReadinessError):
        readiness.validate_runtime(records, config, "b" * 40)


@pytest.fixture
def gateway(config):
    state = SimpleNamespace(
        mode="ok", requests=[], revoked=False, cookie="session_fixture=fresh-probe; csrf_fixture=csrf-probe"
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def do_DELETE(self):
            self.respond()

        def respond(self):
            state.requests.append((self.command, self.path, dict(self.headers)))
            assert self.headers["Host"] == "127.0.0.1:18080"
            assert self.headers["Origin"] == "http://127.0.0.1:18080"
            assert not self.headers.get("X-API-KEY") and not self.headers.get("Authorization")
            status, body, cookies = 200, b"ok", []
            if self.path == "/" and state.mode == "headers":
                try:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    for _ in range(20):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.03)
                    self.wfile.write(b"\r\nContent-Length: 0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == "/":
                body = b'<script src="/_next/static/chunk.js"></script>'
            if self.path == "/api/_tldw-webui/session":
                if state.mode == "bootstrap":
                    status = 503
                else:
                    cookies = ["session_fixture=fresh-probe; Path=/; HttpOnly", "csrf_fixture=csrf-probe; Path=/"]
            if self.path == "/api/v1/users/me/profile":
                if not self.headers.get("Cookie") or state.revoked:
                    status = 401
                elif state.mode == "auth":
                    status = 403
            if self.command == "DELETE":
                assert self.headers.get("Cookie") == state.cookie
                assert self.headers.get("X-CSRF-Token") == "csrf-probe"
                if state.mode == "revoke":
                    status = 503
                elif state.mode != "ignored-revoke":
                    state.revoked = True
            self.send_response(status)
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
            if state.mode == "oversize" and self.path.endswith("session") and self.command == "POST":
                body = b"x" * (2 * 1024 * 1024 + 1)
            self.send_header(
                "Content-Length", str(len(body) + (10 if state.mode == "truncated" and self.command == "POST" else 0))
            )
            self.end_headers()
            if state.mode == "stall" and self.path.endswith("session") and self.command == "POST":
                time.sleep(0.3)
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_gateway_probe_uses_fresh_cookie_then_proves_exact_revocation(config, gateway):
    state, port = gateway
    readiness.probe_gateway(config, "127.0.0.1", port=port)
    assert state.revoked
    assert [r[:2] for r in state.requests] == [
        ("GET", "/internal/ready"),
        ("GET", "/"),
        ("GET", "/_next/static/chunk.js"),
        ("GET", "/api/v1/users/me/profile"),
        ("POST", "/api/_tldw-webui/session"),
        ("GET", "/api/v1/users/me/profile"),
        ("DELETE", "/api/v1/auth/single-user/session"),
        ("GET", "/api/v1/users/me/profile"),
    ]
    assert "Cookie" not in state.requests[4][2]
    assert state.requests[-1][2]["Cookie"] == state.cookie


@pytest.mark.parametrize(
    "mode", ["bootstrap", "auth", "revoke", "ignored-revoke", "oversize", "stall", "headers", "truncated"]
)
def test_healthy_gateway_cannot_mask_auth_or_cleanup_or_body_failure(config, gateway, mode):
    state, port = gateway
    state.mode = mode
    with pytest.raises(readiness.ReadinessError):
        readiness.probe_gateway(config, "127.0.0.1", port=port, budget=0.15 if mode in ("stall", "headers") else 3)
    if mode in ("auth", "oversize", "stall", "truncated"):
        assert state.revoked


def test_slow_trickling_headers_cannot_extend_total_probe_deadline(config, gateway):
    state, port = gateway
    state.mode = "headers"
    started = time.monotonic()
    with pytest.raises(readiness.ReadinessError):
        readiness.probe_gateway(config, "127.0.0.1", port=port, budget=0.15)
    assert time.monotonic() - started < 0.4
