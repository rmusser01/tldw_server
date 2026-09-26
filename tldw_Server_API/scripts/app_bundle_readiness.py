"""Bounded, credential-free gateway probing for the pinned bundle control image."""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import threading
import time
from http.cookies import SimpleCookie
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.scripts.app_bundle_control import InstanceConfig


# These are private-container listeners, never host publication (validated below).
_CONTAINER_LISTEN_HOST = "0.0.0.0"  # nosec B104


class ReadinessError(ValueError):
    """The verified stack did not prove authenticated readiness and cleanup."""


def validate_runtime(records: list[dict[str, Any]], config: InstanceConfig, source_commit: str) -> str:
    """Validate private Docker inspection without printing its secret-bearing fields."""
    try:
        network_name = f"{config.project_id}_private"
        networks = [record for record in records if "Driver" in record and "Config" not in record]
        containers = [record for record in records if "Config" in record]
        if len(records) != 4 or len(networks) != 1 or len(containers) != 3:
            raise ReadinessError("runtime inspection is incomplete")
        network = networks[0]
        if (
            network["Name"] != network_name
            or network["Driver"] != "bridge"
            or network["Labels"]["com.docker.compose.project"] != config.project_id
            or network["Labels"]["com.docker.compose.network"] != "private"
        ):
            raise ReadinessError("runtime network identity differs")
        shared = {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": config.api_key,
            "SINGLE_USER_SESSION_COOKIE_NAME": config.session_cookie_name,
            "CSRF_COOKIE_NAME": config.csrf_cookie_name,
            "TLDW_GATEWAY_HOP_SECRET": config.gateway_hop_secret,
        }
        expected_env = {
            "app": {
                **shared,
                "TLDW_MANAGED_GATEWAY": "1",
                "TLDW_MANAGED_PUBLIC_ORIGIN": f"http://127.0.0.1:{config.public_port}",
                "SESSION_COOKIE_SECURE": "0",
                "PORT": "8000",
                "HOST": _CONTAINER_LISTEN_HOST,
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
                "TLDW_PUBLIC_PORT": str(config.public_port),
                "TLDW_GATEWAY_LISTEN_HOST": _CONTAINER_LISTEN_HOST,
                "TLDW_GATEWAY_LISTEN_PORT": "8080",
                "TLDW_INTERNAL_API_ORIGIN": "http://app:8000",
                "TLDW_INTERNAL_WEBUI_ORIGIN": "http://webui:3000",
            },
        }
        roles = {"app": "backend", "webui": "webui", "gateway": "gateway"}
        seen = set()
        gateway_address = ""
        for container in containers:
            actual = container["Config"]
            labels = actual["Labels"]
            service = labels["com.docker.compose.service"]
            if service not in roles or service in seen:
                raise ReadinessError("runtime service inventory differs")
            seen.add(service)
            if (
                actual["Image"] != config.images[roles[service]]
                or labels["org.opencontainers.image.revision"] != source_commit
                or labels["com.docker.compose.project"] != config.project_id
                or container["State"]["Running"] is not True
                or container["State"]["Health"]["Status"] != "healthy"
            ):
                raise ReadinessError("runtime image, source, project or health differs")
            entries = [entry.split("=", 1) for entry in actual["Env"]]
            env = dict(entries)
            if len(env) != len(entries) or any(env.get(k) != v for k, v in expected_env[service].items()):
                raise ReadinessError("runtime configuration differs")
            connected = container["NetworkSettings"]["Networks"]
            if (
                set(connected) != {network_name}
                or connected[network_name]["NetworkID"] != network["Id"]
                or container["HostConfig"]["NetworkMode"] != network_name
            ):
                raise ReadinessError("runtime private network differs")
            address = ipaddress.ip_address(connected[network_name]["IPAddress"])
            if address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified:
                raise ReadinessError("runtime private address is invalid")
            bindings = (
                {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(config.public_port)}]}
                if service == "gateway"
                else {}
            )
            for actual_ports in (container["HostConfig"]["PortBindings"], container["NetworkSettings"]["Ports"]):
                if {k: v for k, v in (actual_ports or {}).items() if v} != bindings:
                    raise ReadinessError("runtime host port bindings differ")
            if service == "gateway":
                gateway_address = str(address)
        return gateway_address
    except (KeyError, TypeError, ValueError) as exc:
        raise ReadinessError("runtime identity validation failed") from exc


class _GatewayProbe:
    """Keep the disposable cookie jar and HTTP limits within this process."""

    def __init__(self, config: InstanceConfig, address: str, port: int, deadline: float):
        self.config = config
        self.address = address
        self.port = port
        self.deadline = deadline
        self.cookies: dict[str, str] = {}

    def _remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ReadinessError("gateway readiness deadline exceeded")
        return min(remaining, 5.0)

    def request(
        self, method: str, path: str, *, authenticated: bool = False, capture: bool = False
    ) -> tuple[int, bytes]:
        """Request only the inspected gateway, with bounded body and no redirects."""
        headers = {
            "Host": f"127.0.0.1:{self.config.public_port}",
            "Origin": f"http://127.0.0.1:{self.config.public_port}",
            "Connection": "close",
        }
        if authenticated:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
            if method == "DELETE":
                headers["X-CSRF-Token"] = self.cookies.get(self.config.csrf_cookie_name, "")
        connection = http.client.HTTPConnection(self.address, self.port, timeout=self._remaining())
        timer = None
        try:
            connection.connect()
            active_socket = connection.sock

            def interrupt() -> None:
                """Stop trickling headers/bodies when the overall deadline expires."""
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # Already closed by a completed or failed request.

            timer = threading.Timer(self._remaining(), interrupt)
            timer.daemon = True
            timer.start()
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            # Capture before reading: even a broken bootstrap body can mint a session.
            if capture:
                for header in response.headers.get_all("Set-Cookie", []):
                    parsed = SimpleCookie()
                    parsed.load(header)
                    for name in (self.config.session_cookie_name, self.config.csrf_cookie_name):
                        if name in parsed:
                            self.cookies[name] = parsed[name].value
            chunks = []
            size = 0
            while True:
                timeout = self._remaining()
                active_socket.settimeout(timeout)
                chunk = response.read1(min(65536, 2 * 1024 * 1024 + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > 2 * 1024 * 1024:
                    raise ReadinessError("gateway response exceeds readiness limit")
                if response.isclosed():
                    break
            if response.length not in (None, 0):
                raise ReadinessError("gateway response body is incomplete")
            self._remaining()
            return response.status, b"".join(chunks)
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            connection.close()


def probe_gateway(config: InstanceConfig, address: str, *, port: int = 8080, budget: float = 35) -> None:
    """Prove routed readiness/assets/auth, revoke this session, then prove refusal."""
    probe = _GatewayProbe(config, address, port, time.monotonic() + budget)
    try:
        try:
            if probe.request("GET", "/internal/ready")[0] != 200:
                raise ReadinessError("backend readiness failed")
            status, page = probe.request("GET", "/")
            asset = re.search(rb'["\'](/_next/static/[a-zA-Z0-9_./%~-]+)["\']', page)
            if status != 200 or asset is None or probe.request("GET", asset[1].decode("ascii"))[0] != 200:
                raise ReadinessError("WebUI asset readiness failed")
            if probe.request("GET", "/api/v1/users/me/profile")[0] not in (401, 403):
                raise ReadinessError("gateway profile lacks authentication enforcement")
            status, _ = probe.request("POST", "/api/_tldw-webui/session", capture=True)
            if status != 200 or set(probe.cookies) != {config.session_cookie_name, config.csrf_cookie_name}:
                raise ReadinessError("gateway session bootstrap failed")
            if probe.request("GET", "/api/v1/users/me/profile", authenticated=True)[0] != 200:
                raise ReadinessError("gateway cookie authentication failed")
        finally:
            if config.session_cookie_name in probe.cookies:
                # Reserve a separate bounded cleanup window even after probe timeout.
                probe.deadline = time.monotonic() + 10
                if probe.request("DELETE", "/api/v1/auth/single-user/session", authenticated=True)[0] != 200:
                    raise ReadinessError("readiness session revocation failed")
                if probe.request("GET", "/api/v1/users/me/profile", authenticated=True)[0] not in (401, 403):
                    raise ReadinessError("revoked readiness session remains authorized")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # Never expose URLs, cookies, headers, response bodies or raw inspection.
        raise ReadinessError("gateway readiness or temporary-session cleanup failed") from exc
