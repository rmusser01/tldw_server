"""
IP allowlist/denylist enforcement utilities for MCP Unified.

Provides a shared controller for both HTTP routes and WebSocket connections.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from functools import lru_cache

from fastapi import HTTPException, Request
from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode
from ..config import get_config


class IPAccessController:
    """Evaluate client IPs against allow/deny rules."""

    def __init__(
        self,
        allowed: Iterable[str],
        blocked: Iterable[str],
        trust_x_forwarded_for: bool,
        trusted_proxy_depth: int,
        trusted_proxies: Iterable[str],
    ) -> None:
        self.allowed_networks = self._parse_networks(list(allowed))
        self.blocked_networks = self._parse_networks(list(blocked))
        self.trust_x_forwarded_for = trust_x_forwarded_for
        self.trusted_proxy_depth = max(0, trusted_proxy_depth or 0)
        self.trusted_proxy_networks = self._parse_networks(list(trusted_proxies))

    @staticmethod
    def _parse_networks(cidrs: list[str]) -> list[ipaddress._BaseNetwork]:
        networks: list[ipaddress._BaseNetwork] = []
        for entry in cidrs:
            if not entry:
                continue
            try:
                if "/" in entry:
                    networks.append(ipaddress.ip_network(entry, strict=False))
                else:
                    # Single IP - derive correct mask
                    family_mask = "/32" if ":" not in entry else "/128"
                    networks.append(ipaddress.ip_network(f"{entry}{family_mask}", strict=False))
            except ValueError:
                logger.warning(f"Ignoring invalid IP/CIDR entry in MCP config: {entry}")
        return networks

    def _is_trusted_proxy(self, ip_str: str | None) -> bool:
        """Return True when the immediate peer is a trusted proxy."""
        if not ip_str:
            return False
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        if self.trusted_proxy_networks:
            return any(ip_obj in network for network in self.trusted_proxy_networks)

        # Without explicit configuration fall back to loopback range only.
        return ip_obj.is_loopback

    def resolve_client_ip(
        self,
        remote_addr: str | None,
        forwarded_for: str | None,
        real_ip: str | None = None,
    ) -> str | None:
        """Resolve the effective client IP applying X-Forwarded-For rules."""
        # Never trust forwarded headers by default; start from the immediate peer.
        candidate = remote_addr
        try:
            if self.trust_x_forwarded_for and forwarded_for and self._is_trusted_proxy(remote_addr):
                chain = [part.strip() for part in forwarded_for.split(",") if part.strip()]
                if chain:
                    # Use trusted-proxy depth to pick the client hop.
                    # If there are fewer hops than expected, fail safe to the immediate peer.
                    if self.trusted_proxy_depth > 0:
                        if len(chain) >= self.trusted_proxy_depth:
                            candidate = chain[-self.trusted_proxy_depth]
                        else:
                            candidate = remote_addr
                    else:
                        candidate = chain[0]
            # Optionally honor X-Real-IP only when supplied by a trusted proxy.
            elif self.trust_x_forwarded_for and real_ip and self._is_trusted_proxy(remote_addr):
                candidate = real_ip.strip() or remote_addr
        except Exception:
            logger.debug("Failed to parse X-Forwarded-For header")
        return candidate

    def is_allowed(self, ip_str: str | None) -> bool:
        """Return True if the resolved IP passes the allow/block rules."""
        if not ip_str:
            # Unknown IP - only allow when no allowlist is configured.
            return not self.allowed_networks
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            logger.warning(f"Received request with unparsable IP '{ip_str}'")
            return not self.allowed_networks

        # Explicit block overrides everything
        for network in self.blocked_networks:
            if ip_obj in network:
                return False

        # Allowlist: if empty -> allow all (unless blocked). Otherwise require membership.
        if not self.allowed_networks:
            return True

        return any(ip_obj in network for network in self.allowed_networks)


@lru_cache(maxsize=1)
def get_ip_access_controller() -> IPAccessController:
    """Instantiate a cached IP access controller using current configuration."""
    cfg = get_config()
    return IPAccessController(
        allowed=tuple(cfg.allowed_client_ips or []),
        blocked=tuple(cfg.blocked_client_ips or []),
        trust_x_forwarded_for=bool(cfg.trust_x_forwarded_for),
        trusted_proxy_depth=int(cfg.trusted_proxy_depth or 0),
        trusted_proxies=tuple(cfg.trusted_proxy_ips or []),
    )


async def enforce_ip_allowlist(request: Request) -> None:
    """
    FastAPI dependency that enforces the configured IP allow/deny rules.

    Raises:
        HTTPException: when the request should be rejected.
    """
    controller = get_ip_access_controller()
    client = request.client
    remote_ip = client.host if client else None
    forwarded_for = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    real_ip = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
    resolved_ip = controller.resolve_client_ip(remote_ip, forwarded_for, real_ip)

    # Test harnesses (FastAPI TestClient / pytest) often use the synthetic host
    # name "testclient" which is not a valid IP address. Treat it as loopback so
    # that unit tests are not blocked by the allowlist when TEST_MODE/pytest
    # execution is detected.
    if resolved_ip == "testclient":
        resolved_ip = "127.0.0.1"
    else:
        try:
            import os as _os
            if (resolved_ip is None and (
                _os.getenv("PYTEST_CURRENT_TEST") or
                is_test_mode()
            )):
                resolved_ip = "127.0.0.1"
        except Exception as loopback_error:
            logger.debug("MCP IP filter loopback normalization failed", exc_info=loopback_error)

    if not controller.is_allowed(resolved_ip):
        logger.warning(
            f"Rejecting MCP request from disallowed IP {resolved_ip or 'unknown'}",
            extra={"audit": True, "ip": resolved_ip, "path": request.url.path},
        )
        raise HTTPException(status_code=403, detail="Client IP not allowed")
