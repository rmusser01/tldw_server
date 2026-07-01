from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable

from .models import ResolvedAddress


def is_unsafe_egress_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved


class StdlibResolver:
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        seen: set[str] = set()
        addresses: list[ResolvedAddress] = []
        for family, _socket_type, _protocol, _canonical_name, sockaddr in results:
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            ip_text = str(sockaddr[0])
            if ip_text in seen:
                continue
            seen.add(ip_text)
            addresses.append(ResolvedAddress(host=host, ip=ip_text, port=port, family=family))
        return addresses


__all__ = ["StdlibResolver", "is_unsafe_egress_ip"]
