"""Resolve client IPs without trusting headers from untrusted network peers."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def _parse_ip(value: str | None) -> IPAddress | None:
    """Parse a plain IPv4 or IPv6 address, rejecting scoped and malformed values."""
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or "%" in token:
        return None
    try:
        return ipaddress.ip_address(token)
    except ValueError:
        return None


def _parse_networks(entries: Iterable[str]) -> tuple[IPNetwork, ...]:
    """Parse valid trusted-proxy host and network entries, ignoring invalid ones."""
    networks: list[IPNetwork] = []
    for entry in entries:
        token = str(entry).strip()
        if not token:
            continue
        try:
            networks.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _address_is_trusted(address: IPAddress, networks: tuple[IPNetwork, ...]) -> bool:
    """Return whether an address belongs to any same-family trusted network."""
    return any(address.version == network.version and address in network for network in networks)


def is_trusted_proxy_peer(
    physical_peer: str | None,
    trusted_proxy_entries: Iterable[str],
) -> bool:
    """Return whether the physical network peer is configured as trusted."""
    peer = _parse_ip(physical_peer)
    return peer is not None and _address_is_trusted(peer, _parse_networks(trusted_proxy_entries))


def resolve_trusted_client_ip(
    physical_peer: str | None,
    trusted_proxy_entries: Iterable[str],
    *,
    forwarded_for_values: Iterable[str] = (),
    single_forwarded_value: str | None = None,
) -> str | None:
    """Resolve the client address, honoring forwarding headers only from trusted peers."""
    peer = _parse_ip(physical_peer)
    if peer is None:
        return None
    networks = _parse_networks(trusted_proxy_entries)
    if not _address_is_trusted(peer, networks):
        return peer.compressed

    xff_values = tuple(str(value) for value in forwarded_for_values)
    if xff_values:
        parsed_chain: list[IPAddress] = []
        for token in ",".join(xff_values).split(","):
            parsed = _parse_ip(token)
            if parsed is None:
                return peer.compressed
            parsed_chain.append(parsed)
        for address in reversed(parsed_chain):
            if not _address_is_trusted(address, networks):
                return address.compressed
        return peer.compressed

    if single_forwarded_value is not None:
        forwarded = _parse_ip(single_forwarded_value)
        return forwarded.compressed if forwarded is not None else peer.compressed
    return peer.compressed
