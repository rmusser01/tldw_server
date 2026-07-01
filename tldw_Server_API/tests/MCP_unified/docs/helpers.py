from __future__ import annotations

from collections.abc import Iterable

from mcp_unified.docs.acquisition.models import FetchResponse, ResolvedAddress, URLRequest


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        return [ResolvedAddress(host=host, ip=ip, port=port) for ip in self.addresses[host]]


class FakeTransport:
    dials_validated_address = True

    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[ResolvedAddress, URLRequest, float]] = []

    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse:
        self.calls.append((address, request, timeout_seconds))
        return self.responses.pop(0)
