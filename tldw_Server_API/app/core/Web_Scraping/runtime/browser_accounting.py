"""Bounded request-byte reconciliation shared by the two browser owners."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from threading import Lock


def non_negative_integer(value: object) -> int:
    """Validate native transfer lengths without coercion or lossy rounding."""
    if type(value) is int:
        if value < 0:
            raise ValueError("invalid transfer length")
        return value
    if type(value) is not float:
        raise TypeError("invalid transfer length")
    if not math.isfinite(value) or value < 0 or not value.is_integer() or value > 2**53 - 1:
        raise ValueError("invalid transfer length")
    return int(value)


class RequestTransfers:
    """Charge chunks and only the unobserved remainder of a cumulative finish."""

    def __init__(self, capacity: int, charge: Callable[[int], None]) -> None:
        self.capacity, self.charge = capacity, charge
        self._lock = Lock()
        self._bytes: dict[tuple[str, str], int] = {}
        self._finished: set[tuple[str, str]] = set()

    @staticmethod
    def key(session: str, payload: object) -> tuple[str, str]:
        if not isinstance(payload, Mapping):
            raise TypeError("invalid CDP event envelope")
        identifier = payload.get("requestId")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 1024:
            raise ValueError("invalid request identifier")
        return session, identifier

    def data(self, session: str, payload: Mapping) -> None:
        key = self.key(session, payload)
        amount = non_negative_integer(payload.get("encodedDataLength"))
        if amount == 0:
            amount = non_negative_integer(payload.get("dataLength"))
        with self._lock:
            if key in self._finished:
                return
            if key not in self._bytes and len(self._bytes) >= self.capacity:
                raise ValueError("request observation capacity")
            self._bytes[key] = self._bytes.get(key, 0) + amount
        self.charge(amount)

    def finished(self, session: str, payload: Mapping) -> None:
        key = self.key(session, payload)
        total = non_negative_integer(payload.get("encodedDataLength"))
        with self._lock:
            if key in self._finished:
                return
            charged = self._bytes.pop(key, 0)
            if len(self._finished) >= self.capacity:
                self._finished.pop()
            self._finished.add(key)
        self.charge(max(0, total - charged))

    def failed(self, session: str, payload: Mapping) -> None:
        key = self.key(session, payload)
        with self._lock:
            self._bytes.pop(key, None)
