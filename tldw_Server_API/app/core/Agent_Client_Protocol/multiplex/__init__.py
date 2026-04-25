"""Public exports for ACP multiplexing primitives.

This package re-exports the multiplex protocol types and manager used by the
ACP multi-session WebSocket endpoint.
"""

from .manager import MultiplexManager
from .protocol import MultiplexMessage, MultiplexMessageType

__all__ = ["MultiplexManager", "MultiplexMessage", "MultiplexMessageType"]
