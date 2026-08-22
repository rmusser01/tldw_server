"""Production launcher for Uvicorn's guarded MCP WebSocket protocol."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn

from tldw_Server_API.app.core.MCP_unified.transport.guarded_slides_websocket import (
    GuardedSlidesWebSocketProtocol,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),  # nosec
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("UVICORN_WORKERS", "1")),
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    parser.add_argument(
        "--proxy-headers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse supported deployment options and start the guarded server."""

    args = _parser().parse_args(argv)
    uvicorn.run(
        "tldw_Server_API.app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        proxy_headers=args.proxy_headers,
        forwarded_allow_ips=args.forwarded_allow_ips,
        ws=GuardedSlidesWebSocketProtocol,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
