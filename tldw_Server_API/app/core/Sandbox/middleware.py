from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def _has_traversal(p: str) -> bool:
    """Return True when a path contains a `..` traversal segment."""
    return "/../" in p or p.endswith("/..") or p.startswith("../")


def _is_sandbox_runs(p: str) -> bool:
    """Return True for paths under the sandbox runs prefix."""
    return p.startswith("/api/v1/sandbox/runs/")


class SandboxArtifactTraversalGuardMiddleware:
    """Reject path traversal attempts for Sandbox artifact routes before routing.

    Specifically targets raw `..` segments under `/api/v1/sandbox/runs/{id}/artifacts/...`.
    Returns HTTP 400 on detection.

    Implemented as pure ASGI rather than ``BaseHTTPMiddleware``: this guard runs
    on every request in the application but only acts on one route prefix, and a
    ``BaseHTTPMiddleware`` layer costs ~0.08 ms per request regardless of what it
    does, because of the anyio task group and memory object streams it sets up.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            try:
                path = scope.get("path") or ""
                raw_path_b = scope.get("raw_path") or b""
                raw_path = (
                    raw_path_b.decode("latin-1", errors="ignore")
                    if isinstance(raw_path_b, (bytes, bytearray))
                    else str(raw_path_b or "")
                )
                # Prefer raw_path for detection when available, fallback to normalized path.
                # Reject traversal anywhere under sandbox runs (defense in depth).
                for p in (raw_path, path):
                    if p and _is_sandbox_runs(p) and _has_traversal(p):
                        response = JSONResponse(
                            {"detail": "Path traversal detected"}, status_code=400
                        )
                        await response(scope, receive, send)
                        return
            except Exception as guard_error:
                # Never fail a request due to guard errors
                _ = guard_error
        await self.app(scope, receive, send)
