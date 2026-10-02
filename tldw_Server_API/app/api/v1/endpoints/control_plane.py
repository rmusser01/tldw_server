"""Stable control-plane callbacks, independent of an application instance.

FastAPI caches callable classification. Defining these in main makes a cached
function's globals retain every retired main app after a module reload.
"""

from pathlib import Path

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger
from starlette.responses import FileResponse

from tldw_Server_API.app.core.config import route_enabled
from tldw_Server_API.app.core.Metrics import get_metrics_registry, track_metrics
from tldw_Server_API.app.core.Setup.setup_manager import needs_setup
from tldw_Server_API.app.services import readiness_service

BASE_DIR = Path(__file__).resolve().parents[3]
FAVICON_PATH = BASE_DIR / "static" / "favicon.ico"
SETUP_PAGE_PATH = BASE_DIR / "Setup_UI" / "setup.html"
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_REQUEST_GUARD_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


async def serve_setup_page():
    """Serve the first-time setup UI when required."""
    try:
        setup_required = needs_setup()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file missing; cannot render setup UI.") from None

    if not setup_required:
        return RedirectResponse(url="/api/v1/config/quickstart", status_code=307)

    if not SETUP_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Setup UI assets missing. Reinstall the setup UI bundle.")

    return FileResponse(SETUP_PAGE_PATH)


async def favicon():
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


async def root():
    try:
        if needs_setup():
            try:
                if route_enabled("setup"):
                    return RedirectResponse(url="/setup", status_code=307)
            except _REQUEST_GUARD_EXCEPTIONS:
                pass
    except FileNotFoundError:
        logger.warning("config.txt missing while handling root request; serving default message.")

    return {
        "message": "Welcome to the tldw API; if you're seeing this, the server is running! "
        "Check out /api/v1/config/quickstart, /docs, or /metrics to get started."
    }


async def metrics():
    from tldw_Server_API.app.api.v1.endpoints.metrics import build_prometheus_metrics_response

    return await build_prometheus_metrics_response()


@track_metrics(name="tldw_Server_API.app.main.api_metrics", labels={"endpoint": "metrics"})
async def api_metrics():
    """Get current metrics in JSON format."""
    registry = get_metrics_registry()
    return registry.get_all_metrics()


async def _set_diagnostics_no_store(response: Response) -> None:
    """Prevent caching for direct dictionary-based diagnostic aliases."""
    response.headers["Cache-Control"] = "no-store"


async def health_check() -> JSONResponse:
    """Return the immutable public liveness contract."""
    return JSONResponse({"status": "ok"}, headers=_NO_STORE_HEADERS)


async def internal_readiness_check(request: Request) -> JSONResponse:
    """Serve only loopback, detail-free readiness for local orchestrators."""
    if not readiness_service.is_loopback_peer(request):
        return JSONResponse({"detail": "Not Found"}, status_code=404, headers=_NO_STORE_HEADERS)
    snapshot = await readiness_service.collect_readiness_snapshot(request.app)
    return JSONResponse(
        readiness_service.internal_readiness_payload(snapshot),
        status_code=200 if snapshot.ready else 503,
        headers=_NO_STORE_HEADERS,
    )


async def readiness_check(request: Request) -> JSONResponse:
    """Return the authenticated operator readiness projection."""
    snapshot = await readiness_service.collect_readiness_snapshot(request.app)
    return JSONResponse(
        readiness_service.operator_readiness_payload(snapshot),
        status_code=200 if snapshot.ready else 503,
        headers=_NO_STORE_HEADERS,
    )


async def readiness_alias(request: Request) -> JSONResponse:
    return await readiness_check(request)
