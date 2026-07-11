"""Best-effort runtime diagnostics for yt-dlp support."""

from importlib import metadata
from threading import Lock

from loguru import logger


MINIMUM_YT_DLP_VERSION = "2026.7.4"
_UPDATE_COMMAND = f'pip install -U "yt-dlp>={MINIMUM_YT_DLP_VERSION}"'
_check_lock = Lock()
_checked = False


def warn_if_yt_dlp_is_stale() -> None:
    """Warn once when installed yt-dlp is below the supported version floor."""
    global _checked

    with _check_lock:
        if _checked:
            return
        _checked = True

    try:
        from packaging.version import Version

        installed_version = metadata.version("yt-dlp")
        if Version(installed_version) < Version(MINIMUM_YT_DLP_VERSION):
            logger.warning(
                f"Installed yt-dlp version {installed_version} is below the supported "
                f"minimum {MINIMUM_YT_DLP_VERSION}. Update it with: {_UPDATE_COMMAND}"
            )
    except Exception:  # noqa: BLE001 - diagnostics must never block ingestion
        return
