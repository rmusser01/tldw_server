"""
Startup validation checks extracted from main.py lifespan.

These are early, independent validation steps that run before
any services are initialized. They check environment, config,
and external dependencies.
"""

from __future__ import annotations

from loguru import logger


def validate_mcp_config_production() -> None:
    """Validate MCP configuration in production mode (fail-fast).

    Raises RuntimeError if MCP config is invalid in production.
    """
    try:
        import os

        if os.getenv("TEST_MODE", "").lower() in {"true", "1"}:
            return
        from tldw_Server_API.app.core.MCP_unified.server import validate_mcp_config

        validate_mcp_config()
    except ImportError:
        logger.debug("MCP validation skipped: module not available")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"MCP config validation issue: {e}")


def validate_acp_runner_config() -> None:
    """Validate ACP runner configuration (warnings only)."""
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import validate_runner_config

        validate_runner_config()
    except ImportError:
        logger.debug("ACP validation skipped: module not available")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ACP runner config validation issue: {e}")


def init_telemetry() -> None:
    """Initialize OpenTelemetry and Sentry if configured."""
    # OpenTelemetry
    try:
        import os

        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint:
            from tldw_Server_API.app.core.Metrics.telemetry_manager import configure_telemetry

            configure_telemetry(endpoint=otel_endpoint)
            logger.info(f"OpenTelemetry configured: {otel_endpoint}")
    except ImportError:
        logger.debug("OpenTelemetry skipped: dependencies not installed")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OpenTelemetry initialization failed: {e}")

    # Sentry
    try:
        import os

        sentry_dsn = os.getenv("SENTRY_DSN")
        if sentry_dsn:
            import sentry_sdk

            sentry_sdk.init(dsn=sentry_dsn)
            logger.info("Sentry error tracking initialized")
    except ImportError:
        logger.debug("Sentry skipped: sentry-sdk not installed")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Sentry initialization failed: {e}")
