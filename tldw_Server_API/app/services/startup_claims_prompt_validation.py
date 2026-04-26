"""
Claims prompt startup validation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def validate_startup_claims_prompt_validation(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        claims_prompt_validation_error = _get_claims_prompt_validation_error()
        try:
            claims_settings = _get_claims_settings()
            claims_prompt_report = _validate_claims_prompt_preflight(claims_settings)
            if (
                _claims_prompt_report_has_issues(claims_prompt_report)
                and claims_prompt_report.mode != "off"
            ):
                logger.warning(
                    "App Startup: Claims prompt validation found {} issue(s) (mode={}, strict={})",
                    len(claims_prompt_report.issues),
                    claims_prompt_report.mode,
                    claims_prompt_report.strict,
                )
            else:
                logger.info(
                    "App Startup: Claims prompt validation completed (mode={}, strict={})",
                    claims_prompt_report.mode,
                    claims_prompt_report.strict,
                )
        except claims_prompt_validation_error:
            logger.exception("Startup aborted due to claims prompt validation error")
            raise
    except startup_guard_exceptions as exc:
        logger.debug("App Startup: Claims prompt validation skipped/failed: {}", exc)


def _get_claims_prompt_validation_error() -> type[BaseException]:
    from tldw_Server_API.app.core.Claims_Extraction.prompt_validation import (
        ClaimsPromptValidationError,
    )

    return ClaimsPromptValidationError


def _claims_prompt_report_has_issues(report: Any) -> bool:
    from tldw_Server_API.app.core.Claims_Extraction.prompt_validation import (
        claims_prompt_report_has_issues,
    )

    return bool(claims_prompt_report_has_issues(report))


def _validate_claims_prompt_preflight(settings: Any) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.prompt_validation import (
        validate_claims_prompt_preflight,
    )

    return validate_claims_prompt_preflight(settings)


def _get_claims_settings() -> Any:
    from tldw_Server_API.app.core.config import settings

    return settings
