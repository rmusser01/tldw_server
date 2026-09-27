"""Shared base for the read-only web MCP tools (web.fetch / web.search / web.research).

Centralizes the behavior these tools had duplicated: execution eval metadata, the
structured error-result shape, the profile-id context reader, and the domain-list
validator.

These tools also carried a permissive ``sanitize_input`` override, because the
base sanitizer used to reject any string containing ``--``/``/*``/``*/`` -- which
is most real URLs, search queries and punycode domains. That denylist is gone
(TASK-13294) and the base now does exactly what this override did, so the
override has been removed rather than kept in sync.
"""

from __future__ import annotations

from typing import Any

# ``CONTROL_CHARS_RE`` is re-exported: the base owns the one definition, and
# web_fetch_module imports it from this path to reject control chars in URLs.
from ..base import (  # re-exported for module convenience
    CONTROL_CHARS_RE,
    BaseModule,
    create_tool_definition,
)

__all__ = ["CONTROL_CHARS_RE", "WebToolBase", "WebToolError", "create_tool_definition"]


class WebToolError(Exception):
    """Internal control-flow error carrying a structured reason code.

    Shared by all web tools so the domain-list validator and per-tool validation
    can raise a single type that ``execute_tool`` maps to a structured result.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class WebToolBase(BaseModule):
    """Base class with the shared plumbing for read-only web tools.

    Subclasses set the three eval-metadata class attributes and use
    :meth:`_structured_error` / :meth:`_eval_metadata` / :meth:`_validate_domain_list`.
    """

    # Eval-metadata identity; subclasses override.
    _ACTION_FAMILY: str = "web"
    _RESULT_KIND: str = "web_result"
    _TOOL_PROMPT_VERSION: str = "2026.06.14"

    def _structured_error(
        self,
        tool_name: str,
        reason_code: str,
        message: str,
        *,
        context: Any | None = None,
        truncated: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Return the shared ``{ok: false, reason_code, message, eval}`` payload.

        Extra keyword fields (e.g. ``status_code``) are merged into the result.
        """
        result: dict[str, Any] = {
            "ok": False,
            "reason_code": reason_code,
            "message": message,
            "eval": self._eval_metadata(
                tool_name, reason_code=reason_code, truncated=truncated, context=context
            ),
        }
        for key, value in extra.items():
            # Never let caller-supplied extras overwrite the core error fields.
            if value is not None and key not in result:
                result[key] = value
        return result

    def _eval_metadata(
        self,
        tool_name: str,
        *,
        reason_code: str | None,
        truncated: bool = False,
        context: Any | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.core.MCP_unified.tool_observability import (
            build_execution_eval_metadata,
        )

        return build_execution_eval_metadata(
            tool_name=tool_name,
            tool_prompt_id=f"mcp.{tool_name}.v1",
            tool_prompt_version=self._TOOL_PROMPT_VERSION,
            action_family=self._ACTION_FAMILY,
            result_kind=self._RESULT_KIND,
            profile_id=self._profile_id_from_context_metadata(context),
            path_filter_used=False,
            truncated=truncated,
            reason_code=reason_code,
        )

    @staticmethod
    def _profile_id_from_context_metadata(context: Any | None) -> str | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        for key in ("profile_id", "selected_profile_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _validate_domain_list(args: dict[str, Any], name: str) -> list[str] | None:
        """Validate an optional list-of-domains argument.

        Returns ``None`` for absent or empty/whitespace lists (so providers do
        not read an empty list as "exclude everything"); raises
        :class:`WebToolError` for malformed input.
        """
        value = args.get(name)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise WebToolError("invalid_arguments", f"{name} must be a list of non-empty strings")
        stripped = [item.strip() for item in value]
        return stripped or None
