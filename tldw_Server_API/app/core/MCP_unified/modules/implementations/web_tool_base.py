"""Shared base for the read-only web MCP tools (web.fetch / web.search / web.research).

Centralizes the behavior these tools had duplicated: a permissive
``sanitize_input`` override (the MCP protocol sanitizes every tool call before
execution, and the base SQL denylist wrongly rejects legitimate URLs/queries
containing ``--``/``/*``/punycode), execution eval metadata, the structured
error-result shape, the profile-id context reader, and the domain-list
validator.
"""

from __future__ import annotations

import re
from typing import Any

from ..base import BaseModule, create_tool_definition  # re-exported for module convenience

__all__ = ["CONTROL_CHARS_RE", "WebToolBase", "WebToolError", "create_tool_definition"]

# Shared across the web tools: stripped from sanitized inputs, and rejected
# outright in URL validation.
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_SANITIZE_DEPTH = 20


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

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """Permissive sanitizer that strips only NUL/control characters.

        The base ``sanitize_input`` rejects strings containing SQL-injection
        substrings (``--``, ``/*``, ``*/``), but those appear constantly in
        legitimate URLs, search queries (``pip install --no-cache-dir``), and
        punycode domains (``xn--...``). The MCP protocol runs this on every tool
        call before execution, so without the override such inputs would be
        rejected with a protocol ``InvalidParams`` error. We keep only the NUL/
        control-char stripping and a depth guard.
        """
        if _depth > _MAX_SANITIZE_DEPTH:
            raise ValueError("Input too deeply nested")
        if isinstance(input_data, str):
            return CONTROL_CHARS_RE.sub("", input_data)
        if isinstance(input_data, dict):
            return {key: self.sanitize_input(value, _depth + 1) for key, value in input_data.items()}
        if isinstance(input_data, list):
            return [self.sanitize_input(value, _depth + 1) for value in input_data]
        return input_data

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
