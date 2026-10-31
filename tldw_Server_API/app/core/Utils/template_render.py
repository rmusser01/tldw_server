"""Sandboxed Jinja rendering for prompt and workflow templates (TASK-13950.13).

Moved from core/Chat/prompt_template_manager.py, which re-exports both public names, so
that Workflows and other non-chat code can render templates without importing core.Chat.
"""

from typing import Any, Optional

from jinja2.sandbox import SandboxedEnvironment
from loguru import logger

_SANDBOX = SandboxedEnvironment(
    autoescape=True,                # HTML-safe by default
    enable_async=False,             # no await, no async callables
)

def _normalize_template_syntax(template_str: str) -> str:
    """
    Best-effort normalization to support common JS-like operators inside Jinja blocks.
    Converts logical '||' to 'or' and '&&' to 'and' within {{ ... }} or {% ... %} segments.
    Leaves content outside of Jinja blocks untouched.
    """
    import re as _re

    def _norm_block(prefix: str, suffix: str, s: str) -> str:
        pattern = _re.compile(_re.escape(prefix) + r"\s*(.*?)\s*" + _re.escape(suffix), _re.DOTALL)

        def repl(m):
            inner = m.group(1)
            inner = inner.replace("||", " or ").replace("&&", " and ")
            return f"{prefix} {inner} {suffix}"

        return pattern.sub(repl, s)

    # Normalize expression and statement blocks
    out = template_str
    out = _norm_block("{{", "}}", out)
    out = _norm_block("{%", "%}", out)
    return out


def safe_render(template_str: str, data: dict[str, Any]) -> str:
    """Render with a locked-down Jinja sandbox (with light syntax normalization)."""
    try:
        normalized = _normalize_template_syntax(template_str)
        tmpl = _SANDBOX.from_string(normalized)
        return tmpl.render(**data)
    except Exception as exc:  # noqa: BLE001 - template render boundary (moved from prompt_template_manager)
        logger.error("Template render error {}", exc, exc_info=False)
        return template_str      # fail closed: return raw


def apply_template_to_string(template_string: Optional[str], data: dict[str, Any]) -> Optional[str]:
    """
    Applies data to a template string using Jinja2 safe rendering.
    Missing placeholders will typically render as empty strings by Jinja2 default.
    """
    if template_string is None:
        return "" # Returns an empty string if the template_string itself is None
    try:
        # The original was: template_string = safe_render(template_string, data)
        # This needs to assign to a new variable and return it.
        rendered_string = safe_render(template_string, data)
        return rendered_string
    except KeyError as e: # This exception type might not be commonly raised by Jinja's render for missing vars
        logger.warning(f"Placeholder {e} not found in data for template string: '{template_string}'")
        return template_string # Fallback to original
    except Exception as e:  # noqa: BLE001 - template render boundary (moved from prompt_template_manager)
        logger.error(f"Error applying template string '{template_string}': {e}", exc_info=True)
        return template_string # Return original on error
