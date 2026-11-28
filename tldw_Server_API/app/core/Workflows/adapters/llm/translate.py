"""Translate adapter.

This adapter handles text translation using LLM.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal, LLMQuotaExceededError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.llm._config import TranslateConfig


@registry.register(
    "translate",
    category="llm",
    description="Translate text using configured chat provider",
    parallelizable=True,
    tags=["translation", "language"],
    config_model=TranslateConfig,
)
async def run_translate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Translate text using configured chat provider (best-effort), or no-op in test.

    Config:
      - input: str (templated) or defaults to last.text
      - target_lang: str (e.g., 'en', 'fr')
      - provider/model: optional hints

    Output: { text: translated_text, target_lang, provider? }
    """
    from tldw_Server_API.app.core.Utils.template_render import apply_template_to_string as _tmpl

    txt_t = str(config.get("input") or "").strip()
    if txt_t:
        text = _tmpl(txt_t, context) or txt_t
    else:
        prev = context.get("prev") or {}
        text = str(prev.get("text") or prev.get("content") or "")
    target = str(config.get("target_lang") or "en").strip()
    if not text:
        return {"error": "missing_input_text"}

    # Test mode no-op
    if is_test_mode():
        return {"text": text, "target_lang": target, "simulated": True}

    # Try the gateway (openai) first; fall back to returning input. A policy refusal fails the step.
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config, resolve_provider_model

        system = f"You are a professional translator. Translate the user text to {target}. Preserve meaning and tone. Output only the translation."
        resp = await perform_chat_api_call_async(
            api_provider="openai",
            messages=[{"role": "user", "content": text}],
            system_message=system,
            model=resolve_provider_model("openai", ensure_app_config()),
            llm_surface="workflows.step",
        )
        out = extract_openai_content(resp)
        if not out:
            return {"text": text, "target_lang": target, "provider": "openai", "fallback": True}
        return {"text": out, "target_lang": target, "provider": "openai"}
    except (LLMQuotaExceededError, LLMPolicyRefusal):
        raise
    except Exception:
        # Fallback: return original
        return {"text": text, "target_lang": target, "provider": "none", "fallback": True}
