from __future__ import annotations

ALLOWED_PARAKEET_VARIANTS = {"standard", "onnx", "mlx"}
CANONICAL_PARAKEET_ONNX_ALIASES = {"parakeet-tdt-0.6b-v3-onnx"}


def normalize_model_and_variant(
    raw_model: str | None,
    current_model: str,
    current_variant: str,
    variant_override: str | None = None,
) -> tuple[str, str]:
    """
    Normalize streaming STT model + variant selection from potentially combined identifiers.

    Rules (kept in parity with unified WebSocket handler expectations):
    - If model is a known Parakeet ONNX alias (e.g., "parakeet-tdt-0.6b-v3-onnx"
      or "parakeet-onnx"):
      - When base is "parakeet" and no explicit override is given, set model="parakeet"
        and model_variant to the suffix (only if recognized), else keep current variant.
      - For non-parakeet bases (e.g., "whisper-1", "canary-1b"), collapse to base model
        ("whisper", "canary"). Suffix is ignored.
    - If an explicit override (variant/model_variant) is provided and the target model is
      Parakeet, the override wins.
    - If raw_model is None, only apply variant override when current model is Parakeet.
    """

    model_out = current_model
    variant_out = current_variant

    if raw_model is not None:
        s = str(raw_model)
        lowered = s.lower()
        if lowered in CANONICAL_PARAKEET_ONNX_ALIASES:
            model_out = "parakeet"
            variant_out = str(variant_override).lower() if variant_override else "onnx"
            return model_out, variant_out

        base, sep, suffix = s.partition("-")
        base_lower = base.lower()

        if base_lower == "parakeet":
            model_out = "parakeet"
            if variant_override:
                variant_out = str(variant_override).lower()
            elif sep and suffix:
                v = suffix.lower()
                if v in ALLOWED_PARAKEET_VARIANTS:
                    variant_out = v
                # else: keep existing variant
        elif base_lower in ("qwen3", "qwen3_asr", "qwen3asr"):
            # Normalize all qwen3-asr variants to canonical form
            model_out = "qwen3-asr"
            # qwen3-asr does not have variants; ignore variant_override
        else:
            # For non-Parakeet hyphenated names, collapse to base model to match selector logic
            model_out = base if sep else s
            # Don't apply Parakeet variant overrides to non-Parakeet models
            # variant_out unchanged
    elif variant_override and current_model.lower() == "parakeet":
        variant_out = str(variant_override).lower()

    return model_out, variant_out


__all__ = [
    "normalize_model_and_variant",
    "ALLOWED_PARAKEET_VARIANTS",
    "CANONICAL_PARAKEET_ONNX_ALIASES",
]
