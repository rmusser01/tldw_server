from __future__ import annotations

from typing import Any


def get_transcription_models_payload() -> dict[str, Any]:
    """
    Return the static transcription models payload used by /media endpoints.

    Centralizing this here allows both the legacy and modular endpoints
    to share the same definitions without duplicating data.
    """
    models_by_category: dict[str, list[dict[str, str]]] = {
        "Whisper Models": [
            {
                "value": "whisper-tiny",
                "label": "Whisper Tiny (39M)",
                "description": "Fastest, least accurate",
            },
            {
                "value": "whisper-tiny.en",
                "label": "Whisper Tiny English (39M)",
                "description": "English only, faster",
            },
            {
                "value": "whisper-base",
                "label": "Whisper Base (74M)",
                "description": "Fast, good accuracy",
            },
            {
                "value": "whisper-base.en",
                "label": "Whisper Base English (74M)",
                "description": "English only, better",
            },
            {
                "value": "whisper-small",
                "label": "Whisper Small (244M)",
                "description": "Balanced speed/accuracy",
            },
            {
                "value": "whisper-small.en",
                "label": "Whisper Small English (244M)",
                "description": "English only, recommended",
            },
            {
                "value": "whisper-medium",
                "label": "Whisper Medium (769M)",
                "description": "Good accuracy, slower",
            },
            {
                "value": "whisper-medium.en",
                "label": "Whisper Medium English (769M)",
                "description": "English only, high quality",
            },
            {
                "value": "whisper-large-v1",
                "label": "Whisper Large v1 (1550M)",
                "description": "Original large model",
            },
            {
                "value": "whisper-large-v2",
                "label": "Whisper Large v2 (1550M)",
                "description": "Improved large model",
            },
            {
                "value": "whisper-large-v3",
                "label": "Whisper Large v3 (1550M)",
                "description": "Latest, most accurate",
            },
            {
                "value": "whisper-large-v3-turbo",
                "label": "Whisper Large v3 Turbo",
                "description": "Faster large model",
            },
        ],
        "Distil-Whisper Models": [
            {
                "value": "distil-whisper-small.en",
                "label": "Distil-Whisper Small English",
                "description": "6x faster, similar accuracy",
            },
            {
                "value": "distil-whisper-medium.en",
                "label": "Distil-Whisper Medium English",
                "description": "6x faster, good quality",
            },
            {
                "value": "distil-whisper-large-v2",
                "label": "Distil-Whisper Large v2",
                "description": "5.8x faster",
            },
            {
                "value": "distil-whisper-large-v3",
                "label": "Distil-Whisper Large v3",
                "description": "Latest distilled model",
            },
        ],
        "Optimized Models": [
            {
                "value": "deepdml/faster-distil-whisper-large-v3.5",
                "label": "Faster Distil-Whisper Large v3.5",
                "description": "Recommended optimized model",
            },
            {
                "value": "deepdml/faster-whisper-large-v3-turbo-ct2",
                "label": "Faster Whisper Large v3 Turbo CT2",
                "description": "Turbo large model (CTranslate2)",
            },
            {
                "value": "whisper-tiny-ct2",
                "label": "Whisper Tiny CT2",
                "description": "CTranslate2 optimized",
            },
            {
                "value": "whisper-base-ct2",
                "label": "Whisper Base CT2",
                "description": "CTranslate2 optimized",
            },
            {
                "value": "whisper-small-ct2",
                "label": "Whisper Small CT2",
                "description": "CTranslate2 optimized",
            },
            {
                "value": "whisper-medium-ct2",
                "label": "Whisper Medium CT2",
                "description": "CTranslate2 optimized",
            },
            {
                "value": "whisper-large-v2-ct2",
                "label": "Whisper Large v2 CT2",
                "description": "CTranslate2 optimized",
            },
            {
                "value": "whisper-large-v3-ct2",
                "label": "Whisper Large v3 CT2",
                "description": "CTranslate2 optimized",
            },
        ],
        "Nemo Models": [
            {
                "value": "nemo-canary-1b",
                "label": "Nemo Canary 1B",
                "description": "NVIDIA's multilingual model",
            },
            {
                "value": "nemo-parakeet-0.11b",
                "label": "Nemo Parakeet 0.11B",
                "description": "Lightweight model",
            },
            {
                "value": "nemo-parakeet-1.1b",
                "label": "Nemo Parakeet 1.1B",
                "description": "Standard model",
            },
            {
                "value": "nemo-parakeet-tdt-1.1b",
                "label": "Nemo Parakeet TDT 1.1B",
                "description": "Timestamped model",
            },
        ],
        "Parakeet Backends": [
            {
                "value": "parakeet-tdt-0.6b-v3-onnx",
                "label": "Parakeet TDT 0.6B V3 ONNX",
                "description": "Canonical cross-platform ONNX default",
            },
            {
                "value": "parakeet-standard",
                "label": "Parakeet Standard",
                "description": "Default CPU backend",
            },
            {
                "value": "parakeet-cuda",
                "label": "Parakeet CUDA",
                "description": "GPU acceleration (NVIDIA)",
            },
            {
                "value": "parakeet-mlx",
                "label": "Parakeet MLX",
                "description": "Apple Silicon acceleration",
            },
            {
                "value": "parakeet-onnx",
                "label": "Parakeet ONNX (legacy alias)",
                "description": "Compatibility alias for the canonical ONNX default",
            },
        ],
        "VibeVoice-ASR": [
            {
                "value": "vibevoice-asr",
                "label": "VibeVoice-ASR (7B)",
                "description": "Long-form ASR with diarization + hotwords",
            },
            {
                "value": "microsoft/VibeVoice-ASR",
                "label": "microsoft/VibeVoice-ASR",
                "description": "Hugging Face model id (local or vLLM HTTP)",
            },
        ],
        "Qwen3-ASR": [
            {
                "value": "qwen3-asr-1.7b",
                "label": "Qwen3-ASR 1.7B",
                "description": "Production quality, 30 languages + Chinese dialects",
            },
            {
                "value": "qwen3-asr-0.6b",
                "label": "Qwen3-ASR 0.6B",
                "description": "High-throughput, resource-constrained",
            },
        ],
    }

    flat_values: list[str] = []
    for category_models in models_by_category.values():
        flat_values.extend(model["value"] for model in category_models)

    return {
        "categories": models_by_category,
        "all_models": flat_values,
    }


__all__ = ["get_transcription_models_payload"]
