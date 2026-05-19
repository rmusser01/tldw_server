import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.model_utils import (
    normalize_model_and_variant,
    ALLOWED_PARAKEET_VARIANTS,
)


@pytest.mark.unit
def test_parakeet_combined_sets_variant_when_no_override():
    m, v = normalize_model_and_variant(
        raw_model="parakeet-onnx",
        current_model="parakeet",
        current_variant="standard",
        variant_override=None,
    )
    assert m == "parakeet"
    assert v == "onnx"


@pytest.mark.unit
def test_parakeet_canonical_tdt_onnx_alias_sets_variant_when_no_override():
    m, v = normalize_model_and_variant(
        raw_model="parakeet-tdt-0.6b-v3-onnx",
        current_model="parakeet",
        current_variant="standard",
        variant_override=None,
    )
    assert m == "parakeet"
    assert v == "onnx"


@pytest.mark.unit
def test_parakeet_override_wins_over_combined_suffix():
    m, v = normalize_model_and_variant(
        raw_model="parakeet-mlx",
        current_model="parakeet",
        current_variant="standard",
        variant_override="onnx",
    )
    assert m == "parakeet"
    assert v == "onnx"


@pytest.mark.unit
def test_whisper_hyphenated_collapses_to_base():
    m, v = normalize_model_and_variant(
        raw_model="whisper-1",
        current_model="parakeet",
        current_variant="standard",
        variant_override=None,
    )
    assert m == "whisper"
    # variant remains unchanged for non-parakeet
    assert v == "standard"


@pytest.mark.unit
def test_canary_hyphenated_collapses_to_base():
    m, v = normalize_model_and_variant(
        raw_model="canary-1b",
        current_model="parakeet",
        current_variant="onnx",
        variant_override=None,
    )
    assert m == "canary"
    assert v == "onnx"


@pytest.mark.unit
def test_unknown_parakeet_suffix_does_not_set_invalid_variant():
    m, v = normalize_model_and_variant(
        raw_model="parakeet-fast",
        current_model="parakeet",
        current_variant="standard",
        variant_override=None,
    )
    assert m == "parakeet"
    assert v in ALLOWED_PARAKEET_VARIANTS
    assert v != "fast"


@pytest.mark.unit
def test_qwen3_asr_normalizes_to_canonical_form():
    """Test that qwen3-asr model is properly recognized."""
    m, v = normalize_model_and_variant(
        raw_model="qwen3-asr",
        current_model="parakeet",
        current_variant="standard",
        variant_override=None,
    )
    assert m == "qwen3-asr"
    # variant should remain unchanged since qwen3-asr doesn't have variants
    assert v == "standard"


@pytest.mark.unit
def test_qwen3_asr_variants_normalize_to_canonical():
    """Test that qwen3_asr and qwen3asr variants normalize to qwen3-asr."""
    for raw in ["qwen3_asr", "qwen3asr", "qwen3-1.7b"]:
        m, v = normalize_model_and_variant(
            raw_model=raw,
            current_model="whisper",
            current_variant="mlx",
            variant_override=None,
        )
        assert m == "qwen3-asr", f"Expected qwen3-asr for raw={raw}, got {m}"
