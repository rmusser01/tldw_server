# Audio_Transcription_Parakeet_ONNX.py
#########################################
# ONNX Parakeet Model Transcription with Proper Decoding
# This module provides transcription using ONNX-optimized Parakeet models
# with full preprocessing, inference, and decoding support.
#
####################
# Function List
#
# 1. load_parakeet_onnx_model() - Load and cache ONNX model with tokenizer
# 2. preprocess_audio_for_onnx() - Prepare audio for ONNX inference
# 3. decode_onnx_output() - Decode model outputs to text
# 4. transcribe_with_parakeet_onnx() - Main transcription function
# 5. transcribe_chunked_onnx() - Chunked transcription for long audio
#
####################

import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Optional, Union

import numpy as np
import soundfile as sf
from loguru import logger

from tldw_Server_API.app.core.config import get_stt_config
from tldw_Server_API.app.core.exceptions import (
    STTExecutionPlanError,
    STTExecutionUnsupportedError,
    STTTranscriptionError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttBatchExecutionPlan,
    SttExecutionRoute,
    SttLoadedRuntime,
    SttTranscriptionOutcome,
    actual_execution_from_route,
    raise_for_planned_stt_sentinel,
    require_local_execution_route,
    validate_stt_loaded_runtime,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logger.warning("ONNX Runtime not installed. Install with: pip install onnxruntime")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

# Global cache for model and tokenizer
_onnx_model_cache: dict[str, Any] = {}
_PARAKEET_ONNX_ALLOW_PATTERNS = [
    "*.onnx",
    "**/*.onnx",
    "*.onnx.data",
    "**/*.onnx.data",
    "vocab.txt",
    "config.json",
]
_PARAKEET_ONNX_REQUIRED_SIDECARS = ("vocab.txt", "config.json")
_PARAKEET_ONNX_SILENCE_ABS_THRESHOLD = 1e-6

logger = logger


def _resolve_snapshot_download() -> Any:
    """Resolve snapshot_download with patch-friendly precedence.

    Why:
    - Some tests patch ``huggingface_hub.snapshot_download``.
    - Others patch this module's ``snapshot_download`` symbol directly.
    - Import order can otherwise leave a stale function bound here.
    """
    global snapshot_download

    local_fn = snapshot_download
    runtime_fn = None
    try:
        import importlib as _importlib
        hub_mod = _importlib.import_module("huggingface_hub")
        runtime_fn = getattr(hub_mod, "snapshot_download", None)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        runtime_fn = None

    if runtime_fn is None:
        return local_fn
    if local_fn is None:
        snapshot_download = runtime_fn
        return runtime_fn
    if local_fn is runtime_fn:
        return local_fn

    # Respect explicit direct patches against this module-level symbol.
    local_mod = str(getattr(local_fn, "__module__", ""))
    local_cls_mod = str(getattr(getattr(local_fn, "__class__", object), "__module__", ""))
    if local_mod.startswith("unittest.mock") or local_cls_mod.startswith("unittest.mock"):
        return local_fn

    # Otherwise prefer runtime symbol so patched huggingface_hub lookups are honored.
    snapshot_download = runtime_fn
    return runtime_fn


class ParakeetONNXTokenizer:
    """Simple tokenizer for Parakeet ONNX models."""

    def __init__(self, vocab_path: Union[Path, dict[str, int]]):
        """Load vocabulary from file or use provided mapping."""
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

        # If a dict is provided directly, use it
        if isinstance(vocab_path, dict):
            try:
                self.vocab = {str(k): int(v) for k, v in vocab_path.items()}
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
                return
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Invalid vocab dict provided to tokenizer: {e}; falling back to default vocab")
                self._create_default_vocab()
                return

        # Try to load vocabulary from a file path
        if vocab_path.exists():
            try:
                with open(vocab_path, encoding='utf-8') as f:
                    # Try JSON first, but tolerate simple line-based formats from tests
                    try:
                        vocab_data = json.load(f)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        f.seek(0)
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                        # Accept formats like "token idx" or pure token per line
                        parsed = {}
                        for i, line in enumerate(lines):
                            parts = line.split()
                            if len(parts) == 2 and parts[1].isdigit():
                                parsed[parts[0]] = int(parts[1])
                            else:
                                parsed[line] = i
                        vocab_data = parsed
                    if isinstance(vocab_data, dict):
                        self.vocab = vocab_data
                    elif isinstance(vocab_data, list):
                        # List format - create dict
                        self.vocab = {token: idx for idx, token in enumerate(vocab_data)}
                    # Create inverse vocabulary
                    self.inv_vocab = {v: k for k, v in self.vocab.items()}
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load tokenizer vocab from {vocab_path}: {e}; using default vocab")
                self._create_default_vocab()
        else:
            # Use default SentencePiece vocabulary
            self._create_default_vocab()

    def _create_default_vocab(self):
        """Create a default vocabulary for Parakeet."""
        # Common tokens for Parakeet/RNNT models
        special_tokens = ['<pad>', '<s>', '</s>', '<unk>', '<blank>']

        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx

        # Add space
        self.vocab['▁'] = len(self.vocab)  # SentencePiece space token

        # Add ASCII printable characters
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Add common subword units (simplified)
        common_subwords = [
            '▁the', '▁a', '▁to', '▁of', '▁and', '▁in', '▁is', '▁it',
            '▁that', '▁for', '▁was', '▁with', '▁as', '▁on', '▁be',
            '▁have', '▁but', '▁not', '▁you', '▁he', '▁at', '▁this',
            '▁from', '▁by', '▁are', '▁we', '▁an', '▁or', '▁will',
            '▁one', '▁would', '▁there', '▁their', '▁what', '▁so',
            '▁up', '▁out', '▁if', '▁about', '▁who', '▁get', '▁which',
            '▁go', '▁me', '▁when', '▁make', '▁can', '▁like', '▁time',
            'ing', 'ed', 'er', 'ly', 'al', 'es', 'ion', 'en', 'ation'
        ]

        for subword in common_subwords:
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Created default vocabulary with {len(self.vocab)} tokens")

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inv_vocab:
                token = self.inv_vocab[token_id]
                # Skip special tokens
                if (
                    token not in ['<pad>', '<s>', '</s>', '<blank>', '<blk>', '<unk>']
                    and not token.startswith("<|")
                ):
                    tokens.append(token)

        # Join tokens and clean up
        text = ''.join(tokens)
        # Replace SentencePiece space token with actual space
        text = text.replace('▁', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        return text.strip()


class ParakeetOnnxAsrRuntime:
    """Thin adapter around upstream onnx-asr for Parakeet TDT exports."""

    def __init__(self, upstream_model: Any) -> None:
        self.upstream_model = upstream_model

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe with upstream onnx-asr instead of local TDT decoding."""
        waveform = np.asarray(audio_data, dtype=np.float32)
        if waveform.size == 0:
            return "[No speech detected]"
        if float(np.max(np.abs(waveform))) <= _PARAKEET_ONNX_SILENCE_ABS_THRESHOLD:
            return "[No speech detected]"
        text = self.upstream_model.recognize(
            waveform,
            sample_rate=int(sample_rate),
            channel="mean",
        )
        if isinstance(text, list):
            text = " ".join(str(part).strip() for part in text if str(part).strip())
        result = str(text).strip()
        return result if result else "[No speech detected]"


def get_mel_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract mel-spectrogram features from audio.

    Args:
        audio: Audio samples
        sample_rate: Sample rate

    Returns:
        Mel-spectrogram features
    """
    try:
        import librosa
        use_librosa = True
    except ImportError:
        logger.debug("librosa not installed; using lightweight fallback feature extractor")
        use_librosa = False

    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize audio
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    if use_librosa:
        # Extract mel-spectrogram via librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,  # 10ms hop
            win_length=400,  # 25ms window
            n_mels=80,
            fmin=0,
            fmax=8000
        )
        log_mel = np.log(mel_spec + 1e-10).T  # (time, features)
    else:
        # Minimal fallback: frame and compute simple energy-based features
        frame = 400
        hop = 160
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        # Pad to full frames
        total = len(audio)
        if total < frame:
            pad = frame - total
            audio = np.pad(audio, (0, pad), mode='constant')
            total = len(audio)
        num_frames = 1 + max(0, (total - frame) // hop)
        feats = np.zeros((num_frames, 80), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop
            end = start + frame
            window = audio[start:end]
            if window.size < frame:
                window = np.pad(window, (0, frame - window.size), mode='constant')
            # Simple features: RMS energy + downsampled autocorrelation like proxy
            rms = np.sqrt(np.mean(window ** 2) + 1e-10)
            # Fill first channel with rms and others as scaled variants
            feats[i, 0] = rms
            if rms > 0:
                for k in range(1, 80):
                    feats[i, k] = feats[i, 0] * (1.0 - (k / 80.0))
        log_mel = feats

    # Normalize
    mean = np.mean(log_mel, axis=0, keepdims=True)
    std = np.std(log_mel, axis=0, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1e-10)

    return log_mel.astype(np.float32)


# Backwards-compatible private name used in tests for patching
def _preprocess_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compatibility wrapper used in tests; delegates to get_mel_features."""
    features = get_mel_features(audio, sample_rate)
    # Add batch dimension like tests often expect
    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)
    return features


def _onnx_input_name(input_meta: Any) -> str:
    """Return a stable string input name from ONNX metadata or test doubles."""
    name = getattr(input_meta, "name", "")
    if isinstance(name, str):
        return name
    mock_name = getattr(input_meta, "_mock_name", "")
    return str(mock_name or name or "")


def _onnx_input_rank(input_meta: Any) -> int | None:
    """Return the declared ONNX input rank when metadata exposes one."""
    shape = getattr(input_meta, "shape", None)
    if isinstance(shape, (list, tuple)):
        return len(shape)
    return None


def _is_length_input_name(name: str) -> bool:
    lname = str(name or "").lower()
    if not lname:
        return False
    return (
        "length" in lname
        or "lens" in lname
        or lname.endswith("_len")
        or lname.endswith("_lens")
        or lname in {"len", "lens", "lengths"}
    )


def _is_raw_waveform_input(input_meta: Any) -> bool:
    name = _onnx_input_name(input_meta)
    lname = name.lower()
    rank = _onnx_input_rank(input_meta)
    if _is_length_input_name(name):
        return False
    if (
        lname in {"targets", "target"}
        or lname.startswith("input_state")
        or lname.startswith("output_state")
        or "encoder_output" in lname
    ):
        return False
    if "feature" in lname or "mel" in lname or "encoder" in lname or "processed_signal" in lname:
        return False
    if "waveform" in lname or "audio_signal" in lname or "raw_audio" in lname or lname in {"audio", "samples"}:
        return rank is None or rank <= 2
    return rank == 2


def _is_feature_input(input_meta: Any) -> bool:
    name = _onnx_input_name(input_meta)
    lname = name.lower()
    rank = _onnx_input_rank(input_meta)
    if _is_length_input_name(name):
        return False
    if lname in {"targets", "target"} or lname.startswith("input_state") or lname.startswith("output_state"):
        return False
    return (
        "feature" in lname
        or "mel" in lname
        or "encoder" in lname
        or "processed_signal" in lname
        or "speech" in lname
        or rank is not None and rank >= 3
    )


def _prepare_waveform_input(audio_data: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio_data, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)
    return np.expand_dims(waveform, axis=0)


def _prepare_onnx_inputs(
    session: Any,
    features: np.ndarray,
    waveform: Optional[np.ndarray] = None,
    signal_length: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Build a best-effort ONNX input map from runtime input names.

    Different exported Parakeet variants use different names for the feature
    or waveform tensor (e.g. `input_features`, `encoder_outputs`, `waveforms`).
    This helper keeps mapping resilient across those variants and test doubles.
    """
    inputs_meta = list(session.get_inputs() or [])
    input_names = [_onnx_input_name(inp) for inp in inputs_meta]
    prepared: dict[str, np.ndarray] = {}
    explicit_signal_length: int | None = None
    if signal_length is not None:
        try:
            explicit_signal_length = max(0, int(signal_length))
        except (OverflowError, TypeError, ValueError):
            explicit_signal_length = None

    signal_name: str | None = None
    signal_tensor = features
    signal_length = int(features.shape[1]) if features.ndim >= 2 else int(features.size)

    if waveform is not None:
        for input_meta in inputs_meta:
            if _is_raw_waveform_input(input_meta):
                signal_name = _onnx_input_name(input_meta)
                signal_tensor = waveform.astype(np.float32, copy=False)
                signal_length = (
                    explicit_signal_length
                    if explicit_signal_length is not None
                    else int(signal_tensor.shape[-1])
                )
                break

    if signal_name is None:
        for input_meta in inputs_meta:
            if _is_feature_input(input_meta):
                signal_name = _onnx_input_name(input_meta)
                signal_tensor = features
                signal_length = int(features.shape[1]) if features.ndim >= 2 else int(features.size)
                break

    if signal_name is None:
        for input_meta in inputs_meta:
            name = _onnx_input_name(input_meta)
            if _is_length_input_name(name):
                continue
            signal_name = name
            if waveform is not None and _onnx_input_rank(input_meta) == 2:
                signal_tensor = waveform.astype(np.float32, copy=False)
                signal_length = (
                    explicit_signal_length
                    if explicit_signal_length is not None
                    else int(signal_tensor.shape[-1])
                )
            break

    if signal_name is not None:
        prepared[signal_name] = signal_tensor

    for name in input_names:
        if name == signal_name:
            continue
        lname = str(name).lower()
        if _is_length_input_name(name) or "seq_len" in lname:
            prepared[name] = np.array([signal_length], dtype=np.int64)
        elif "batch" in lname:
            prepared[name] = np.array([signal_tensor.shape[0]], dtype=np.int64)

    return prepared


def _existing_path(path: Path) -> Path | None:
    """Return a path only when it exists."""
    return path if path.exists() else None


def _select_graph_path(model_dir: Path, candidates: list[str]) -> Path | None:
    """Pick the first available ONNX graph from an ordered candidate list."""
    for candidate in candidates:
        path = _existing_path(model_dir / candidate)
        if path is not None:
            return path
    return None


def _resolve_parakeet_tdt_bundle_paths(
    model_dir: Path,
    quantization: str | None,
) -> dict[str, Path] | None:
    """Resolve the multi-graph Parakeet TDT ONNX export layout when present."""
    preprocessor_path = _select_graph_path(model_dir, ["nemo128.onnx", "nemo80.onnx"])
    suffix = ".int8" if quantization == "int8" else ""
    encoder_path = _existing_path(model_dir / f"encoder-model{suffix}.onnx")
    decoder_joint_path = _existing_path(model_dir / f"decoder_joint-model{suffix}.onnx")
    vocab_path = _existing_path(model_dir / "vocab.txt")
    config_path = _existing_path(model_dir / "config.json")

    if preprocessor_path and encoder_path and decoder_joint_path and vocab_path and config_path:
        return {
            "preprocessor": preprocessor_path,
            "encoder": encoder_path,
            "decoder_joint": decoder_joint_path,
            "vocab": vocab_path,
            "config": config_path,
        }
    return None


def _resolve_parakeet_tdt_quantization(model_dir: Path) -> str | None:
    """Return the preferred onnx-asr quantization suffix for available graph files."""
    if (
        (model_dir / "encoder-model.int8.onnx").exists()
        and (model_dir / "decoder_joint-model.int8.onnx").exists()
    ):
        return "int8"
    return None


def _resolve_onnx_asr_load_model() -> Callable[..., Any] | None:
    """Resolve upstream onnx-asr load_model when installed."""
    try:
        import importlib

        onnx_asr = importlib.import_module("onnx_asr")
    except (ImportError, ModuleNotFoundError, RuntimeError):
        return None
    load_model = getattr(onnx_asr, "load_model", None)
    return load_model if callable(load_model) else None


def _has_parakeet_tdt_graphs(model_dir: Path) -> bool:
    """Return True when a directory has the Parakeet TDT multi-graph ONNX files."""
    return bool(
        _select_graph_path(model_dir, ["nemo128.onnx", "nemo80.onnx"])
        and _select_graph_path(model_dir, ["encoder-model.int8.onnx", "encoder-model.onnx"])
        and _select_graph_path(model_dir, ["decoder_joint-model.int8.onnx", "decoder_joint-model.onnx"])
    )


def _remote_parakeet_cache_needs_refresh(model_dir: Path) -> bool:
    """Return whether a remote cache lacks required metadata or TDT graphs."""
    if any(
        not (model_dir / sidecar).exists()
        for sidecar in _PARAKEET_ONNX_REQUIRED_SIDECARS
    ):
        return True

    try:
        with (model_dir / "config.json").open("rt", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return True

    if not isinstance(config, dict) or config.get("model_type") != "nemo-conformer-tdt":
        return False

    quantization = _resolve_parakeet_tdt_quantization(model_dir)
    return _resolve_parakeet_tdt_bundle_paths(model_dir, quantization) is None


def _config_declares_parakeet_tdt(model_dir: Path) -> bool:
    """Return whether config.json identifies an onnx-asr Parakeet TDT bundle."""
    try:
        with (model_dir / "config.json").open("rt", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "nemo-conformer-tdt"


def _is_explicit_local_model_path(model_path: str) -> bool:
    """Recognize unambiguous local paths without requiring them to exist."""
    return bool(
        Path(model_path).expanduser().is_absolute()
        or PureWindowsPath(model_path).is_absolute()
        or model_path.startswith(("./", "../", ".\\", "..\\", "~/", "~\\"))
    )


def _load_parakeet_features_size(config_path: Path, model_dir: Path) -> int | None:
    """Read the positive integer feature count required by onnx-asr."""
    try:
        with config_path.open("rt", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, TypeError, ValueError):
        config = None

    if config is None:
        logger.error("Invalid Parakeet ONNX config in {}: unreadable or malformed JSON", model_dir)
        return None

    if not isinstance(config, dict):
        logger.error("Invalid Parakeet ONNX config in {}: expected a JSON object", model_dir)
        return None

    features_size = config.get("features_size")
    if isinstance(features_size, bool) or not isinstance(features_size, int) or features_size <= 0:
        logger.error(
            "Invalid Parakeet ONNX config in {}: features_size must be a positive integer",
            model_dir,
        )
        return None
    return features_size


def _encoder_features_match_config(
    runtime: Any,
    encoder_path: Path,
    session_options: Any,
    configured_features: int,
    model_dir: Path,
) -> bool:
    """Validate a static encoder feature axis with a CPU-only metadata session."""
    try:
        encoder_session = runtime.InferenceSession(
            str(encoder_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        audio_input = next(
            (input_meta for input_meta in encoder_session.get_inputs() if _onnx_input_name(input_meta) == "audio_signal"),
            None,
        )
    except Exception as exc:
        logger.exception("Failed to inspect Parakeet ONNX encoder in {}: {}", model_dir, exc)
        return False

    if audio_input is None:
        logger.error("Invalid Parakeet ONNX encoder in {}: audio_signal input is missing", model_dir)
        return False

    shape = getattr(audio_input, "shape", None)
    declared_features = shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else None
    if (
        isinstance(declared_features, int)
        and not isinstance(declared_features, bool)
        and declared_features > 0
        and declared_features != configured_features
    ):
        logger.error(
            "Parakeet ONNX feature mismatch in {}: config features_size={} but encoder expects {}",
            model_dir,
            configured_features,
            declared_features,
        )
        return False
    return True


def _middle_trimmed_chunk_text(text: str, chunk_duration: float, overlap_duration: float) -> str:
    """Trim the start of a chunk transcript using the existing middle-merge heuristic."""
    if chunk_duration <= 0 or overlap_duration <= 0:
        return text
    overlap_chars = int(len(text) * overlap_duration / chunk_duration)
    if overlap_chars <= 0:
        return text
    return text[overlap_chars // 2:]


def _effective_onnx_device(session: object) -> str | None:
    devices = {
        "CUDAExecutionProvider": "cuda",
        "CoreMLExecutionProvider": "mps",
        "CPUExecutionProvider": "cpu",
    }
    pending = [session]
    visited: set[int] = set()
    effective: list[str] = []
    nested_attributes = (
        "upstream_model",
        "model",
        "asr",
        "_encoder",
        "_decoder_joint",
        "_decoder",
        "_model",
    )
    while pending:
        component = pending.pop()
        component_id = id(component)
        if component_id in visited:
            continue
        visited.add(component_id)
        get_providers = getattr(component, "get_providers", None)
        if callable(get_providers):
            for provider in map(str, get_providers()):
                if provider in devices:
                    effective.append(devices[provider])
                    break
        for attribute in nested_attributes:
            nested = getattr(component, attribute, None)
            if nested is not None:
                pending.append(nested)
    if not effective or len(set(effective)) != 1:
        return None
    return effective[0]


def _onnx_loaded_runtime(
    route: SttExecutionRoute,
    session: object,
    tokenizer: object,
) -> SttLoadedRuntime:
    device = _effective_onnx_device(session)
    if device is None:
        raise STTExecutionPlanError(
            "Parakeet ONNX did not expose an effective execution provider"
        )
    actual = actual_execution_from_route(
        route,
        device=device,
    )
    return SttLoadedRuntime(
        components=(session, tokenizer),
        actual_execution=actual,
    )


def validate_local_onnx_artifact(model_path: str | Path) -> Path:
    """Require a complete local ONNX graph and tokenizer artifact."""
    path = Path(model_path)
    if not path.is_dir() or path.is_symlink():
        raise STTExecutionUnsupportedError(
            "Planned Parakeet ONNX requires a complete local artifact"
        )
    resolved = path.resolve()
    quantization = _resolve_parakeet_tdt_quantization(resolved)
    if _resolve_parakeet_tdt_bundle_paths(resolved, quantization) is not None:
        return resolved
    has_graph = any(resolved.glob("*.onnx"))
    has_tokenizer = any(
        (resolved / name).is_file()
        for name in ("vocab.json", "tokenizer.json")
    )
    if not has_graph or not has_tokenizer:
        raise STTExecutionUnsupportedError(
            "Planned Parakeet ONNX requires a complete local artifact"
        )
    return resolved


def load_parakeet_onnx_model(
    model_path: Optional[str] = None,
    device: str = "cpu",
    *,
    allow_download: bool = True,
    execution_route: SttExecutionRoute | None = None,
) -> tuple[Any, Any] | SttLoadedRuntime:
    """
    Load Parakeet ONNX model and tokenizer.

    Args:
        model_path: Path to ONNX model directory or HuggingFace repo
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Tuple of (ONNX session, tokenizer) or (None, None) if loading fails
    """
    global _onnx_model_cache

    global ort
    planned = execution_route is not None
    if ort is None or not hasattr(ort, 'InferenceSession'):
        # Attempt a late import to support tests that patch onnxruntime
        try:
            import onnxruntime as _ort
            ort = _ort
        except ImportError:
            try:
                import sys as _sys
                ort = _sys.modules.get('onnxruntime', None)
            except (AttributeError, RuntimeError, TypeError):
                ort = None
    if ort is None or not hasattr(ort, 'InferenceSession'):
        logger.error("ONNX Runtime not available")
        if planned:
            raise STTExecutionUnsupportedError(
                "Planned Parakeet ONNX execution requires ONNX Runtime"
            )
        return None, None

    if planned:
        require_local_execution_route(
            execution_route,
            provider="parakeet",
            backend="onnxruntime",
        )
        if allow_download:
            raise STTExecutionPlanError(
                "Planned Parakeet ONNX execution must prohibit downloads"
            )
    if not allow_download:
        if model_path is None:
            raise STTExecutionUnsupportedError(
                "No-download Parakeet ONNX execution requires an explicit local model directory"
            )
        planned_dir = validate_local_onnx_artifact(model_path)
        model_path = str(planned_dir)
        stt_cfg: dict[str, Any] = {}
    else:
        try:
            stt_cfg = get_stt_config() or {}
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError):
            stt_cfg = {}

    # Default model
    if model_path is None:
        configured_model_id = str(stt_cfg.get("parakeet_onnx_model_id", "")).strip()
        model_path = configured_model_id or "istupakov/parakeet-tdt-0.6b-v3-onnx"

    raw_configured_revision = stt_cfg.get("parakeet_onnx_revision", "")
    configured_revision = (
        str(raw_configured_revision).strip()
        if raw_configured_revision is not None
        else None
    )
    revision = (
        None
        if planned
        else configured_revision or os.getenv("PARAKEET_ONNX_REVISION")
    )

    cache_key = f"{model_path}_{revision or ''}_{device}"
    if cache_key in _onnx_model_cache:
        logger.debug(
            "Using cached planned local ONNX model"
            if planned
            else f"Using cached ONNX model: {model_path}"
        )
        cached_session, cached_tokenizer = _onnx_model_cache[cache_key]
        if execution_route is None:
            return cached_session, cached_tokenizer
        return _onnx_loaded_runtime(
            execution_route,
            cached_session,
            cached_tokenizer,
        )

    try:
        # Check if it's a local path or HuggingFace repo
        model_dir = Path(model_path).expanduser()

        download_fn = _resolve_snapshot_download() if allow_download else None

        is_existing_local_dir = model_dir.exists() and model_dir.is_dir()
        is_explicit_local_path = _is_explicit_local_model_path(model_path)
        is_local_model = is_existing_local_dir or is_explicit_local_path
        if is_explicit_local_path and not is_existing_local_dir:
            logger.error("Local Parakeet ONNX model directory does not exist: {}", model_dir)
            return None, None

        remote_model_id = None if is_local_model or not download_fn else model_path
        repair_attempted = False
        if not is_local_model and download_fn:
            # Download from HuggingFace
            logger.info(f"Downloading ONNX model from HuggingFace: {model_path}")
            cache_dir = Path.home() / '.cache' / 'parakeet_onnx'
            revision_token = (
                str(revision).replace("/", "_").replace(":", "_")
                if revision
                else "default"
            )
            model_dir = cache_dir / f"{model_path.replace('/', '_')}_{revision_token}"

            # Existing caches may contain only part of a TDT graph bundle from an
            # interrupted or older download, so refresh any incomplete bundle.
            if not model_dir.exists() or _remote_parakeet_cache_needs_refresh(model_dir):
                download_fn(
                    repo_id=model_path,
                    local_dir=str(model_dir),
                    revision=revision,
                    allow_patterns=_PARAKEET_ONNX_ALLOW_PATTERNS,
                )  # nosec B615
                repair_attempted = True
                if _remote_parakeet_cache_needs_refresh(model_dir):
                    logger.error(
                        "Remote Parakeet ONNX cache remains incomplete after refresh: {}",
                        model_dir,
                    )
                    return None, None

        # Set up providers
        providers = [
            "CUDAExecutionProvider"
            if device == "cuda"
            else "CPUExecutionProvider"
        ] if planned else (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        # Create ONNX sessions. Use a fresh import so patched attributes are respected.
        try:
            import importlib as _importlib
            _runtime = _importlib.import_module('onnxruntime')
        except (ImportError, ModuleNotFoundError, RuntimeError):
            _runtime = ort
        if planned:
            get_available_providers = getattr(
                _runtime,
                "get_available_providers",
                None,
            )
            available = (
                set(map(str, get_available_providers()))
                if callable(get_available_providers)
                else set()
            )
            if callable(get_available_providers) and providers[0] not in available:
                raise STTExecutionUnsupportedError(
                    "Planned Parakeet ONNX device provider is unavailable"
                )

        session_options = _runtime.SessionOptions()
        session_options.graph_optimization_level = _runtime.GraphOptimizationLevel.ORT_ENABLE_ALL

        for _attempt in range(2):
            quantization = _resolve_parakeet_tdt_quantization(model_dir)
            bundle_paths = _resolve_parakeet_tdt_bundle_paths(model_dir, quantization)
            if bundle_paths is None:
                break

            features_size = _load_parakeet_features_size(bundle_paths["config"], model_dir)
            encoder_matches = features_size is not None and _encoder_features_match_config(
                _runtime,
                bundle_paths["encoder"],
                session_options,
                features_size,
                model_dir,
            )
            if not encoder_matches:
                if remote_model_id is not None and not repair_attempted:
                    logger.warning("Refreshing remote Parakeet ONNX cache after validation failure")
                    download_fn(
                        repo_id=remote_model_id,
                        local_dir=str(model_dir),
                        revision=revision,
                        allow_patterns=_PARAKEET_ONNX_ALLOW_PATTERNS,
                    )  # nosec B615
                    repair_attempted = True
                    continue
                return None, None

            load_onnx_asr_model = _resolve_onnx_asr_load_model()
            if load_onnx_asr_model is None:
                if planned:
                    raise STTExecutionUnsupportedError(
                        "Planned Parakeet ONNX execution requires onnx-asr"
                    )
                logger.error(
                    "Parakeet ONNX TDT export found in {} but onnx-asr is not installed. "
                    "Install with: pip install 'onnx-asr[hub]'",
                    model_dir,
                )
                return None, None

            if planned:
                logger.info("Loading planned local Parakeet ONNX graph bundle")
            else:
                logger.info(
                    "Loading Parakeet TDT ONNX graph bundle through upstream onnx-asr from: {}",
                    model_dir,
                )
            try:
                upstream_model = load_onnx_asr_model(
                    "nemo-conformer-tdt",
                    path=model_dir,
                    quantization=quantization,
                    sess_options=session_options,
                    providers=providers,
                    preprocessor_config={"use_numpy_preprocessors": True},
                )
            except Exception:
                if remote_model_id is not None and not repair_attempted:
                    logger.warning("Refreshing remote Parakeet ONNX cache after model load failure")
                    download_fn(
                        repo_id=remote_model_id,
                        local_dir=str(model_dir),
                        revision=revision,
                        allow_patterns=_PARAKEET_ONNX_ALLOW_PATTERNS,
                    )  # nosec B615
                    repair_attempted = True
                    continue
                raise
            tokenizer = ParakeetONNXTokenizer(bundle_paths["vocab"])
            session = ParakeetOnnxAsrRuntime(upstream_model)
            _onnx_model_cache[cache_key] = (session, tokenizer)
            logger.info("Successfully loaded Parakeet TDT ONNX graph bundle through onnx-asr")
            if execution_route is None:
                return session, tokenizer
            return _onnx_loaded_runtime(execution_route, session, tokenizer)
        if _config_declares_parakeet_tdt(model_dir):
            if planned:
                raise STTExecutionUnsupportedError(
                    "Planned Parakeet ONNX graph bundle is incomplete"
                )
            logger.error("Parakeet ONNX TDT graph bundle in {} remains incomplete", model_dir)
            return None, None
        if _has_parakeet_tdt_graphs(model_dir):
            missing_sidecars = [
                sidecar
                for sidecar in _PARAKEET_ONNX_REQUIRED_SIDECARS
                if not (model_dir / sidecar).exists()
            ]
            if planned:
                detail = (
                    "missing required sidecars"
                    if missing_sidecars
                    else "lacks a compatible encoder/decoder graph pair"
                )
                raise STTExecutionUnsupportedError(
                    f"Planned Parakeet ONNX graph bundle {detail}"
                )
            if missing_sidecars:
                logger.error(
                    "Parakeet ONNX TDT graph bundle in {} is missing required sidecar(s): {}",
                    model_dir,
                    ", ".join(missing_sidecars),
                )
            else:
                logger.error(
                    "Parakeet ONNX TDT graph bundle in {} lacks a compatible encoder/decoder graph pair",
                    model_dir,
                )
            return None, None

        # Find ONNX files
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            # Legacy mode preserves its historical mocked-session fallback.
            logger.warning(
                f"No ONNX files found in {model_dir}; proceeding with placeholder path for session initialization"
            )
            onnx_path = model_dir / "model.onnx"
        else:
            # Use the first ONNX file (usually encoder.onnx or model.onnx)
            onnx_path = onnx_files[0]
        logger.info(
            "Loading planned local ONNX model"
            if planned
            else f"Loading ONNX model from: {onnx_path}"
        )

        session = _runtime.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers
        )

        # Load tokenizer
        vocab_path = model_dir / "vocab.json"
        if not vocab_path.exists():
            vocab_path = model_dir / "tokenizer.json"

        tokenizer = ParakeetONNXTokenizer(vocab_path)

        # Cache the model
        _onnx_model_cache[cache_key] = (session, tokenizer)

        logger.info(f"Successfully loaded ONNX model with {len(session.get_inputs())} inputs")
        if execution_route is None:
            return session, tokenizer
        return _onnx_loaded_runtime(execution_route, session, tokenizer)

    except (STTExecutionPlanError, STTExecutionUnsupportedError):
        raise
    except Exception as e:
        if planned:
            raise STTExecutionUnsupportedError(
                "Planned local Parakeet ONNX artifact could not be loaded"
            ) from None
        logger.exception(f"Failed to load ONNX model: {e}")
        return None, None


def transcribe_with_parakeet_onnx(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    model_path: Optional[str] = None,
    device: str = 'cpu',
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 0.5,
    merge_algo: str = 'middle',
    chunk_callback: Optional[Callable[[int, int], None]] = None,
    *,
    execution_plan: SttBatchExecutionPlan | None = None,
    execution_route: SttExecutionRoute | None = None,
) -> str | SttTranscriptionOutcome:
    """
    Transcribe audio using Parakeet ONNX model.

    Args:
        audio_data: Audio data as numpy array or file path
        sample_rate: Sample rate of audio
        model_path: Path to ONNX model or HuggingFace repo
        device: Device to run on ('cpu' or 'cuda')
        chunk_duration: Duration for chunking in seconds (None = no chunking)
        overlap_duration: Overlap between chunks
        merge_algo: Algorithm for merging chunks ('middle', 'overlap', 'simple')
        chunk_callback: Progress callback for chunks

    Returns:
        Transcribed text
    """
    loaded_runtime: SttLoadedRuntime | None = None

    def _finish(text: str) -> str | SttTranscriptionOutcome:
        if loaded_runtime is None:
            return text
        raise_for_planned_stt_sentinel(text)
        return SttTranscriptionOutcome(
            artifact={
                "text": text,
                "segments": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 0.0,
                        "Text": text,
                    }
                ],
                "language": execution_plan.language if execution_plan else None,
            },
            actual_execution=loaded_runtime.actual_execution,
        )

    # Load model
    try:
        loaded = load_parakeet_onnx_model(
            model_path,
            device,
            allow_download=execution_plan is None,
            execution_route=(
                execution_route or execution_plan.descriptor.primary_route
                if execution_plan is not None
                else None
            ),
        )
        if execution_plan is not None:
            route = execution_route or execution_plan.descriptor.primary_route
            loaded_runtime = validate_stt_loaded_runtime(loaded, route)
            session, tokenizer = loaded.components
        else:
            session, tokenizer = loaded
    except (STTExecutionPlanError, STTExecutionUnsupportedError):
        raise
    except Exception as e:
        if execution_plan is not None:
            raise STTTranscriptionError(
                "Parakeet ONNX model load failed during planned execution"
            ) from None
        logger.exception(f"Failed to load ONNX model: {e}")
        return _finish("[Error: Failed to load ONNX model]")
    if session is None or tokenizer is None:
        return _finish("[Error: Failed to load ONNX model]")

    # Load audio if it's a file path
    if isinstance(audio_data, (str, Path)):
        try:
            audio_data, file_sr = sf.read(str(audio_data))
            if file_sr != sample_rate:
                # Resample if needed
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=file_sr,
                    target_sr=sample_rate
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            if execution_plan is not None:
                raise STTTranscriptionError(
                    "Parakeet ONNX audio decode failed during planned execution"
                ) from None
            logger.exception(f"Failed to load audio file: {e}")
            return _finish("[Error: Failed to load audio]")

    # Ensure numpy array
    if not isinstance(audio_data, np.ndarray):
        return _finish("[Error: Invalid audio data type]")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Check if we need chunking
    audio_duration = len(audio_data) / sample_rate

    bundle_transcribe = getattr(session, "transcribe", None)
    session_run = getattr(session, "run", None)
    if isinstance(session, ParakeetOnnxAsrRuntime) or (
        callable(bundle_transcribe) and not callable(session_run)
    ):
        try:
            if chunk_duration and audio_duration > chunk_duration:
                chunk_samples = int(chunk_duration * sample_rate)
                overlap_samples = int(overlap_duration * sample_rate)
                stride_samples = max(1, chunk_samples - overlap_samples)
                transcripts: list[str] = []
                total_samples = len(audio_data)
                num_chunks = max(
                    1,
                    int(np.ceil(max(1, total_samples - overlap_samples) / stride_samples)),
                )
                for i in range(num_chunks):
                    start = i * stride_samples
                    end = min(start + chunk_samples, total_samples)
                    chunk = audio_data[start:end]
                    text = bundle_transcribe(chunk, sample_rate)
                    if text and not text.startswith("["):
                        if merge_algo == "middle" and i > 0 and overlap_samples > 0:
                            text = _middle_trimmed_chunk_text(
                                text,
                                chunk_duration,
                                overlap_duration,
                            )
                        transcripts.append(text)
                    if chunk_callback:
                        chunk_callback(i + 1, num_chunks)
                result = (
                    merge_with_overlap_removal(transcripts)
                    if merge_algo == "overlap"
                    else " ".join(transcripts)
                )
                return _finish(
                    result.strip() if result.strip() else "[No speech detected]"
                )
            return _finish(bundle_transcribe(audio_data, sample_rate))
        except (STTExecutionPlanError, STTExecutionUnsupportedError, STTTranscriptionError):
            raise
        except Exception as e:
            if execution_plan is not None:
                raise STTTranscriptionError(
                    "Parakeet ONNX transcription failed during planned execution"
                ) from None
            logger.exception(f"Parakeet TDT ONNX graph bundle transcription error: {e}")
            return _finish("[Error: Parakeet ONNX transcription failed]")

    if chunk_duration and audio_duration > chunk_duration:
        # Use chunked transcription
        return _finish(transcribe_chunked_onnx(
            audio_data,
            sample_rate,
            session,
            tokenizer,
            chunk_duration,
            overlap_duration,
            merge_algo,
            chunk_callback
        ))

    # Single transcription
    try:
        # Extract features
        features = get_mel_features(audio_data, sample_rate)

        if features.size == 0:
            return _finish("[Error: Feature extraction failed]")

        # Prepare input for ONNX
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        waveform = _prepare_waveform_input(audio_data)

        output_names = [out.name for out in session.get_outputs()]

        # Prepare inputs
        inputs = _prepare_onnx_inputs(session, features, waveform=waveform)

        # Run inference
        outputs = session.run(output_names, inputs)

        # Decode outputs
        if outputs and len(outputs) > 0:
            # Get the main output (usually logits or token IDs)
            output = outputs[0]

            # Handle different output formats
            if output.ndim == 3:
                # (batch, time, vocab) - take argmax
                token_ids = np.argmax(output[0], axis=-1)
            elif output.ndim == 2:
                # (batch, time) - already token IDs
                token_ids = output[0]
            else:
                token_ids = output.flatten()

            # Remove padding and blank tokens
            token_ids = token_ids[token_ids > 0]

            # Decode to text
            text = tokenizer.decode(token_ids.tolist())
            return _finish(text if text else "[No speech detected]")

        return _finish("[Error: No output from model]")

    except (STTExecutionPlanError, STTExecutionUnsupportedError, STTTranscriptionError):
        raise
    except Exception as e:
        if execution_plan is not None:
            raise STTTranscriptionError(
                "Parakeet ONNX transcription failed during planned execution"
            ) from None
        logger.exception(f"Transcription error: {e}")
        return _finish("[Error: Parakeet ONNX transcription failed]")


def transcribe_chunked_onnx(
    audio_data: np.ndarray,
    sample_rate: int,
    session: ort.InferenceSession,
    tokenizer: ParakeetONNXTokenizer,
    chunk_duration: float,
    overlap_duration: float,
    merge_algo: str,
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe long audio using chunking with ONNX model.

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        session: ONNX inference session
        tokenizer: Tokenizer for decoding
        chunk_duration: Chunk duration in seconds
        overlap_duration: Overlap between chunks
        merge_algo: Merge algorithm ('middle', 'overlap', 'simple')
        chunk_callback: Progress callback

    Returns:
        Merged transcription text
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride_samples = chunk_samples - overlap_samples

    total_samples = len(audio_data)
    num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / stride_samples)))

    transcripts = []

    # Get input/output names
    output_names = [out.name for out in session.get_outputs()]

    for i in range(num_chunks):
        start = i * stride_samples
        end = min(start + chunk_samples, total_samples)

        # Extract chunk
        raw_chunk = audio_data[start:end]
        chunk_length = len(raw_chunk)
        chunk = raw_chunk

        # Pad if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        try:
            # Extract features
            features = get_mel_features(chunk, sample_rate)

            if features.size == 0:
                continue

            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            waveform = _prepare_waveform_input(chunk)

            # Prepare inputs
            inputs = _prepare_onnx_inputs(
                session,
                features,
                waveform=waveform,
                signal_length=chunk_length,
            )

            # Run inference
            outputs = session.run(output_names, inputs)

            if outputs and len(outputs) > 0:
                output = outputs[0]

                # Get token IDs
                if output.ndim == 3:
                    token_ids = np.argmax(output[0], axis=-1)
                elif output.ndim == 2:
                    token_ids = output[0]
                else:
                    token_ids = output.flatten()

                # Remove padding
                token_ids = token_ids[token_ids > 0]

                # Decode
                text = tokenizer.decode(token_ids.tolist())

                if text:
                    if merge_algo == 'middle' and i > 0 and overlap_samples > 0:
                        text = _middle_trimmed_chunk_text(
                            text,
                            chunk_duration,
                            overlap_duration,
                        )

                    transcripts.append(text)

        except Exception as e:
            logger.exception(f"Error processing chunk {i+1}/{num_chunks}: {e}")

        # Progress callback
        if chunk_callback:
            chunk_callback(i + 1, num_chunks)

    # Merge transcripts
    if merge_algo == 'simple':
        # Simple concatenation
        result = ' '.join(transcripts)
    elif merge_algo == 'overlap':
        # Remove duplicate words at boundaries
        result = merge_with_overlap_removal(transcripts)
    else:  # 'middle'
        # Already handled trimming above
        result = ' '.join(transcripts)

    return result.strip() if result else "[No speech detected]"


def merge_with_overlap_removal(transcripts: list[str]) -> str:
    """
    Merge transcripts by removing duplicate words at boundaries.

    Args:
        transcripts: List of transcript segments

    Returns:
        Merged transcript
    """
    if not transcripts:
        return ""

    if len(transcripts) == 1:
        return transcripts[0]

    result = transcripts[0]

    for i in range(1, len(transcripts)):
        current = transcripts[i]
        if not current:
            continue

        # Find overlapping words
        prev_words = result.split()
        curr_words = current.split()

        if not prev_words or not curr_words:
            result = result + " " + current
            continue

        # Look for overlap (simplified - check last few words)
        overlap_found = False
        for overlap_size in range(min(5, len(prev_words), len(curr_words)), 0, -1):
            if prev_words[-overlap_size:] == curr_words[:overlap_size]:
                # Found overlap, merge without duplicates
                result = ' '.join(prev_words + curr_words[overlap_size:])
                overlap_found = True
                break

        if not overlap_found:
            result = result + " " + current

    return result


def unload_onnx_models():
    """Unload all cached ONNX models to free memory."""
    global _onnx_model_cache
    _onnx_model_cache.clear()
    logger.info("Unloaded all ONNX models from cache")


#######################################################################################################################
# End of Audio_Transcription_Parakeet_ONNX.py
#######################################################################################################################
