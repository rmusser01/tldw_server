# Audio_Transcription_Parakeet_MLX.py
#########################################
# Parakeet MLX Implementation for Apple Silicon
# Based on: https://github.com/senstella/parakeet-mlx
#
# This module provides optimized Parakeet transcription for Apple Silicon Macs
# using the MLX framework for efficient on-device inference.
#
####################
# Function List
#
# 1. load_parakeet_mlx_model() - Load the MLX Parakeet model
# 2. transcribe_with_parakeet_mlx() - Transcribe using MLX model
# 3. install_parakeet_mlx() - Install the parakeet-mlx package if needed
#
####################

import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from loguru import logger

# Check if we're on macOS
IS_MACOS = sys.platform == 'darwin'

# Global model cache
_mlx_model_cache: Optional[Any] = None
_DEFAULT_MLX_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely coerce a value to ``float`` and return ``default`` on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely coerce a value to ``int`` and return ``default`` on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _supports_kwarg(callable_obj: Any, kwarg_name: str) -> bool:
    """Return ``True`` when a callable accepts a keyword argument."""
    try:
        return kwarg_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _token_to_dict(token: Any) -> dict[str, Any]:
    """Normalize a Parakeet token object to the server token schema."""
    start = _safe_float(getattr(token, "start", None), 0.0) or 0.0
    duration = _safe_float(getattr(token, "duration", None), None)
    end_from_attr = _safe_float(getattr(token, "end", None), None)
    end = end_from_attr if end_from_attr is not None else (start + (duration or 0.0))
    if duration is None:
        duration = max(end - start, 0.0)
    return {
        "id": getattr(token, "id", None),
        "text": str(getattr(token, "text", "") or ""),
        "start": float(start),
        "end": float(end),
        "duration": float(duration),
        "confidence": float(_safe_float(getattr(token, "confidence", None), 1.0) or 1.0),
    }


def _sentence_to_dict(sentence: Any) -> dict[str, Any]:
    """Normalize a Parakeet sentence object to a serializable sentence dictionary."""
    tokens_raw = getattr(sentence, "tokens", None)
    token_dicts: list[dict[str, Any]] = []
    if isinstance(tokens_raw, list):
        token_dicts = [_token_to_dict(token) for token in tokens_raw]

    start = _safe_float(getattr(sentence, "start", None), None)
    end = _safe_float(getattr(sentence, "end", None), None)
    duration = _safe_float(getattr(sentence, "duration", None), None)
    if start is None and token_dicts:
        start = token_dicts[0]["start"]
    if end is None and token_dicts:
        end = token_dicts[-1]["end"]
    if start is None:
        start = 0.0
    if end is None:
        end = start
    if duration is None:
        duration = max(end - start, 0.0)

    return {
        "text": str(getattr(sentence, "text", "") or ""),
        "start": float(start),
        "end": float(end),
        "duration": float(duration),
        "confidence": float(_safe_float(getattr(sentence, "confidence", None), 1.0) or 1.0),
        "tokens": token_dicts,
    }


def _as_structured_artifact(result: Any) -> dict[str, Any]:
    """Convert MLX transcription output into a normalized structured artifact."""
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
        raw_sentences = result.get("sentences") if isinstance(result.get("sentences"), list) else []
        raw_tokens = result.get("tokens") if isinstance(result.get("tokens"), list) else []
        sentences = [dict(sentence) for sentence in raw_sentences if isinstance(sentence, dict)]
        tokens = [dict(token) for token in raw_tokens if isinstance(token, dict)]
        if not tokens and sentences:
            for sentence in sentences:
                words = sentence.get("tokens")
                if isinstance(words, list):
                    for token in words:
                        if isinstance(token, dict):
                            tokens.append(dict(token))
        return {
            "text": text,
            "sentences": sentences,
            "tokens": tokens,
        }

    text = str(getattr(result, "text", result) or "")
    sentences_raw = getattr(result, "sentences", None)
    sentences: list[dict[str, Any]] = []
    if isinstance(sentences_raw, list):
        sentences = [_sentence_to_dict(sentence) for sentence in sentences_raw]

    tokens_raw = getattr(result, "tokens", None)
    tokens: list[dict[str, Any]] = []
    if isinstance(tokens_raw, list):
        tokens = [_token_to_dict(token) for token in tokens_raw]
    elif sentences:
        for sentence in sentences:
            for token in sentence.get("tokens", []):
                tokens.append(token)

    return {
        "text": text,
        "sentences": sentences,
        "tokens": tokens,
    }


def _build_decoding_config(
    *,
    decoding_mode: Optional[str] = None,
    beam_size: Optional[int] = None,
    length_penalty: Optional[float] = None,
    patience: Optional[float] = None,
    duration_reward: Optional[float] = None,
    sentence_max_words: Optional[int] = None,
    sentence_silence_gap: Optional[float] = None,
    sentence_max_duration: Optional[float] = None,
) -> Optional[Any]:
    """Build an optional ``parakeet_mlx.DecodingConfig`` from user/config overrides."""
    has_sentence_overrides = any(
        value is not None
        for value in (sentence_max_words, sentence_silence_gap, sentence_max_duration)
    )
    mode = str(decoding_mode or "").strip().lower()
    has_beam_overrides = mode == "beam" or any(
        value is not None for value in (beam_size, length_penalty, patience, duration_reward)
    )
    if not has_sentence_overrides and not has_beam_overrides and mode != "greedy":
        return None

    try:
        import parakeet_mlx
    except Exception as exc:
        logger.debug("Unable to import parakeet_mlx for decoding config construction: {}", exc)
        return None

    SentenceConfig = getattr(parakeet_mlx, "SentenceConfig", None)
    DecodingConfig = getattr(parakeet_mlx, "DecodingConfig", None)
    Greedy = getattr(parakeet_mlx, "Greedy", None)
    Beam = getattr(parakeet_mlx, "Beam", None)
    if SentenceConfig is None or DecodingConfig is None or Greedy is None or Beam is None:
        logger.debug("parakeet_mlx decoding classes unavailable; skipping decoding config")
        return None

    sentence_obj = SentenceConfig()
    if sentence_max_words is not None:
        sentence_obj.max_words = int(sentence_max_words)
    if sentence_silence_gap is not None:
        sentence_obj.silence_gap = float(sentence_silence_gap)
    if sentence_max_duration is not None:
        sentence_obj.max_duration = float(sentence_max_duration)

    if mode == "beam" or has_beam_overrides:
        beam_kwargs: dict[str, Any] = {}
        if beam_size is not None:
            beam_kwargs["beam_size"] = int(beam_size)
        if length_penalty is not None:
            beam_kwargs["length_penalty"] = float(length_penalty)
        if patience is not None:
            beam_kwargs["patience"] = float(patience)
        if duration_reward is not None:
            beam_kwargs["duration_reward"] = float(duration_reward)
        decode_obj = Beam(**beam_kwargs) if beam_kwargs else Beam()
    else:
        decode_obj = Greedy()
    return DecodingConfig(decoding=decode_obj, sentence=sentence_obj)

#######################################################################################################################
# Installation and Setup
#

def check_mlx_available() -> bool:
    """Check if MLX is available and we're on macOS."""
    if not IS_MACOS:
        logger.debug("MLX is only available on macOS")
        return False

    try:
        mlx_spec = importlib.util.find_spec("mlx")
    except (ImportError, ModuleNotFoundError, ValueError):
        logger.debug("MLX not installed")
        return False
    if mlx_spec is None:
        logger.debug("MLX not installed")
        return False
    try:
        mlx_core_spec = importlib.util.find_spec("mlx.core")
    except (ImportError, ModuleNotFoundError, ValueError):
        logger.debug("MLX core not available")
        return False
    if mlx_core_spec is None:
        logger.debug("MLX core not available")
        return False
    return True


def check_parakeet_mlx_installed() -> bool:
    """Check if parakeet-mlx is installed."""
    try:
        return importlib.util.find_spec("parakeet_mlx") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Some test doubles register modules with __spec__ = None, which makes
        # find_spec() raise ValueError. If the module is already present in
        # sys.modules, treat it as available.
        return sys.modules.get("parakeet_mlx") is not None


def install_parakeet_mlx() -> bool:
    """
    Install parakeet-mlx package from GitHub.

    Returns:
        True if installation successful, False otherwise
    """
    if not IS_MACOS:
        logger.error("parakeet-mlx is only supported on macOS")
        return False

    if check_parakeet_mlx_installed():
        logger.info("parakeet-mlx is already installed")
        return True

    try:
        logger.info("Installing parakeet-mlx from GitHub...")

        # Install using pip
        cmd = [
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/senstella/parakeet-mlx.git"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Successfully installed parakeet-mlx")
            return True
        else:
            logger.error(f"Failed to install parakeet-mlx: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error installing parakeet-mlx: {e}")
        return False


#######################################################################################################################
# Model Loading and Management
#

def load_parakeet_mlx_model(
    *,
    force_reload: bool = False,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """
    Load the Parakeet MLX model.

    Args:
        force_reload: Force reload even if model is cached

    Returns:
        Loaded model instance or None if loading fails
    """
    global _mlx_model_cache

    if not IS_MACOS:
        logger.error("Parakeet MLX is only supported on macOS with Apple Silicon")
        return None

    if _mlx_model_cache and not force_reload:
        logger.debug("Using cached Parakeet MLX model")
        return _mlx_model_cache

    # Check MLX availability (tests may monkeypatch this to True)
    if not check_mlx_available():
        logger.error("MLX is not available. Install with: pip install mlx")
        return None

    # Check parakeet-mlx
    if not check_parakeet_mlx_installed():
        logger.error(
            "parakeet-mlx is not installed. Install during setup/deploy (runtime auto-install is disabled)."
        )
        return None

    try:
        import parakeet_mlx
        stt_cfg: dict[str, Any] = {}
        try:
            from tldw_Server_API.app.core.config import get_stt_config

            stt_cfg = get_stt_config() or {}
        except Exception:
            stt_cfg = {}

        # dtype is optional for tests; if mlx is unavailable, proceed without dtype
        try:
            import mlx.core as mx
            _dtype = getattr(mx, 'bfloat16', None)
        except Exception:
            _dtype = None

        logger.info("Loading Parakeet MLX model...")

        # Initialize the model
        # The parakeet-mlx library handles model downloading and caching
        # Try to load from Hugging Face model ID
        # Default model from the parakeet-mlx CLI (v3)
        model_id = (
            model_path
            or str(stt_cfg.get("mlx_model_id", "")).strip()
            or _DEFAULT_MLX_MODEL_ID
        )
        model_cache_dir = cache_dir or str(stt_cfg.get("mlx_cache_dir", "")).strip() or None
        from_pretrained_kwargs: dict[str, Any] = {}
        if _dtype is not None and _supports_kwarg(parakeet_mlx.from_pretrained, "dtype"):
            from_pretrained_kwargs["dtype"] = _dtype
        if model_cache_dir and _supports_kwarg(parakeet_mlx.from_pretrained, "cache_dir"):
            from_pretrained_kwargs["cache_dir"] = model_cache_dir

        try:
            # Try to load the model from Hugging Face
            logger.info(f"Loading model from: {model_id}")
            model = parakeet_mlx.from_pretrained(model_id, **from_pretrained_kwargs)
        except FileNotFoundError:
            # Model might need to be downloaded first
            logger.info("Model not found locally, downloading from Hugging Face...")
            try:
                # The model will be downloaded automatically
                model = parakeet_mlx.from_pretrained(model_id, **from_pretrained_kwargs)
            except Exception as e2:
                logger.exception(f"Failed to download/load model: {e2}")
                return None
        except Exception as e:
            logger.exception(f"Failed to load model {model_id}: {e}")
            return None

        _mlx_model_cache = model
        logger.info("Successfully loaded Parakeet MLX model")

        return model

    except ImportError as e:
        logger.exception(f"Failed to import parakeet: {e}")
        logger.info("Try installing manually: pip install git+https://github.com/senstella/parakeet-mlx.git")
        return None
    except Exception as e:
        logger.exception(f"Failed to load Parakeet MLX model: {e}")
        return None


#######################################################################################################################
# Transcription Functions
#

def transcribe_with_parakeet_mlx(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    language: Optional[str] = None,
    batch_size: int = 1,
    verbose: bool = False,
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 15.0,
    chunk_callback: Optional[Callable[[int, int], None]] = None,
    *,
    return_structured: bool = False,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    decoding_mode: Optional[str] = None,
    beam_size: Optional[int] = None,
    length_penalty: Optional[float] = None,
    patience: Optional[float] = None,
    duration_reward: Optional[float] = None,
    sentence_max_words: Optional[int] = None,
    sentence_silence_gap: Optional[float] = None,
    sentence_max_duration: Optional[float] = None,
) -> Union[str, dict[str, Any]]:
    """
    Transcribe audio using Parakeet MLX model.

    Args:
        audio_data: Audio data as numpy array, file path, or Path object
        sample_rate: Sample rate of audio (will be resampled if needed)
        language: Language hint (not used by Parakeet, included for compatibility)
        batch_size: Batch size for processing
        verbose: Enable verbose output
        chunk_duration: Duration in seconds for chunking long audio (None = no chunking)
        overlap_duration: Overlap between chunks in seconds (default 15.0)
        chunk_callback: Callback function for chunk progress (current, total)

    Returns:
        Transcribed text string
    """
    _ = (language, batch_size)
    # Attempt to load the model first (tests may monkeypatch loader)
    model = load_parakeet_mlx_model(model_path=model_path, cache_dir=cache_dir)
    if model is None:
        # Preserve the original platform check semantics only when loading fails
        if not IS_MACOS:
            return "[Error: Parakeet MLX is only supported on macOS with Apple Silicon]"
        return "[Error: Failed to load Parakeet MLX model]"

    try:
        # Handle different input types
        audio_file_path: Optional[str] = None
        audio_np: Optional[np.ndarray] = None

        if isinstance(audio_data, (str, Path)):
            # Already a file path
            audio_path = Path(audio_data)
            if not audio_path.exists():
                return "[Error: Audio file not found]"
            audio_file_path = str(audio_path)

        elif isinstance(audio_data, np.ndarray):
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if sample_rate != 16000:
                import librosa

                audio_data = librosa.resample(
                    audio_data,  # type: ignore[arg-type]
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            audio_np = np.asarray(audio_data, dtype=np.float32)
        else:
            return "[Error: Invalid audio data type]"

        # Normalize audio to [-1, 1] range (only if we have numpy array)
        if isinstance(audio_np, np.ndarray) and audio_np.size > 0 and np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()

        # Transcribe using parakeet-mlx
        if verbose:
            if isinstance(audio_np, np.ndarray):
                logger.info(f"Transcribing audio of length {len(audio_np)/16000:.2f} seconds")
            else:
                logger.info("Transcribing audio file")

        try:
            from tldw_Server_API.app.core.config import get_stt_config

            stt_cfg = get_stt_config() or {}
        except Exception:
            stt_cfg = {}

        transcribe_kwargs: dict[str, Any] = {}
        if chunk_duration is not None:
            transcribe_kwargs['chunk_duration'] = chunk_duration
            transcribe_kwargs['overlap_duration'] = overlap_duration
        if chunk_callback is not None:
            transcribe_kwargs['chunk_callback'] = chunk_callback

        decoding_mode = decoding_mode or str(stt_cfg.get("mlx_decoding_mode", "")).strip() or None
        beam_size = beam_size if beam_size is not None else _safe_int(stt_cfg.get("mlx_beam_size"))
        length_penalty = (
            length_penalty if length_penalty is not None else _safe_float(stt_cfg.get("mlx_length_penalty"))
        )
        patience = patience if patience is not None else _safe_float(stt_cfg.get("mlx_patience"))
        duration_reward = (
            duration_reward if duration_reward is not None else _safe_float(stt_cfg.get("mlx_duration_reward"))
        )
        sentence_max_words = (
            sentence_max_words
            if sentence_max_words is not None
            else _safe_int(stt_cfg.get("mlx_sentence_max_words"))
        )
        sentence_silence_gap = (
            sentence_silence_gap
            if sentence_silence_gap is not None
            else _safe_float(stt_cfg.get("mlx_sentence_silence_gap"))
        )
        sentence_max_duration = (
            sentence_max_duration
            if sentence_max_duration is not None
            else _safe_float(stt_cfg.get("mlx_sentence_max_duration"))
        )

        decoding_config = _build_decoding_config(
            decoding_mode=decoding_mode,
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
            sentence_max_words=sentence_max_words,
            sentence_silence_gap=sentence_silence_gap,
            sentence_max_duration=sentence_max_duration,
        )
        if decoding_config is not None and _supports_kwarg(model.transcribe, "decoding_config"):
            transcribe_kwargs["decoding_config"] = decoding_config

        temp_audio_path: Optional[str] = None
        try:
            if audio_file_path:
                result = model.transcribe(audio_file_path, **transcribe_kwargs)
            elif isinstance(audio_np, np.ndarray):
                import soundfile as sf

                # Persist numpy audio to a temp file so downstream callers (and tests)
                # receive a filesystem path consistently.
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_np, 16000, format='WAV')
                    temp_audio_path = tmp_file.name
                result = model.transcribe(temp_audio_path, **transcribe_kwargs)
            else:
                return "[Error: Invalid audio data format]"
        finally:
            if temp_audio_path:
                try:
                    os.remove(temp_audio_path)
                except Exception as rm_err:
                    logger.debug(f"Failed to remove temp audio file (Parakeet_MLX): path={temp_audio_path}, error={rm_err}")

        artifact = _as_structured_artifact(result)
        text = artifact["text"]

        if verbose:
            logger.info(f"Transcription complete: {text[:100]}...")

        if return_structured:
            return artifact
        return text

    except ImportError as e:
        logger.exception(f"Missing required library: {e}")
        return "[Error: Missing required library]"
    except Exception as e:
        import traceback
        logger.exception(f"Error during Parakeet MLX transcription: {e}")
        logger.exception(f"Traceback: {traceback.format_exc()}")
        return "[Error: Parakeet MLX transcription failed]"


@dataclass
class ParakeetMLXStreamingSession:
    """Thin wrapper around parakeet-mlx StreamingParakeet."""

    model: Any
    streamer: Any
    _closed: bool = False

    def add_audio(self, audio_np: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
        if self._closed:
            return {"text": "", "sentences": [], "tokens": []}
        if not isinstance(audio_np, np.ndarray):
            return {"text": "", "sentences": [], "tokens": []}
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1).astype(np.float32)
        if sample_rate != 16000:
            import librosa

            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
        if audio_np.size == 0:
            return {"text": "", "sentences": [], "tokens": []}

        import mlx.core as mx

        self.streamer.add_audio(mx.array(audio_np, dtype=mx.float32))
        return _as_structured_artifact(self.streamer.result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.streamer.__exit__(None, None, None)
        except Exception as exc:
            logger.debug(f"Parakeet MLX streaming session close warning: {exc}")


def create_parakeet_mlx_streaming_session(
    *,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    context_size: tuple[int, int] = (256, 256),
    depth: int = 1,
    keep_original_attention: bool = False,
    decoding_mode: Optional[str] = None,
    beam_size: Optional[int] = None,
    length_penalty: Optional[float] = None,
    patience: Optional[float] = None,
    duration_reward: Optional[float] = None,
    sentence_max_words: Optional[int] = None,
    sentence_silence_gap: Optional[float] = None,
    sentence_max_duration: Optional[float] = None,
) -> Optional[ParakeetMLXStreamingSession]:
    model = load_parakeet_mlx_model(model_path=model_path, cache_dir=cache_dir)
    if model is None:
        return None
    if not hasattr(model, "transcribe_stream"):
        logger.warning("Loaded Parakeet MLX model does not expose transcribe_stream; streaming session unavailable")
        return None

    try:
        decoding_config = _build_decoding_config(
            decoding_mode=decoding_mode,
            beam_size=beam_size,
            length_penalty=length_penalty,
            patience=patience,
            duration_reward=duration_reward,
            sentence_max_words=sentence_max_words,
            sentence_silence_gap=sentence_silence_gap,
            sentence_max_duration=sentence_max_duration,
        )
        stream_kwargs: dict[str, Any] = {
            "context_size": context_size,
            "depth": depth,
        }
        if decoding_config is not None and _supports_kwarg(model.transcribe_stream, "decoding_config"):
            stream_kwargs["decoding_config"] = decoding_config
        if _supports_kwarg(model.transcribe_stream, "keep_original_attention"):
            stream_kwargs["keep_original_attention"] = keep_original_attention

        streamer = model.transcribe_stream(**stream_kwargs)
        if hasattr(streamer, "__enter__"):
            streamer = streamer.__enter__()
        return ParakeetMLXStreamingSession(model=model, streamer=streamer)
    except Exception as exc:
        logger.warning(f"Failed to create Parakeet MLX streaming session: {exc}")
        return None


def transcribe_streaming_mlx(
    audio_stream,
    chunk_size: int = 16000,  # 1 second chunks at 16kHz
    overlap: float = 0.1,      # 10% overlap
    verbose: bool = False
) -> list[str]:
    """
    Transcribe audio stream using Parakeet MLX.

    Args:
        audio_stream: Iterator or generator yielding audio chunks
        chunk_size: Size of chunks to process
        overlap: Overlap ratio between chunks
        verbose: Enable verbose output

    Yields:
        Transcribed text for each chunk
    """
    if not IS_MACOS:
        yield "[Error: Parakeet MLX is only supported on macOS]"
        return

    session = create_parakeet_mlx_streaming_session()
    if session is None:
        model = load_parakeet_mlx_model()
        if model is None:
            yield "[Error: Failed to load model]"
            return

    buffer = np.array([], dtype=np.float32)
    overlap_size = int(chunk_size * overlap)

    try:
        for audio_chunk in audio_stream:
            # Add chunk to buffer
            buffer = np.concatenate([buffer, audio_chunk])

            # Process when we have enough data
            while len(buffer) >= chunk_size:
                # Extract chunk
                chunk = buffer[:chunk_size]

                # Transcribe chunk
                if session is not None:
                    artifact = session.add_audio(chunk, sample_rate=16000)
                    text = artifact.get("text", "")
                else:
                    text = transcribe_with_parakeet_mlx(
                        chunk,
                        sample_rate=16000,
                        verbose=verbose
                    )

                if text and not text.startswith("[Error"):
                    yield text

                # Move buffer forward with overlap
                buffer = buffer[chunk_size - overlap_size:]

        # Process remaining buffer
        if len(buffer) > 0:
            if session is not None:
                artifact = session.add_audio(buffer, sample_rate=16000)
                text = artifact.get("text", "")
            else:
                text = transcribe_with_parakeet_mlx(
                    buffer,
                    sample_rate=16000,
                    verbose=verbose
                )
            if text and not text.startswith("[Error"):
                yield text

    except Exception as e:
        logger.exception(f"Error in streaming transcription: {e}")
        yield "[Error: Parakeet MLX streaming transcription failed]"
    finally:
        if session is not None:
            session.close()


def unload_parakeet_mlx_model():
    """Unload the cached Parakeet MLX model to free memory."""
    global _mlx_model_cache

    if _mlx_model_cache is not None:
        try:
            # MLX models can be deleted directly
            del _mlx_model_cache
            _mlx_model_cache = None

            # MLX specific cleanup
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception as cache_err:
                logger.debug(f"Failed to clear MLX cache: error={cache_err}")

            # Force garbage collection
            import gc
            gc.collect()

            logger.info("Unloaded Parakeet MLX model from memory")
        except Exception as e:
            logger.exception(f"Error unloading model: {e}")


#######################################################################################################################
# Utility Functions
#

def get_mlx_device_info() -> dict[str, Any]:
    """Get information about MLX device and capabilities."""
    info = {
        'available': False,
        'platform': sys.platform,
        'is_apple_silicon': False,
        'metal_available': False
    }

    if not IS_MACOS:
        return info

    try:
        import mlx.core as mx

        info['available'] = True
        info['is_apple_silicon'] = True
        info['metal_available'] = mx.metal.is_available()

        # Get device info
        if info['metal_available']:
            info['device'] = 'Metal GPU'
            # Additional Metal info could be added here

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error getting MLX device info: {e}")

    return info


def benchmark_parakeet_mlx(audio_duration: float = 10.0) -> dict[str, float]:
    """
    Benchmark Parakeet MLX performance.

    Args:
        audio_duration: Duration of test audio in seconds

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'status': 'failed',
        'audio_duration': audio_duration,
        'transcription_time': 0,
        'real_time_factor': 0,
        'model_load_time': 0
    }

    if not IS_MACOS:
        results['error'] = 'Not on macOS'
        return results

    try:
        import time

        # Time model loading
        start = time.time()
        model = load_parakeet_mlx_model(force_reload=True)
        results['model_load_time'] = time.time() - start

        if model is None:
            results['error'] = 'Failed to load model'
            return results

        # Create test audio (silence)
        sample_rate = 16000
        test_audio = np.zeros(int(audio_duration * sample_rate), dtype=np.float32)

        # Time transcription
        start = time.time()
        text = transcribe_with_parakeet_mlx(test_audio, sample_rate=sample_rate)
        results['transcription_time'] = time.time() - start

        # Calculate real-time factor
        results['real_time_factor'] = audio_duration / results['transcription_time']
        results['status'] = 'success'
        results['transcription_length'] = len(text) if text and not text.startswith("[Error") else 0

    except Exception as e:
        results['error'] = str(e)

    return results


#######################################################################################################################
# Integration with main Nemo module
#

def integrate_with_nemo_module():
    """
    Update the main Audio_Transcription_Nemo.py to use this MLX implementation.
    This function patches the _load_parakeet_mlx function.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Nemo

        # Replace the placeholder MLX loader with our implementation
        def _load_parakeet_mlx_patched():
            """Load MLX variant of Parakeet model using specialized implementation."""
            return load_parakeet_mlx_model()

        # Patch the function
        Audio_Transcription_Nemo._load_parakeet_mlx = _load_parakeet_mlx_patched

        logger.info("Successfully integrated Parakeet MLX with main Nemo module")
        return True

    except Exception as e:
        logger.exception(f"Failed to integrate with Nemo module: {e}")
        return False


#######################################################################################################################
# Main entry point for testing
#

if __name__ == "__main__":
    print("Parakeet MLX Module Information:")
    print("-" * 40)

    # Check system
    info = get_mlx_device_info()
    print(f"Platform: {info['platform']}")
    print(f"MLX Available: {info['available']}")
    print(f"Apple Silicon: {info['is_apple_silicon']}")
    print(f"Metal Available: {info.get('metal_available', False)}")

    if not IS_MACOS:
        print("\nThis module requires macOS with Apple Silicon")
        sys.exit(1)

    # Try to load model
    print("\nLoading Parakeet MLX model...")
    model = load_parakeet_mlx_model()

    if model:
        print("✓ Model loaded successfully")

        # Run benchmark
        print("\nRunning benchmark...")
        results = benchmark_parakeet_mlx(audio_duration=5.0)

        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"Model Load Time: {results['model_load_time']:.2f}s")
            print(f"Transcription Time: {results['transcription_time']:.2f}s")
            print(f"Real-time Factor: {results['real_time_factor']:.2f}x")
        else:
            print(f"Error: {results.get('error', 'Unknown')}")
    else:
        print("✗ Failed to load model")
        print("Try: pip install git+https://github.com/senstella/parakeet-mlx.git")


#######################################################################################################################
# End of Audio_Transcription_Parakeet_MLX.py
#######################################################################################################################
