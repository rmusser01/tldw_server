# audio_utils.py
# Description: Audio processing utilities for TTS voice cloning/reference
#
# Imports
import asyncio
import base64
import binascii
import importlib.util
import re
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

_AUDIO_PROCESS_EXCEPTIONS = (
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_AUDIO_BASE64_EXCEPTIONS = (binascii.Error, TypeError, ValueError)
_AUDIO_INT_PARSE_EXCEPTIONS = (OverflowError, TypeError, ValueError)

#
#######################################################################################################################
#
# Audio Processing Utilities for Voice Cloning

class AudioProcessor:
    """Process and validate audio for voice cloning/reference"""

    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    # Provider-specific requirements
    PROVIDER_REQUIREMENTS = {
        'higgs': {
            'min_duration': 3.0,
            'max_duration': 10.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3', '.flac'}
        },
        'chatterbox': {
            'min_duration': 5.0,
            'max_duration': 20.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3'}
        },
        'vibevoice': {
            'min_duration': 3.0,
            'max_duration': 30.0,
            'preferred_sample_rate': 22050,
            'formats': {'.wav', '.mp3'}
        },
        'pocket_tts': {
            'min_duration': 1.0,
            'max_duration': 60.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        },
        'lux_tts': {
            'min_duration': 3.0,
            'max_duration': 60.0,
            'preferred_sample_rate': 24000,
            'formats': {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        },
        'echo_tts': {
            'min_duration': 1.0,
            'max_duration': 300.0,
            'preferred_sample_rate': 44100,
            'formats': {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        }
    }

    def __init__(self):
        """Initialize audio processor"""
        self.ffmpeg_path: Optional[str] = None
        self.ffmpeg_available = self._check_ffmpeg()
        self.librosa_available = self._check_librosa()

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                return False
            result = subprocess.run(  # nosec B603
                [ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                check=False,
            )
            self.ffmpeg_path = ffmpeg_path
            return result.returncode == 0
        except _AUDIO_PROCESS_EXCEPTIONS as e:
            logger.warning(
                "ffmpeg not found or not runnable; audio conversion limited ({})",
                type(e).__name__,
            )
            return False

    def _check_librosa(self) -> bool:
        """Check if librosa is available for audio processing"""
        if importlib.util.find_spec("librosa") is None:
            logger.warning("librosa not installed - advanced audio processing unavailable")
            return False
        return True

    def decode_base64_audio(self, base64_data: str) -> bytes:
        """
        Decode base64-encoded audio data.

        Args:
            base64_data: Base64-encoded audio string

        Returns:
            Raw audio bytes

        Raises:
            ValueError: If decoding fails
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]

            # Decode base64
            audio_bytes = base64.b64decode(base64_data)
            return audio_bytes

        except _AUDIO_BASE64_EXCEPTIONS as e:
            raise ValueError(f"Failed to decode base64 audio: {e}") from e

    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """
        Encode audio bytes to base64.

        Args:
            audio_bytes: Raw audio bytes

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(audio_bytes).decode('utf-8')

    def validate_audio(
        self,
        audio_bytes: bytes,
        provider: str,
        check_duration: bool = True,
        check_quality: bool = False
    ) -> tuple[bool, Optional[str], dict[str, Any]]:
        """
        Validate audio for a specific provider.

        Args:
            audio_bytes: Raw audio bytes
            provider: TTS provider name
            check_duration: Whether to check duration requirements
            check_quality: Whether to check audio quality (noise, etc.)

        Returns:
            Tuple of (is_valid, error_message, audio_info)
        """
        info = {}

        if provider.lower() not in self.PROVIDER_REQUIREMENTS:
            return True, None, info  # No specific requirements

        requirements = self.PROVIDER_REQUIREMENTS[provider.lower()]

        if not self.librosa_available:
            logger.warning("Cannot validate audio without librosa")
            return True, None, info

        try:
            import librosa

            # Write to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)

                # Get audio info
                info['sample_rate'] = sample_rate
                info['duration'] = len(audio_data) / sample_rate
                info['channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[0]

                # Check duration
                if check_duration:
                    if info['duration'] < requirements['min_duration']:
                        return False, f"Audio too short: {info['duration']:.1f}s (minimum {requirements['min_duration']}s)", info
                    if info['duration'] > requirements['max_duration']:
                        return False, f"Audio too long: {info['duration']:.1f}s (maximum {requirements['max_duration']}s)", info

                # Check quality (optional)
                if check_quality:
                    # Check for silence
                    rms = np.sqrt(np.mean(audio_data**2))
                    if rms < 0.001:
                        return False, "Audio appears to be silent", info

                    # Check for clipping
                    if np.any(np.abs(audio_data) > 0.99):
                        info['has_clipping'] = True
                        logger.warning("Audio may have clipping")

                return True, None, info

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except _AUDIO_PROCESS_EXCEPTIONS as e:
            logger.error("Audio validation error ({})", type(e).__name__)
            return False, f"Failed to validate audio: {e}", info

    def convert_audio(
        self,
        audio_bytes: bytes,
        target_format: str = 'wav',
        target_sample_rate: Optional[int] = None,
        provider: Optional[str] = None
    ) -> bytes:
        """
        Convert audio to target format and sample rate.

        Args:
            audio_bytes: Input audio bytes
            target_format: Target format (wav, mp3, flac)
            target_sample_rate: Target sample rate (Hz)
            provider: Provider name for specific requirements

        Returns:
            Converted audio bytes
        """
        if not self.ffmpeg_available and not self.librosa_available:
            logger.warning("No audio conversion libraries available")
            return audio_bytes

        # Get provider requirements if specified
        if provider and provider.lower() in self.PROVIDER_REQUIREMENTS:
            requirements = self.PROVIDER_REQUIREMENTS[provider.lower()]
            if not target_sample_rate:
                target_sample_rate = requirements['preferred_sample_rate']

        try:
            if self.librosa_available:
                import librosa
                import soundfile as sf

                # Write input to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
                    input_file.write(audio_bytes)
                    input_path = input_file.name

                # Output temp file
                output_suffix = f'.{target_format.lower()}'
                with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
                    output_path = output_file.name

                try:
                    # Load and resample
                    audio_data, original_sr = librosa.load(input_path, sr=None)

                    if target_sample_rate and target_sample_rate != original_sr:
                        audio_data = librosa.resample(
                            audio_data,
                            orig_sr=original_sr,
                            target_sr=target_sample_rate
                        )
                        sample_rate = target_sample_rate
                    else:
                        sample_rate = original_sr

                    # Write output
                    sf.write(output_path, audio_data, sample_rate, format=target_format.upper())

                    # Read converted audio
                    with open(output_path, 'rb') as f:
                        converted_bytes = f.read()

                    return converted_bytes

                finally:
                    # Clean up temp files
                    Path(input_path).unlink(missing_ok=True)
                    Path(output_path).unlink(missing_ok=True)

            elif self.ffmpeg_available:
                # Use ffmpeg for conversion
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
                    input_file.write(audio_bytes)
                    input_path = input_file.name

                output_suffix = f'.{target_format.lower()}'
                with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
                    output_path = output_file.name

                try:
                    ffmpeg_path = getattr(self, 'ffmpeg_path', None) or shutil.which('ffmpeg')
                    if not ffmpeg_path:
                        raise RuntimeError("ffmpeg executable not found")

                    # Build ffmpeg command
                    cmd = [ffmpeg_path, '-i', input_path, '-y']

                    if target_sample_rate:
                        cmd.extend(['-ar', str(target_sample_rate)])

                    cmd.append(output_path)

                    # Run conversion
                    result = subprocess.run(  # nosec B603
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

                    # Read converted audio
                    with open(output_path, 'rb') as f:
                        converted_bytes = f.read()

                    return converted_bytes

                finally:
                    # Clean up temp files
                    Path(input_path).unlink(missing_ok=True)
                    Path(output_path).unlink(missing_ok=True)

        except _AUDIO_PROCESS_EXCEPTIONS as e:
            logger.error(f"Audio conversion failed ({type(e).__name__})")
            return audio_bytes  # Return original if conversion fails

    async def convert_audio_async(
        self,
        audio_bytes: bytes,
        target_format: str = 'wav',
        target_sample_rate: Optional[int] = None,
        provider: Optional[str] = None
    ) -> bytes:
        """
        Async-friendly wrapper around convert_audio.

        Offloads the potentially blocking conversion (ffmpeg/librosa) to a
        background thread so it does not block the event loop when called
        from async adapters.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.convert_audio(
                audio_bytes,
                target_format=target_format,
                target_sample_rate=target_sample_rate,
                provider=provider,
            ),
        )

    def extract_clean_segment(
        self,
        audio_bytes: bytes,
        target_duration: float = 10.0
    ) -> bytes:
        """
        Extract a clean segment from audio (remove silence, find best part).

        Args:
            audio_bytes: Input audio bytes
            target_duration: Target duration in seconds

        Returns:
            Extracted audio segment bytes
        """
        if not self.librosa_available:
            return audio_bytes

        try:
            import librosa
            import soundfile as sf

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_path, sr=None)

                # Remove silence from beginning and end
                audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)

                # If still too long, take the middle portion
                if len(audio_trimmed) / sample_rate > target_duration:
                    target_samples = int(target_duration * sample_rate)
                    start = (len(audio_trimmed) - target_samples) // 2
                    audio_trimmed = audio_trimmed[start:start + target_samples]

                # Write output
                output_path = tmp_path + '_clean.wav'
                sf.write(output_path, audio_trimmed, sample_rate)

                # Read cleaned audio
                with open(output_path, 'rb') as f:
                    clean_bytes = f.read()

                Path(output_path).unlink(missing_ok=True)
                return clean_bytes

            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except _AUDIO_PROCESS_EXCEPTIONS as e:
            logger.error("Failed to extract clean segment ({})", type(e).__name__)
            return audio_bytes


def process_voice_reference(
    base64_audio: str,
    provider: str,
    validate: bool = True,
    convert: bool = True
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Process voice reference audio for a specific provider.

    Args:
        base64_audio: Base64-encoded audio data
        provider: TTS provider name
        validate: Whether to validate audio
        convert: Whether to convert to optimal format

    Returns:
        Tuple of (processed_audio_bytes, error_message)
    """
    processor = AudioProcessor()

    try:
        # Decode base64
        audio_bytes = processor.decode_base64_audio(base64_audio)

        # Validate if requested
        if validate:
            is_valid, error_msg, info = processor.validate_audio(
                audio_bytes,
                provider,
                check_duration=True,
                check_quality=True
            )

            if not is_valid:
                return None, error_msg

            logger.info(f"Voice reference validated: {info}")

        # Convert if requested
        if convert and provider.lower() in processor.PROVIDER_REQUIREMENTS:
            processor.PROVIDER_REQUIREMENTS[provider.lower()]
            audio_bytes = processor.convert_audio(
                audio_bytes,
                target_format='wav',
                provider=provider
            )
            logger.info(f"Voice reference converted for {provider}")

        return audio_bytes, None

    except _AUDIO_PROCESS_EXCEPTIONS as e:
        logger.error("Failed to process voice reference ({})", type(e).__name__)
        return None, str(e)


async def process_voice_reference_async(
    base64_audio: str,
    provider: str,
    validate: bool = True,
    convert: bool = True
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Async-friendly variant of process_voice_reference for use in adapters.

    Uses the same validation logic but offloads heavy conversion work to a
    background thread so it does not block the event loop.
    """
    processor = AudioProcessor()

    try:
        # Decode base64
        audio_bytes = processor.decode_base64_audio(base64_audio)

        # Validate if requested
        if validate:
            is_valid, error_msg, info = processor.validate_audio(
                audio_bytes,
                provider,
                check_duration=True,
                check_quality=True
            )

            if not is_valid:
                return None, error_msg

            logger.info(f"Voice reference validated: {info}")

        # Convert if requested
        if convert and provider.lower() in processor.PROVIDER_REQUIREMENTS:
            audio_bytes = await processor.convert_audio_async(
                audio_bytes,
                target_format='wav',
                provider=provider,
            )
            logger.info(f"Voice reference converted for {provider}")

        return audio_bytes, None

    except _AUDIO_PROCESS_EXCEPTIONS as e:
        logger.error("Failed to process voice reference (async) ({})", type(e).__name__)
        return None, str(e)

def split_text_into_chunks(
    text: str,
    target_chars: int = 120,
    max_chars: int = 150,
    min_chars: int = 50,
) -> list[str]:
    """Split text into sentence-based chunks with soft length targets."""
    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?。！？])\s*", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            words = sentence.split()
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 <= target_chars:
                    buf = f"{buf} {word}".strip()
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = word
            if buf:
                chunks.append(buf.strip())
            continue

        if len(current) + len(sentence) + 1 <= target_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def crossfade_audio(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Crossfade two audio arrays by a fixed duration."""
    if left is None or left.size == 0:
        return right
    if right is None or right.size == 0:
        return left

    try:
        fade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    except _AUDIO_INT_PARSE_EXCEPTIONS:
        fade_samples = 0
    if fade_samples <= 0:
        return np.concatenate([left, right])

    fade_samples = min(fade_samples, left.shape[-1], right.shape[-1])
    if fade_samples <= 0:
        return np.concatenate([left, right])

    def _to_float(audio: np.ndarray) -> np.ndarray:
        if np.issubdtype(audio.dtype, np.integer):
            return audio.astype(np.float32) / np.iinfo(audio.dtype).max
        return audio.astype(np.float32)

    left_f = _to_float(left)
    right_f = _to_float(right)
    t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    left_tail = left_f[-fade_samples:] * (1.0 - t)
    right_head = right_f[:fade_samples] * t
    merged = np.concatenate([left_f[:-fade_samples], left_tail + right_head, right_f[fade_samples:]])

    if np.issubdtype(left.dtype, np.integer):
        max_val = np.iinfo(left.dtype).max
        merged = np.clip(merged, -1.0, 1.0) * max_val
        return merged.astype(left.dtype)
    return merged.astype(np.float32)


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= arr.shape[-1] else arr.mean(axis=1)
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        max_val = np.iinfo(arr.dtype).max
        if max_val > 0:
            return arr.astype(np.float32) / float(max_val)
    return arr.astype(np.float32)


def compute_audio_rms(audio: np.ndarray) -> float:
    """Compute RMS for a PCM audio buffer."""
    audio_f = _to_float_mono(audio)
    if audio_f.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_f ** 2)))


def compute_audio_peak(audio: np.ndarray) -> float:
    """Compute peak absolute amplitude for a PCM audio buffer."""
    audio_f = _to_float_mono(audio)
    if audio_f.size == 0:
        return 0.0
    return float(np.max(np.abs(audio_f)))


def trailing_silence_duration_ms(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
) -> int:
    """Return trailing silence duration in milliseconds."""
    if sample_rate <= 0:
        return 0
    audio_f = _to_float_mono(audio)
    if audio_f.size == 0:
        return 0
    mask = np.abs(audio_f) > float(threshold)
    if not np.any(mask):
        return int(round((len(audio_f) / float(sample_rate)) * 1000))
    last_idx = int(np.where(mask)[0][-1])
    trailing_samples = len(audio_f) - last_idx - 1
    return int(round((trailing_samples / float(sample_rate)) * 1000))


def trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    min_silence_ms: int = 0,
) -> np.ndarray:
    """Trim trailing silence from a PCM buffer, preserving dtype."""
    if sample_rate <= 0:
        return audio
    audio_f = _to_float_mono(audio)
    if audio_f.size == 0:
        return audio
    mask = np.abs(audio_f) > float(threshold)
    if not np.any(mask):
        return audio
    last_idx = int(np.where(mask)[0][-1])
    trailing_samples = len(audio_f) - last_idx - 1
    if min_silence_ms > 0:
        min_samples = int(sample_rate * (min_silence_ms / 1000.0))
        if trailing_samples < min_samples:
            return audio
    return np.asarray(audio).reshape(-1)[: last_idx + 1]


def analyze_audio_signal(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.01,
) -> dict[str, float]:
    """Compute basic audio metrics for quality checks."""
    audio_f = _to_float_mono(audio)
    duration_sec = len(audio_f) / float(sample_rate or 1)
    rms = compute_audio_rms(audio_f)
    peak = compute_audio_peak(audio_f)
    trailing_ms = trailing_silence_duration_ms(audio_f, sample_rate, threshold=silence_threshold)
    return {
        "duration_sec": float(duration_sec),
        "rms": float(rms),
        "peak": float(peak),
        "trailing_silence_ms": float(trailing_ms),
    }


def evaluate_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    *,
    text_length: int = 0,
    min_text_length: int = 40,
    min_rms: float = 0.001,
    min_peak: float = 0.02,
    silence_threshold: float = 0.01,
    trailing_silence_ms: int = 800,
    expected_chars_per_sec: float = 15.0,
    min_duration_ratio: float = 0.5,
    min_duration_seconds: float = 0.4,
) -> tuple[dict[str, float], list[str]]:
    """Evaluate audio quality and return metrics + warning codes."""
    metrics = analyze_audio_signal(audio, sample_rate, silence_threshold=silence_threshold)
    warnings: list[str] = []

    if metrics["rms"] < min_rms or metrics["peak"] < min_peak:
        warnings.append(
            f"low_levels(rms={metrics['rms']:.4f}, peak={metrics['peak']:.4f})"
        )

    if trailing_silence_ms > 0 and metrics["trailing_silence_ms"] >= trailing_silence_ms:
        warnings.append(f"trailing_silence_ms={metrics['trailing_silence_ms']:.0f}")

    if expected_chars_per_sec > 0 and text_length >= min_text_length:
        expected = text_length / float(expected_chars_per_sec)
        min_expected = max(min_duration_seconds, expected * min_duration_ratio)
        if metrics["duration_sec"] < min_expected:
            warnings.append(
                f"duration_short(actual={metrics['duration_sec']:.2f}s, expected>={min_expected:.2f}s)"
            )

    return metrics, warnings

#
# End of audio_utils.py
#######################################################################################################################
