# streaming_audio_writer.py
# Description: Handles streaming audio format conversions for TTS
# Based on Kokoro-FastAPI implementation
#
import os
import tempfile
import wave
from io import BytesIO
from typing import Optional

import av
import numpy as np
from loguru import logger


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class StreamingAudioWriter:
    """Handles streaming audio format conversions for TTS output.

    WAV output defers emission until finalize so headers can be rewritten.
    When WAV data exceeds `max_in_memory_bytes`, raw PCM is spooled to a temp
    file to avoid unbounded RAM use; prefer mp3/opus for long/real-time streams.
    """

    def __init__(
        self,
        format: str,
        sample_rate: int,
        channels: int = 1,
        max_in_memory_bytes: int = 8 * 1024 * 1024,
    ):
        """
        Initialize the streaming audio writer.

        Args:
            format: Target audio format (wav, mp3, opus, flac, aac, pcm)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            max_in_memory_bytes: Soft limit before WAV buffering spills to a temp file
        """
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0  # Presentation timestamp for audio frames
        self.max_in_memory_bytes = max(1024, int(max_in_memory_bytes or 0))
        self._wav_file_path: Optional[str] = None
        self._wav_chunk_size = 64 * 1024

        # Map formats to codecs
        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }

        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            # WAV is handled via raw PCM buffering (with optional disk spill) so we
            # can rewrite the header on finalize without holding unbounded memory.
            if self.format == "wav":
                self.output_buffer = BytesIO()
            elif self.format != "pcm":
                self.output_buffer = BytesIO()
                container_options = {}

                # Disable Xing VBR header for MP3 to fix iOS timeline reading issues
                if self.format == "mp3":
                    container_options = {"write_xing": "0"}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")

                # Open the container for writing
                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options,
                )

                # Add audio stream with appropriate codec
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )

                # Set bit rate for applicable codecs
                if self.format in ["mp3", "aac", "opus"]:
                    # Use reasonable default bitrates
                    if self.format == "mp3":
                        self.stream.bit_rate = 128000  # 128 kbps
                    elif self.format == "aac":
                        self.stream.bit_rate = 96000  # 96 kbps
                    elif self.format == "opus":
                        self.stream.bit_rate = 64000  # 64 kbps
        else:
            raise ValueError(f"Unsupported audio format: {self.format}")

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """
        Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: NumPy array of audio samples (int16). When `finalize` is
                True, this argument is ignored.
            finalize: If True, flush and finalize the stream. The expected call
                pattern is one or more `write_chunk(audio_data=...)` calls
                followed by a final `write_chunk(finalize=True)` call; do not
                combine `audio_data` and `finalize=True` in a single call.

        Returns:
            Bytes of encoded audio in the target format
        """
        if self.format == "wav":
            return self._write_chunk_wav(audio_data=audio_data, finalize=finalize)

        if finalize:
            if self.format != "pcm":
                # Flush stream encoder
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)

                # Close the container to write final data
                self.container.close()
                self.container = None

                # Return only the new bytes written since the last chunk,
                # to avoid duplicating audio when concatenating chunks.
                current_pos = self.output_buffer.tell()
                self.output_buffer.seek(self.bytes_written)
                data = self.output_buffer.read(current_pos - self.bytes_written)
                self.bytes_written = current_pos
                logger.debug(
                    f"StreamingAudioWriter finalize: format={self.format}, new_bytes={len(data)}, "
                    f"total_bytes={self.bytes_written}"
                )
                self.output_buffer.close()
                self.output_buffer = None
                return data
            else:
                logger.debug("StreamingAudioWriter finalize: PCM format, no trailer bytes")
                return b""

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # For PCM, just return raw bytes
            data = audio_data.tobytes()
            logger.debug(
                f"StreamingAudioWriter chunk: format=pcm, samples={len(audio_data)}, bytes={len(data)}"
            )
            return data
        else:
            # Create audio frame from numpy array
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo"
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            self.pts += frame.samples

            # Encode the frame
            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            # For streaming, we need to read what's been written so far
            # This is not ideal for all formats but works for streaming
            current_pos = self.output_buffer.tell()
            self.output_buffer.seek(self.bytes_written)
            data = self.output_buffer.read(current_pos - self.bytes_written)
            self.bytes_written = current_pos
            logger.debug(
                f"StreamingAudioWriter chunk: format={self.format}, samples={len(audio_data)}, "
                f"new_bytes={len(data)}, total_bytes={self.bytes_written}"
            )
            return data

    def close(self):
        """Clean up resources."""
        container = getattr(self, "container", None)
        if container is not None:
            try:
                # Some container implementations do not expose .closed; best-effort close.
                container.close()
                self.container = None
            except Exception as e:
                logger.error(
                    f"Error closing container; exception_type={_safe_exception_label(e)}"
                )

        buf = getattr(self, "output_buffer", None)
        if buf is not None:
            try:
                buf.close()
                self.output_buffer = None
            except Exception as e:
                logger.error(
                    f"Error closing output buffer; exception_type={_safe_exception_label(e)}"
                )

        wav_path = getattr(self, "_wav_file_path", None)
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError as e:
                logger.debug(
                    f"Error removing WAV temp file; exception_type={_safe_exception_label(e)}"
                )
            finally:
                self._wav_file_path = None

    #
    # WAV-specific helpers
    #
    def _write_chunk_wav(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Specialized handling for WAV to allow header rewrite and disk spill."""
        if finalize:
            return self._finalize_wav()

        if audio_data is None or len(audio_data) == 0:
            return b""

        data = audio_data.tobytes()
        self.bytes_written += len(data)

        if self._wav_file_path:
            with open(self._wav_file_path, "ab") as f:
                f.write(data)
        else:
            self.output_buffer.write(data)
            current_size = self.output_buffer.tell()
            if current_size > self.max_in_memory_bytes:
                self._spill_wav_buffer_to_file()

        logger.debug(
            f"StreamingAudioWriter chunk: format=wav, samples={len(audio_data)}, "
            f"new_bytes=0, total_bytes={self.bytes_written} (deferred until finalize)"
        )
        return b""

    def _spill_wav_buffer_to_file(self) -> None:
        """Persist the current in-memory PCM buffer to a temp file and continue writing there."""
        try:
            fd, path = tempfile.mkstemp(prefix="tts_wav_pcm_", suffix=".pcm")
            with os.fdopen(fd, "wb") as tmp:
                self.output_buffer.seek(0)
                tmp.write(self.output_buffer.read())
            self.output_buffer.close()
            self.output_buffer = None  # type: ignore[assignment]
            self._wav_file_path = path
            logger.warning(
                "StreamingAudioWriter WAV buffer spilled to disk "
                f"after exceeding {self.max_in_memory_bytes} bytes"
            )
        except Exception as exc:
            logger.error(
                "Failed to spill WAV buffer to temp file; "
                f"exception_type={_safe_exception_label(exc)}"
            )
            raise

    def _finalize_wav(self) -> bytes:
        """Finalize WAV output, writing headers and returning bytes."""
        if self._wav_file_path:
            return self._finalize_wav_from_file()

        buf = getattr(self, "output_buffer", None)
        if buf is None:
            logger.warning("StreamingAudioWriter finalize called with no WAV buffer present.")
            return b""

        buf.seek(0)
        pcm_bytes = buf.read()
        out = BytesIO()
        with wave.open(out, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_bytes)
        out.seek(0)
        data = out.read()
        logger.debug(
            f"StreamingAudioWriter finalize: format=wav (in-memory), "
            f"wav_bytes={len(data)}, pcm_bytes={len(pcm_bytes)}"
        )
        buf.close()
        self.output_buffer = None
        return data

    def _finalize_wav_from_file(self) -> bytes:
        """Finalize WAV output when PCM has been spooled to disk."""
        pcm_path = self._wav_file_path
        if not pcm_path:
            return b""

        temp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="tts_wav_final_", suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            with wave.open(temp_wav_path, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # int16
                wav_file.setframerate(self.sample_rate)
                with open(pcm_path, "rb") as pcm_file:
                    while True:
                        chunk = pcm_file.read(self._wav_chunk_size)
                        if not chunk:
                            break
                        wav_file.writeframes(chunk)

            with open(temp_wav_path, "rb") as final_file:
                data = final_file.read()

            logger.debug(
                f"StreamingAudioWriter finalize: format=wav (file-backed), "
                f"wav_bytes={len(data)}"
            )
            return data
        finally:
            for path in (pcm_path, temp_wav_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError as e:
                        logger.debug(
                            f"Error removing temp WAV file; exception_type={type(e).__name__}"
                        )
            self._wav_file_path = None


class AudioNormalizer:
    """Normalizes audio data for consistent output"""

    def normalize(self, audio_data: np.ndarray, target_dtype=np.int16) -> np.ndarray:
        """
        Normalize audio data to target dtype with proper scaling.

        Args:
            audio_data: Input audio as numpy array (typically float32)
            target_dtype: Target data type (default int16)

        Returns:
            Normalized audio array
        """
        if audio_data.dtype == target_dtype:
            return audio_data

        # Handle float to int16 conversion
        if audio_data.dtype in [np.float32, np.float64]:
            # Clip to [-1, 1] range
            audio_data = np.clip(audio_data, -1.0, 1.0)

            if target_dtype == np.int16:
                # Scale to int16 range
                return (audio_data * 32767).astype(np.int16)
            elif target_dtype == np.int32:
                # Scale to int32 range
                return (audio_data * 2147483647).astype(np.int32)

        # Handle int to float conversion
        elif target_dtype in [np.float32, np.float64]:
            if audio_data.dtype == np.int16:
                return audio_data.astype(target_dtype) / 32767.0
            elif audio_data.dtype == np.int32:
                return audio_data.astype(target_dtype) / 2147483647.0

        # Default: just cast
        return audio_data.astype(target_dtype)

#
# End of streaming_audio_writer.py
#######################################################################################################################
