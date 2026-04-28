# audio_converter.py
# Description: Audio conversion and processing utilities for voice samples
#
# Imports
import asyncio
import contextlib
import os
import shutil
# Fixed ffmpeg/ffprobe probes resolve executables and do not use shell.
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
from loguru import logger

#
# Local Imports
from .tts_exceptions import TTSError

_AUDIO_CLEANUP_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
)

_AUDIO_CONVERSION_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    AttributeError,
    KeyError,
    ImportError,
    subprocess.SubprocessError,
)

#
#######################################################################################################################
#
# Audio Conversion Service

class AudioConversionError(TTSError):
    """Error during audio conversion"""
    pass


class AudioConverter:
    """Audio conversion and processing utilities"""

    # Common audio formats and their codecs
    AUDIO_CODECS = {
        'wav': 'pcm_s16le',
        'mp3': 'libmp3lame',
        'flac': 'flac',
        'ogg': 'libvorbis',
        'opus': 'libopus',
        'm4a': 'aac',
        'm4b': 'aac',
    }

    @staticmethod
    def _write_concat_list(input_paths: list[Path]) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with tmp:
            for path in input_paths:
                path_str = str(path).replace("'", "'\\''")
                tmp.write(f"file '{path_str}'\n".encode())
        return tmp.name

    @staticmethod
    async def concat_audio_files(
        input_paths: list[Path],
        output_path: Path,
        target_format: str,
        **kwargs,
    ) -> bool:
        if not input_paths:
            logger.error("No input files provided for concatenation")
            return False
        codec = AudioConverter.AUDIO_CODECS.get(target_format.lower())
        if not codec:
            logger.error(f"Unsupported concat format: {target_format}")
            return False
        output_path = output_path.with_suffix(f".{target_format.lower()}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_path = AudioConverter._write_concat_list(input_paths)

        try:
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path]
            if 'sample_rate' in kwargs:
                cmd.extend(['-ar', str(kwargs['sample_rate'])])
            if 'channels' in kwargs:
                cmd.extend(['-ac', str(kwargs['channels'])])
            if 'bitrate' in kwargs:
                cmd.extend(['-b:a', str(kwargs['bitrate'])])
            cmd.extend(['-c:a', codec, str(output_path)])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Concat failed: {stderr.decode()}")
                return False
            logger.info(f"Concatenated {len(input_paths)} files into {output_path.name}")
            return True
        finally:
            with contextlib.suppress(_AUDIO_CLEANUP_EXCEPTIONS):
                os.unlink(list_path)

    @staticmethod
    async def package_m4b_with_chapters(
        input_paths: list[Path],
        output_path: Path,
        chapter_titles: list[str],
        *,
        metadata: Optional[dict[str, str]] = None,
    ) -> bool:
        if not input_paths:
            logger.error("No input files provided for M4B packaging")
            return False
        output_path = output_path.with_suffix(".m4b")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_path = AudioConverter._write_concat_list(input_paths)
        meta_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ffmeta")
        try:
            durations_ms: list[int] = []
            for path in input_paths:
                duration = await AudioConverter.get_duration(path)
                durations_ms.append(max(1, int(round(duration * 1000))))

            meta_text = AudioConverter._build_ffmetadata(chapter_titles, durations_ms, metadata)
            with meta_file:
                meta_file.write(meta_text.encode())

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-i', meta_file.name,
                '-map_metadata', '1',
                '-map_chapters', '1',
                '-c:a', 'aac',
                str(output_path),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"M4B packaging failed: {stderr.decode()}")
                return False
            logger.info(f"Packaged M4B with {len(input_paths)} chapters: {output_path.name}")
            return True
        finally:
            with contextlib.suppress(_AUDIO_CLEANUP_EXCEPTIONS):
                os.unlink(list_path)
            with contextlib.suppress(_AUDIO_CLEANUP_EXCEPTIONS):
                os.unlink(meta_file.name)

    @staticmethod
    def _build_ffmetadata(
        chapter_titles: list[str],
        chapter_durations_ms: list[int],
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        lines = [";FFMETADATA1"]
        if metadata:
            title = metadata.get("title")
            artist = metadata.get("artist")
            if title:
                lines.append(f"title={title}")
            if artist:
                lines.append(f"artist={artist}")

        current_ms = 0
        for idx, duration_ms in enumerate(chapter_durations_ms):
            duration_ms = max(1, int(duration_ms))
            start = current_ms
            end = current_ms + duration_ms
            current_ms = end
            chapter_title = chapter_titles[idx] if idx < len(chapter_titles) else f"Chapter {idx + 1}"
            lines.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start}",
                    f"END={end}",
                    f"title={chapter_title}",
                ]
            )

        return "\n".join(lines) + "\n"

    @staticmethod
    async def convert_to_wav(
        input_path: Path,
        output_path: Path,
        sample_rate: int = 22050,
        channels: int = 1,
        bit_depth: int = 16
    ) -> bool:
        """
        Convert audio file to WAV format with specified parameters.

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            bit_depth: Bit depth (16 or 24)

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine codec/sample format based on requested bit depth
            if bit_depth == 16:
                codec = 'pcm_s16le'
                sample_fmt = 's16'
            elif bit_depth == 24:
                codec = 'pcm_s24le'
                sample_fmt = 's24'
            else:
                raise AudioConversionError(f"Unsupported bit depth: {bit_depth}. Supported values: 16, 24.")

            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(input_path),  # Input file
                '-ar', str(sample_rate),  # Sample rate
                '-ac', str(channels),  # Audio channels
                '-sample_fmt', sample_fmt,  # Bit depth
                '-c:a', codec,  # PCM codec for WAV
                str(output_path)  # Output file
            ]

            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
                return False

            logger.info(f"Converted {input_path.name} to WAV ({sample_rate}Hz, {channels}ch, {bit_depth}bit)")
            return True

        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            raise AudioConversionError("FFmpeg is required for audio conversion") from None
        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Audio conversion error")
            return False

    @staticmethod
    async def convert_format(
        input_path: Path,
        output_path: Path,
        target_format: str,
        **kwargs
    ) -> bool:
        """
        Convert audio between formats.

        Args:
            input_path: Path to input audio file
            output_path: Path for output file
            target_format: Target format (wav, mp3, flac, etc.)
            **kwargs: Additional ffmpeg parameters

        Returns:
            True if successful
        """
        try:
            # Get codec for target format
            codec = AudioConverter.AUDIO_CODECS.get(target_format.lower())
            if not codec:
                logger.error(f"Unsupported format: {target_format}")
                return False

            # Ensure output has correct extension
            output_path = output_path.with_suffix(f".{target_format.lower()}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build command
            cmd = ['ffmpeg', '-y', '-i', str(input_path)]

            # Add optional parameters
            if 'sample_rate' in kwargs:
                cmd.extend(['-ar', str(kwargs['sample_rate'])])
            if 'channels' in kwargs:
                cmd.extend(['-ac', str(kwargs['channels'])])
            if 'bitrate' in kwargs:
                cmd.extend(['-b:a', str(kwargs['bitrate'])])

            # Add codec and output
            cmd.extend(['-c:a', codec, str(output_path)])

            # Run conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Format conversion failed: {stderr.decode()}")
                return False

            logger.info(f"Converted {input_path.name} to {target_format}")
            return True

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Format conversion error")
            return False

    @staticmethod
    async def validate_duration(
        file_path: Path,
        min_seconds: float = 0,
        max_seconds: float = float('inf')
    ) -> tuple[bool, float]:
        """
        Check if audio duration is within specified range.

        Args:
            file_path: Path to audio file
            min_seconds: Minimum duration in seconds
            max_seconds: Maximum duration in seconds

        Returns:
            Tuple of (is_valid, actual_duration)
        """
        try:
            duration = await AudioConverter.get_duration(file_path)

            is_valid = min_seconds <= duration <= max_seconds

            if not is_valid:
                if duration < min_seconds:
                    logger.warning(f"Audio duration {duration:.1f}s is below minimum {min_seconds}s")
                else:
                    logger.warning(f"Audio duration {duration:.1f}s exceeds maximum {max_seconds}s")

            return is_valid, duration

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Duration validation error")
            return False, 0.0

    @staticmethod
    async def get_duration(file_path: Path) -> float:
        """
        Get audio file duration in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                return float(stdout.decode().strip())
            else:
                logger.error(f"Could not get duration: {stderr.decode()}")
                return 0.0

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Error getting duration")
            return 0.0

    @staticmethod
    async def get_audio_info(file_path: Path) -> dict[str, Any]:
        """
        Get detailed audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio properties
        """
        info = {
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'codec': '',
            'bitrate': 0,
            'format': file_path.suffix[1:] if file_path.suffix else ''
        }

        try:
            # Get duration
            info['duration'] = await AudioConverter.get_duration(file_path)

            # Get detailed stream info
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate',
                '-of', 'json',
                str(file_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                import json
                data = json.loads(stdout.decode())
                if data.get('streams'):
                    stream = data['streams'][0]
                    info['codec'] = stream.get('codec_name', '')
                    info['sample_rate'] = int(stream.get('sample_rate', 0))
                    info['channels'] = int(stream.get('channels', 0))
                    info['bitrate'] = int(stream.get('bit_rate', 0))

            return info

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Error getting audio info")
            return info

    @staticmethod
    async def normalize_audio(
        input_path: Path,
        output_path: Path,
        target_level: float = -23.0
    ) -> bool:
        """
        Normalize audio loudness using EBU R128 standard.

        Args:
            input_path: Path to input audio
            output_path: Path for normalized output
            target_level: Target loudness in LUFS (default -23)

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # First pass: analyze loudness
            cmd_analyze = [
                'ffmpeg', '-i', str(input_path),
                '-af', 'loudnorm=print_format=json',
                '-f', 'null', '-'
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd_analyze,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Parse loudness info from stderr (ffmpeg outputs to stderr)
            import json
            import re

            # Extract JSON from stderr
            json_match = re.search(r'\{[^}]+\}', stderr.decode())
            if not json_match:
                logger.error("Could not analyze audio loudness")
                return False

            json.loads(json_match.group())

            # Second pass: apply normalization
            cmd_normalize = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd_normalize,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Normalization failed: {stderr.decode()}")
                return False

            logger.info(f"Normalized audio to {target_level} LUFS")
            return True

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Audio normalization error")
            return False

    @staticmethod
    def _build_atempo_filter(speed_ratio: float) -> str:
        if speed_ratio <= 0:
            raise ValueError("speed_ratio must be positive")
        factors: list[float] = []
        remaining = speed_ratio
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        factors.append(remaining)
        return ",".join(f"atempo={factor:.6g}" for factor in factors)

    @staticmethod
    async def time_stretch(
        input_path: Path,
        output_path: Path,
        speed_ratio: float,
    ) -> bool:
        """
        Time-stretch audio using ffmpeg atempo filter.

        Args:
            input_path: Path to input audio
            output_path: Path for stretched output
            speed_ratio: Playback speed ratio (e.g., 1.05 to speed up)

        Returns:
            True if successful
        """
        if speed_ratio <= 0:
            logger.error("Time-stretch speed_ratio must be positive")
            return False
        if speed_ratio == 1.0:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(input_path.read_bytes())
                return True
            except _AUDIO_CONVERSION_EXCEPTIONS:
                logger.error("Time-stretch noop copy failed")
                return False
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            filter_spec = AudioConverter._build_atempo_filter(speed_ratio)
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-filter:a', filter_spec,
                str(output_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Time-stretch failed: {stderr.decode()}")
                return False
            logger.info(f"Time-stretch applied (ratio: {speed_ratio})")
            return True
        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Time-stretch error")
            return False

    @staticmethod
    async def trim_silence(
        input_path: Path,
        output_path: Path,
        threshold: float = -40.0,
        duration: float = 0.5
    ) -> bool:
        """
        Trim silence from beginning and end of audio.

        Args:
            input_path: Path to input audio
            output_path: Path for trimmed output
            threshold: Silence threshold in dB
            duration: Minimum silence duration to trim (seconds)

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use silenceremove filter
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f"silenceremove=start_periods=1:start_duration={duration}:start_threshold={threshold}dB:"
                       f"stop_periods=1:stop_duration={duration}:stop_threshold={threshold}dB",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Silence trimming failed: {stderr.decode()}")
                return False

            logger.info(f"Trimmed silence from audio (threshold: {threshold}dB)")
            return True

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Silence trimming error")
            return False

    @staticmethod
    async def extract_segment(
        input_path: Path,
        output_path: Path,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Extract a segment from audio file.

        Args:
            input_path: Path to input audio
            output_path: Path for segment output
            start_time: Start time in seconds
            duration: Segment duration in seconds

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),  # Seek to start
                '-i', str(input_path),
                '-t', str(duration),  # Duration
                '-c', 'copy',  # Copy codec (no re-encoding)
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Segment extraction failed: {stderr.decode()}")
                return False

            logger.info(f"Extracted {duration}s segment starting at {start_time}s")
            return True

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Segment extraction error")
            return False

    @staticmethod
    async def resample_audio(
        input_path: Path,
        output_path: Path,
        target_sample_rate: int
    ) -> bool:
        """
        Resample audio to target sample rate.

        Args:
            input_path: Path to input audio
            output_path: Path for resampled output
            target_sample_rate: Target sample rate in Hz

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-ar', str(target_sample_rate),
                '-c:a', 'pcm_s16le' if output_path.suffix == '.wav' else 'copy',
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Resampling failed: {stderr.decode()}")
                return False

            logger.info(f"Resampled audio to {target_sample_rate}Hz")
            return True

        except _AUDIO_CONVERSION_EXCEPTIONS:
            logger.error("Resampling error")
            return False

    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if FFmpeg is installed and available."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return False

        try:
            # Fixed executable and arguments, no shell.
            result = subprocess.run(  # nosec B603
                [ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def check_ffprobe_installed() -> bool:
        """Check if FFprobe is installed and available."""
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            return False

        try:
            # Fixed executable and arguments, no shell.
            result = subprocess.run(  # nosec B603
                [ffprobe_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# Utility function for quick checks
async def validate_audio_tools() -> tuple[bool, str]:
    """
    Validate that required audio processing tools are installed.

    Returns:
        Tuple of (all_tools_available, error_message)
    """
    errors = []

    if not AudioConverter.check_ffmpeg_installed():
        errors.append("FFmpeg is not installed or not in PATH")

    if not AudioConverter.check_ffprobe_installed():
        errors.append("FFprobe is not installed or not in PATH")

    if errors:
        return False, "; ".join(errors)

    return True, ""
