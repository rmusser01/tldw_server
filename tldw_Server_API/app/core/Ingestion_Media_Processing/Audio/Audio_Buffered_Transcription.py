# Audio_Buffered_Transcription.py
#########################################
# Advanced Buffered/Chunked Transcription for Long Audio
# Based on NVIDIA's buffered inference implementation for RNNT models
# Supports multiple merge algorithms for optimal transcription quality
#
####################
# Function List
#
# 1. BufferedTranscriber - Base class for buffered transcription
# 2. MiddleMergeTranscriber - Middle token merge algorithm
# 3. LCSMergeTranscriber - Longest Common Subsequence merge
# 4. TDTMergeTranscriber - TDT-specific merge algorithm
# 5. transcribe_long_audio() - Main entry point for long audio transcription
#
####################

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from loguru import logger

logger = logger


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to float, returning ``default`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    """Safely coerce a value to int, returning ``default`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _token_from_mapping(token: Any) -> Optional[dict[str, Any]]:
    """Normalize a token-like mapping into the internal token schema."""
    if not isinstance(token, dict):
        return None
    text = str(token.get("text", "") or "")
    start = _safe_float(token.get("start"), 0.0)
    end = _safe_float(token.get("end"), start)
    if end < start:
        end = start
    duration = _safe_float(token.get("duration"), max(end - start, 0.0))
    token_id = _safe_int(token.get("id"), -1)
    if token_id < 0:
        token_id = _safe_int(token.get("token_id"), -1)
    return {
        "id": token_id,
        "text": text,
        "start": start,
        "end": end,
        "duration": duration,
        "confidence": _safe_float(token.get("confidence"), 1.0),
    }


def _extract_tokens_from_mlx_artifact(artifact: Any) -> list[dict[str, Any]]:
    """Extract normalized token dictionaries from a Parakeet MLX artifact payload."""
    if not isinstance(artifact, dict):
        return []
    tokens: list[dict[str, Any]] = []
    raw_tokens = artifact.get("tokens")
    if isinstance(raw_tokens, list):
        for token in raw_tokens:
            token_norm = _token_from_mapping(token)
            if token_norm is not None:
                tokens.append(token_norm)
    if tokens:
        return tokens

    raw_sentences = artifact.get("sentences")
    if isinstance(raw_sentences, list):
        for sentence in raw_sentences:
            if not isinstance(sentence, dict):
                continue
            for token in sentence.get("tokens", []):
                token_norm = _token_from_mapping(token)
                if token_norm is not None:
                    tokens.append(token_norm)
    return tokens


def _token_match(a: dict[str, Any], b: dict[str, Any], overlap_duration: float) -> bool:
    """Return ``True`` when two tokens are considered the same overlap token."""
    return (
        _safe_int(a.get("id"), -1) >= 0
        and _safe_int(a.get("id"), -1) == _safe_int(b.get("id"), -1)
        and abs(_safe_float(a.get("start")) - _safe_float(b.get("start"))) < (overlap_duration / 2.0)
    )


def _merge_tokens_midpoint(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge token lists by splitting at the midpoint of overlap boundaries."""
    if not existing:
        return incoming
    if not incoming:
        return existing
    existing_end = _safe_float(existing[-1].get("end"), 0.0)
    incoming_start = _safe_float(incoming[0].get("start"), 0.0)
    cutoff_time = (existing_end + incoming_start) / 2.0
    return [token for token in existing if _safe_float(token.get("end"), 0.0) <= cutoff_time] + [
        token for token in incoming if _safe_float(token.get("start"), 0.0) >= cutoff_time
    ]


def _merge_tokens_longest_contiguous(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    overlap_duration: float,
) -> list[dict[str, Any]]:
    """Merge token lists using the longest contiguous overlap run."""
    if not existing or not incoming:
        return existing if existing else incoming

    existing_end = _safe_float(existing[-1].get("end"), float("-inf"))
    incoming_start = _safe_float(incoming[0].get("start"), float("inf"))
    if existing_end <= incoming_start:
        return existing + incoming

    overlap_existing = [
        token
        for token in existing
        if _safe_float(token.get("end"), float("-inf")) > incoming_start - overlap_duration
    ]
    overlap_incoming = [
        token
        for token in incoming
        if _safe_float(token.get("start"), float("inf")) < existing_end + overlap_duration
    ]
    enough_pairs = len(overlap_existing) // 2
    if len(overlap_existing) < 2 or len(overlap_incoming) < 2:
        return _merge_tokens_midpoint(existing, incoming)

    best_contiguous: list[tuple[int, int]] = []
    for i in range(len(overlap_existing)):
        for j in range(len(overlap_incoming)):
            if _token_match(overlap_existing[i], overlap_incoming[j], overlap_duration):
                current: list[tuple[int, int]] = []
                k, incoming_idx = i, j
                while (
                    k < len(overlap_existing)
                    and incoming_idx < len(overlap_incoming)
                    and _token_match(overlap_existing[k], overlap_incoming[incoming_idx], overlap_duration)
                ):
                    current.append((k, incoming_idx))
                    k += 1
                    incoming_idx += 1
                if len(current) > len(best_contiguous):
                    best_contiguous = current

    if len(best_contiguous) < enough_pairs or not best_contiguous:
        return _merge_tokens_midpoint(existing, incoming)

    existing_overlap_start_idx = len(existing) - len(overlap_existing)
    overlap_indices_existing = [existing_overlap_start_idx + pair[0] for pair in best_contiguous]
    overlap_indices_incoming = [pair[1] for pair in best_contiguous]

    merged: list[dict[str, Any]] = []
    merged.extend(existing[: overlap_indices_existing[0]])
    for idx in range(len(best_contiguous)):
        idx_existing = overlap_indices_existing[idx]
        idx_incoming = overlap_indices_incoming[idx]
        merged.append(existing[idx_existing])
        if idx < len(best_contiguous) - 1:
            next_existing = overlap_indices_existing[idx + 1]
            next_incoming = overlap_indices_incoming[idx + 1]
            gap_existing = existing[idx_existing + 1 : next_existing]
            gap_incoming = incoming[idx_incoming + 1 : next_incoming]
            merged.extend(gap_incoming if len(gap_incoming) > len(gap_existing) else gap_existing)
    merged.extend(incoming[overlap_indices_incoming[-1] + 1 :])
    return merged


def _merge_tokens_longest_common_subsequence(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    overlap_duration: float,
) -> list[dict[str, Any]]:
    """Merge token lists by aligning the overlap region with LCS matching."""
    if not existing or not incoming:
        return existing if existing else incoming

    existing_end = _safe_float(existing[-1].get("end"), float("-inf"))
    incoming_start = _safe_float(incoming[0].get("start"), float("inf"))
    if existing_end <= incoming_start:
        return existing + incoming

    overlap_existing = [
        token
        for token in existing
        if _safe_float(token.get("end"), float("-inf")) > incoming_start - overlap_duration
    ]
    overlap_incoming = [
        token
        for token in incoming
        if _safe_float(token.get("start"), float("inf")) < existing_end + overlap_duration
    ]
    if len(overlap_existing) < 2 or len(overlap_incoming) < 2:
        return _merge_tokens_midpoint(existing, incoming)

    dp = [[0 for _ in range(len(overlap_incoming) + 1)] for _ in range(len(overlap_existing) + 1)]
    for i in range(1, len(overlap_existing) + 1):
        for j in range(1, len(overlap_incoming) + 1):
            if _token_match(overlap_existing[i - 1], overlap_incoming[j - 1], overlap_duration):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    pairs: list[tuple[int, int]] = []
    i = len(overlap_existing)
    j = len(overlap_incoming)
    while i > 0 and j > 0:
        if _token_match(overlap_existing[i - 1], overlap_incoming[j - 1], overlap_duration):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    if not pairs:
        return _merge_tokens_midpoint(existing, incoming)

    existing_overlap_start_idx = len(existing) - len(overlap_existing)
    overlap_indices_existing = [existing_overlap_start_idx + pair[0] for pair in pairs]
    overlap_indices_incoming = [pair[1] for pair in pairs]

    merged: list[dict[str, Any]] = []
    merged.extend(existing[: overlap_indices_existing[0]])
    for idx in range(len(pairs)):
        idx_existing = overlap_indices_existing[idx]
        idx_incoming = overlap_indices_incoming[idx]
        merged.append(existing[idx_existing])
        if idx < len(pairs) - 1:
            next_existing = overlap_indices_existing[idx + 1]
            next_incoming = overlap_indices_incoming[idx + 1]
            gap_existing = existing[idx_existing + 1 : next_existing]
            gap_incoming = incoming[idx_incoming + 1 : next_incoming]
            merged.extend(gap_incoming if len(gap_incoming) > len(gap_existing) else gap_existing)
    merged.extend(incoming[overlap_indices_incoming[-1] + 1 :])
    return merged


def _tokens_to_sentence_dicts(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group token timing into sentence-like segments with aggregate confidence."""
    if not tokens:
        return []
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for token in tokens:
        current.append(token)
        token_text = str(token.get("text", ""))
        is_sentence_boundary = any(symbol in token_text for symbol in ("!", "?", "。", "？", "！")) or "." in token_text
        if is_sentence_boundary:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    out: list[dict[str, Any]] = []
    for sentence_tokens in sentences:
        text = "".join(str(token.get("text", "")) for token in sentence_tokens).strip()
        if not text:
            continue
        start = _safe_float(sentence_tokens[0].get("start"))
        end = _safe_float(sentence_tokens[-1].get("end"), start)
        confidences = np.array([max(_safe_float(token.get("confidence"), 1.0), 1e-10) for token in sentence_tokens])
        confidence = float(np.exp(np.mean(np.log(confidences)))) if confidences.size else 1.0
        out.append(
            {
                "text": text,
                "start": start,
                "end": max(end, start),
                "duration": max(end - start, 0.0),
                "confidence": confidence,
                "tokens": sentence_tokens,
            }
        )
    return out


class MergeAlgorithm(Enum):
    """Supported merge algorithms for chunked transcription."""
    MIDDLE = "middle"      # Middle token merge (default for RNNT)
    LCS = "lcs"           # Longest Common Subsequence
    TDT = "tdt"           # TDT-specific algorithm
    OVERLAP = "overlap"    # Simple overlap removal
    SIMPLE = "simple"      # Simple concatenation


@dataclass
class BufferedTranscriptionConfig:
    """Configuration for buffered transcription."""
    chunk_duration: float = 2.0        # Chunk length in seconds
    total_buffer: float = 4.0          # Total buffer (chunk + padding) in seconds
    batch_size: int = 1                # Batch size for processing
    merge_algo: MergeAlgorithm = MergeAlgorithm.MIDDLE
    max_steps_per_timestep: int = 5    # For RNNT decoding
    stateful_decoding: bool = False    # Maintain state between chunks
    model_stride: float = 0.08         # Model's time stride in seconds (e.g., 80ms)
    preserve_timestamps: bool = False   # Preserve timing information
    device: str = 'cpu'                # Device for processing


class BufferedTranscriber:
    """
    Base class for buffered transcription of long audio files.

    Implements chunking and buffering strategy similar to NVIDIA's approach.
    """

    def __init__(self, config: BufferedTranscriptionConfig):
        """Initialize buffered transcriber with strict validation."""
        self.config = config
        self.transcription_history = []
        self.timestamp_history = []

        # Strict validation (do not silently correct invalid settings)
        if config.chunk_duration <= 0:
            raise ValueError("chunk_duration must be > 0 seconds")
        if config.total_buffer <= 0:
            raise ValueError("total_buffer must be > 0 seconds")
        if config.total_buffer < config.chunk_duration:
            raise ValueError(
                f"total_buffer ({config.total_buffer}s) must be >= chunk_duration ({config.chunk_duration}s)"
            )
        # Positive stride condition: total_buffer < 3 * chunk_duration
        if config.total_buffer >= 3.0 * config.chunk_duration:
            raise ValueError(
                "Invalid buffer settings: total_buffer must be < 3x chunk_duration to ensure positive stride. "
                f"Got chunk_duration={config.chunk_duration}s, total_buffer={config.total_buffer}s"
            )

        # Calculate chunk parameters
        self.chunk_samples_at_16k = int(config.chunk_duration * 16000)
        self.buffer_samples_at_16k = int(config.total_buffer * 16000)

        # Calculate tokens per chunk based on model stride
        self.tokens_per_chunk = math.ceil(config.chunk_duration / config.model_stride)

        # Calculate delays for merging
        left_padding = (config.total_buffer - config.chunk_duration) / 2
        self.mid_delay = math.ceil((config.chunk_duration + left_padding) / config.model_stride)

    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        transcribe_fn: Callable[[np.ndarray], str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Process long audio with chunking and merging.

        Args:
            audio_data: Complete audio data
            sample_rate: Sample rate of audio
            transcribe_fn: Function to transcribe a single chunk
            progress_callback: Optional progress callback

        Returns:
            Complete transcription
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_data = self._resample(audio_data, sample_rate, 16000)
            sample_rate = 16000

        # Precompute file-specific expected/allowed chunk counts
        padding_samples_pre = self.buffer_samples_at_16k - self.chunk_samples_at_16k
        stride_samples_pre = self.chunk_samples_at_16k - padding_samples_pre // 2
        if stride_samples_pre <= 0:
            raise ValueError(
                "Computed non-positive stride from provided settings. "
                f"chunk_samples={self.chunk_samples_at_16k}, buffer_samples={self.buffer_samples_at_16k}, "
                f"padding_samples={padding_samples_pre}. Ensure total_buffer < 3 * chunk_duration."
            )
        expected_chunks = max(1, math.ceil(len(audio_data) / stride_samples_pre))
        allowed_max_chunks = expected_chunks * 2  # 2x safety margin

        # Calculate chunks
        chunks = self._create_chunks(audio_data)
        num_chunks = len(chunks)
        # Enforce proportional upper bound
        if num_chunks > allowed_max_chunks:
            raise ValueError(
                f"Chunking produced {num_chunks} chunks which exceeds the file-specific maximum ({allowed_max_chunks}). "
                f"Expected around {expected_chunks} based on stride. Adjust chunk_duration/total_buffer."
            )

        logger.info(f"Processing {num_chunks} chunks with {self.config.merge_algo.value} algorithm")

        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            # Transcribe chunk
            result = transcribe_fn(chunk['audio'])

            # Store result with metadata
            chunk_results.append({
                'text': result,
                'start': chunk['start'],
                'end': chunk['end'],
                'overlap_start': chunk.get('overlap_start', 0),
                'overlap_end': chunk.get('overlap_end', 0)
            })

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, num_chunks)

        # Merge results
        merged_text = self._merge_results(chunk_results)

        return merged_text

    def _create_chunks(self, audio_data: np.ndarray) -> list[dict]:
        """
        Create overlapping chunks from audio data.

        Returns list of chunk dictionaries with audio and metadata.
        """
        chunks = []
        total_samples = len(audio_data)

        # Calculate stride (non-overlapping part)
        padding_samples = self.buffer_samples_at_16k - self.chunk_samples_at_16k
        stride_samples = self.chunk_samples_at_16k - padding_samples // 2
        if stride_samples <= 0:
            raise ValueError(
                "Computed non-positive stride from provided settings. "
                f"chunk_samples={self.chunk_samples_at_16k}, buffer_samples={self.buffer_samples_at_16k}, "
                f"padding_samples={padding_samples}. Ensure total_buffer < 3 * chunk_duration."
            )

        # Create chunks
        position = 0
        while position < total_samples:
            # Calculate chunk boundaries
            chunk_start = max(0, position - padding_samples // 2)
            chunk_end = min(total_samples, position + self.chunk_samples_at_16k + padding_samples // 2)

            # Extract chunk
            chunk_audio = audio_data[chunk_start:chunk_end]

            # Pad if needed
            if len(chunk_audio) < self.buffer_samples_at_16k:
                chunk_audio = np.pad(
                    chunk_audio,
                    (0, self.buffer_samples_at_16k - len(chunk_audio)),
                    mode='constant'
                )

            chunks.append({
                'audio': chunk_audio,
                'start': chunk_start / 16000,  # Convert to seconds
                'end': chunk_end / 16000,
                'overlap_start': (position - chunk_start) / 16000 if position > chunk_start else 0,
                'overlap_end': (chunk_end - (position + self.chunk_samples_at_16k)) / 16000
                             if chunk_end > position + self.chunk_samples_at_16k else 0
            })

            # Move to next position
            position += stride_samples

        return chunks

    def _merge_results(self, chunk_results: list[dict]) -> str:
        """
        Merge chunk results based on configured algorithm.

        Override this in subclasses for specific merge strategies.
        """
        if self.config.merge_algo == MergeAlgorithm.SIMPLE:
            # Simple concatenation
            return ' '.join(r['text'] for r in chunk_results if r['text'])
        else:
            # Default to middle merge
            return self._middle_merge(chunk_results)

    def _middle_merge(self, chunk_results: list[dict]) -> str:
        """
        Implement middle token merge algorithm.

        Removes overlapping portions from chunk boundaries.
        """
        if not chunk_results:
            return ""

        merged_texts = []

        for i, result in enumerate(chunk_results):
            text = result['text']
            if not text:
                continue

            if i > 0 and result['overlap_start'] > 0:
                # Remove overlapping start portion
                words = text.split()
                overlap_ratio = result['overlap_start'] / self.config.chunk_duration
                words_to_skip = int(len(words) * overlap_ratio * 0.5)  # Take middle
                text = ' '.join(words[words_to_skip:]) if words_to_skip < len(words) else text

            if i < len(chunk_results) - 1 and result['overlap_end'] > 0:
                # Remove overlapping end portion
                words = text.split()
                overlap_ratio = result['overlap_end'] / self.config.chunk_duration
                words_to_keep = len(words) - int(len(words) * overlap_ratio * 0.5)
                text = ' '.join(words[:words_to_keep]) if words_to_keep > 0 else text

            merged_texts.append(text)

        return ' '.join(merged_texts)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Return audio AT ``target_sr``, resampling if necessary.

        Post-condition: the returned array is at ``target_sr``. Both callers relabel
        ``sample_rate = target_sr`` immediately after calling this, so returning the
        input unchanged silently mislabels the stream - a 48 kHz buffer declared as
        16 kHz is fed to the model at 3x speed with every timestamp 3x short.

        Falls back to linear interpolation when librosa is absent, matching
        Audio_Transcription_Lib._resample_audio_if_needed, rather than lying.
        """
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.debug("librosa not available; using linear-interpolation resample")
            ratio = float(target_sr) / float(orig_sr)
            new_len = max(1, int(round(len(audio) * ratio)))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


class LCSMergeTranscriber(BufferedTranscriber):
    """
    Transcriber using Longest Common Subsequence merge algorithm.

    Better handling of overlapping regions by finding common sequences.
    """

    def _merge_results(self, chunk_results: list[dict]) -> str:
        """Merge using LCS algorithm."""
        if not chunk_results:
            return ""

        if len(chunk_results) == 1:
            return chunk_results[0]['text']

        # Start with first chunk
        merged = chunk_results[0]['text']

        for i in range(1, len(chunk_results)):
            current = chunk_results[i]['text']
            if not current:
                continue

            # Find LCS between end of merged and start of current
            merged = self._merge_with_lcs(merged, current)

        return merged

    def _merge_with_lcs(self, text1: str, text2: str) -> str:
        """
        Merge two texts using LCS to find overlap.
        """
        words1 = text1.split()
        words2 = text2.split()

        if not words1:
            return text2
        if not words2:
            return text1

        # First try a quick contiguous overlap check on boundaries
        max_overlap = min(len(words1), len(words2), 10)
        for overlap_size in range(max_overlap, 0, -1):
            if words1[-overlap_size:] == words2[:overlap_size]:
                return ' '.join(words1 + words2[overlap_size:])

        # Fallback to true LCS over boundary windows to remove duplicated phrases
        w1 = words1[-20:]  # last window of first text
        w2 = words2[:20]   # first window of second text
        m, n = len(w1), len(w2)
        # Build DP table
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if w1[i] == w2[j]:
                    L[i][j] = 1 + L[i + 1][j + 1]
                else:
                    L[i][j] = max(L[i + 1][j], L[i][j + 1])
        lcs_len = L[0][0]

        # If LCS is reasonably informative (>=2 words), skip up to last match in w2
        if lcs_len >= 2:
            # Reconstruct to get last matched index in w2
            i = j = 0
            last_j = -1
            while i < m and j < n:
                if w1[i] == w2[j]:
                    last_j = j
                    i += 1
                    j += 1
                elif L[i + 1][j] >= L[i][j + 1]:
                    i += 1
                else:
                    j += 1
            # Merge skipping the overlapped prefix in words2
            skip = (last_j + 1) if last_j >= 0 else 0
            return ' '.join(words1 + words2[skip:])

        # No overlap found, simple concatenation
        return text1 + ' ' + text2

    def _lcs_length(self, X: list[str], Y: list[str]) -> int:
        """
        Calculate length of longest common subsequence.
        """
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])

        return L[m][n]


def transcribe_long_audio(
    audio_path: Union[str, Path, np.ndarray],
    model_name: str = 'parakeet',
    variant: str = 'mlx',
    chunk_duration: float = 30.0,
    total_buffer: Optional[float] = None,
    merge_algo: str = 'middle',
    device: str = 'cpu',
    progress_callback: Optional[Callable[[int, int], None]] = None,
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
    Transcribe long audio files using buffered/chunked processing.

    Args:
        audio_path: Path to audio file or numpy array
        model_name: Model to use ('parakeet', 'canary')
        variant: Model variant ('standard', 'onnx', 'mlx')
        chunk_duration: Duration of each chunk in seconds
        total_buffer: Total buffer size (chunk + padding)
        merge_algo: Merge algorithm ('middle', 'lcs', 'overlap', 'simple')
        device: Device for processing
        progress_callback: Optional callback for progress

    Returns:
        Complete transcription
    """
    import soundfile as sf

    # Load audio
    if isinstance(audio_path, (str, Path)):
        audio_data, sample_rate = sf.read(str(audio_path))
    else:
        audio_data = audio_path
        sample_rate = 16000  # Assume 16kHz

    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Determine default total_buffer if not provided: 1.5x chunk (strictly < 3x)
    if total_buffer is None:
        total_buffer = max(chunk_duration, min(chunk_duration * 1.5, chunk_duration * 2.9))

    # Create config
    config = BufferedTranscriptionConfig(
        chunk_duration=chunk_duration,
        total_buffer=total_buffer,
        merge_algo=MergeAlgorithm(merge_algo),
        device=device
    )

    # Select transcriber based on merge algorithm
    if config.merge_algo == MergeAlgorithm.LCS:
        transcriber = LCSMergeTranscriber(config)
    else:
        transcriber = BufferedTranscriber(config)

    # Create transcribe function
    def transcribe_chunk(chunk_audio: np.ndarray) -> Union[str, dict[str, Any]]:
        # Import appropriate transcription function
        if variant == 'mlx':
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx,
            )
            return transcribe_with_parakeet_mlx(
                chunk_audio,
                sample_rate=16000,
                return_structured=return_structured,
                model_path=model_path,
                cache_dir=cache_dir,
                decoding_mode=decoding_mode,
                beam_size=beam_size,
                length_penalty=length_penalty,
                patience=patience,
                duration_reward=duration_reward,
                sentence_max_words=sentence_max_words,
                sentence_silence_gap=sentence_silence_gap,
                sentence_max_duration=sentence_max_duration,
            )
        elif variant == 'onnx':
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                transcribe_with_parakeet_onnx,
            )
            return transcribe_with_parakeet_onnx(chunk_audio, sample_rate=16000, device=device)
        else:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_nemo,
            )
            return transcribe_with_nemo(
                chunk_audio,
                sample_rate=16000,
                model=model_name,
                variant=variant
            )

    if variant == "mlx" and return_structured:
        if sample_rate != 16000:
            audio_data = transcriber._resample(audio_data, sample_rate, 16000)
            sample_rate = 16000

        chunks = transcriber._create_chunks(audio_data)
        total_chunks = len(chunks)
        merged_tokens: list[dict[str, Any]] = []
        fallback_text_chunks: list[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_result = transcribe_chunk(chunk["audio"])
            if progress_callback:
                progress_callback(idx + 1, total_chunks)

            if isinstance(chunk_result, dict):
                tokens = _extract_tokens_from_mlx_artifact(chunk_result)
                if tokens:
                    chunk_start = _safe_float(chunk.get("start"), 0.0)
                    for token in tokens:
                        token["start"] = _safe_float(token.get("start"), 0.0) + chunk_start
                        token["end"] = _safe_float(token.get("end"), token["start"]) + chunk_start
                        if token["end"] < token["start"]:
                            token["end"] = token["start"]
                        token["duration"] = max(token["end"] - token["start"], 0.0)

                    if not merged_tokens:
                        merged_tokens = tokens
                    else:
                        overlap_for_merge = max(
                            _safe_float(chunk.get("overlap_start"), 0.0),
                            _safe_float(chunk.get("overlap_end"), 0.0),
                            0.0,
                        )
                        if config.merge_algo == MergeAlgorithm.LCS:
                            merged_tokens = _merge_tokens_longest_common_subsequence(
                                merged_tokens,
                                tokens,
                                overlap_duration=overlap_for_merge,
                            )
                        elif config.merge_algo == MergeAlgorithm.SIMPLE:
                            merged_tokens.extend(tokens)
                        else:
                            merged_tokens = _merge_tokens_longest_contiguous(
                                merged_tokens,
                                tokens,
                                overlap_duration=overlap_for_merge,
                            )
                else:
                    chunk_text = str(chunk_result.get("text", "")).strip()
                    if chunk_text:
                        fallback_text_chunks.append(chunk_text)
            else:
                chunk_text = str(chunk_result).strip()
                if chunk_text:
                    fallback_text_chunks.append(chunk_text)

        if merged_tokens:
            sentences = _tokens_to_sentence_dicts(merged_tokens)
            text = "".join(str(token.get("text", "")) for token in merged_tokens).strip()
            if not text and sentences:
                text = " ".join(str(sentence.get("text", "")) for sentence in sentences).strip()
            return {
                "text": text,
                "sentences": sentences,
                "tokens": merged_tokens,
            }

        return {
            "text": " ".join(fallback_text_chunks).strip(),
            "sentences": [],
            "tokens": [],
        }

    # Process audio with legacy text merge
    return transcriber.process_audio(
        audio_data,
        sample_rate,
        transcribe_chunk,
        progress_callback
    )


#######################################################################################################################
# End of Audio_Buffered_Transcription.py
#######################################################################################################################
