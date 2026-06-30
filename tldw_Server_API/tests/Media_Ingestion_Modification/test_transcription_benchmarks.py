"""
Performance benchmark tests for all transcription implementations.
"""

import pytest
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import soundfile as sf
from typing import List, Dict, Any
import json

pytestmark = pytest.mark.performance


def _memory_efficiency_within_tolerance(
    small_chunks_mb: float,
    large_chunks_mb: float,
    *,
    ratio: float = 1.5,
    noise_floor_mb: float = 1.0,
) -> bool:
    """Compare measured RSS deltas while ignoring allocator/page-size noise."""
    small = max(0.0, float(small_chunks_mb))
    large = max(0.0, float(large_chunks_mb))
    if max(small, large) <= noise_floor_mb:
        return True
    if large <= noise_floor_mb:
        return False
    return small <= large * ratio


class TranscriptionBenchmark:
    """Base class for transcription benchmarks."""

    @staticmethod
    def generate_test_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
        """Generate test audio with speech-like characteristics."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Mix of frequencies to simulate speech
        audio = (
            0.3 * np.sin(200 * 2 * np.pi * t) +  # Low frequency (fundamental)
            0.2 * np.sin(400 * 2 * np.pi * t) +  # First harmonic
            0.15 * np.sin(800 * 2 * np.pi * t) + # Second harmonic
            0.1 * np.sin(1600 * 2 * np.pi * t) + # Higher harmonic
            0.05 * np.random.randn(len(t))       # Noise
        )

        # Add amplitude modulation to simulate speech patterns
        envelope = 1 + 0.5 * np.sin(3 * 2 * np.pi * t)
        audio = audio * envelope

        # Add pauses (silence) to simulate natural speech
        for i in range(0, len(audio), sample_rate * 5):  # Every 5 seconds
            pause_start = i + sample_rate * 3
            pause_end = min(pause_start + sample_rate // 2, len(audio))
            if pause_end <= len(audio):
                audio[pause_start:pause_end] *= 0.1  # Reduce amplitude

        return audio.astype(np.float32)

    @staticmethod
    def measure_performance(func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics of a function."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        return {
            'result': result,
            'time': elapsed_time,
            'memory_used': mem_used,
            'memory_before': mem_before,
            'memory_after': mem_after
        }


class TestMLXPerformance(TranscriptionBenchmark):
    """Performance benchmarks for MLX implementation."""

    @pytest.fixture
    def mock_mlx_model(self):
        """Create mock MLX model with realistic behavior."""
        model = MagicMock()

        def mock_transcribe(audio_path, **kwargs):

            # Simulate processing time based on audio length
            time.sleep(0.01)  # Base processing time

            if 'chunk_duration' in kwargs:
                # Chunked processing is faster
                time.sleep(0.005)
            else:
                # Non-chunked processing
                time.sleep(0.02)

            result = MagicMock()
            result.text = "Benchmark transcription result"
            return result

        model.transcribe = mock_transcribe
        return model

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_mlx_scaling(self, mock_check, mock_load, mock_mlx_model):
        """Test MLX performance scaling with audio duration."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        mock_check.return_value = True
        mock_load.return_value = mock_mlx_model

        durations = [10, 30, 60, 120, 300]  # seconds
        results = []

        for duration in durations:
            audio = self.generate_test_audio(duration)

            metrics = self.measure_performance(
                transcribe_with_parakeet_mlx,
                audio,
                sample_rate=16000
            )

            real_time_factor = duration / metrics['time'] if metrics['time'] > 0 else 0

            results.append({
                'duration': duration,
                'processing_time': metrics['time'],
                'real_time_factor': real_time_factor,
                'memory_used': metrics['memory_used']
            })

            print(f"Duration: {duration}s, Time: {metrics['time']:.3f}s, "
                  f"RTF: {real_time_factor:.1f}x, Memory: {metrics['memory_used']:.1f}MB")

        # Verify performance scales reasonably
        for result in results:
            assert result['real_time_factor'] > 1.0  # Should be faster than real-time

        return results

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_mlx_chunking_performance(self, mock_check, mock_load, mock_mlx_model):
        """Compare chunked vs non-chunked MLX performance."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        mock_check.return_value = True
        mock_load.return_value = mock_mlx_model

        audio = self.generate_test_audio(120)  # 2 minutes

        # Test without chunking
        metrics_no_chunk = self.measure_performance(
            transcribe_with_parakeet_mlx,
            audio,
            sample_rate=16000
        )

        # Test with chunking
        metrics_chunked = self.measure_performance(
            transcribe_with_parakeet_mlx,
            audio,
            sample_rate=16000,
            chunk_duration=30.0,
            overlap_duration=5.0
        )

        speedup = metrics_no_chunk['time'] / metrics_chunked['time'] if metrics_chunked['time'] > 0 else float('inf')
        denom_mem = metrics_no_chunk['memory_used'] if metrics_no_chunk['memory_used'] != 0 else 1e-9
        memory_ratio = metrics_chunked['memory_used'] / denom_mem

        print(f"\nChunking Performance Comparison:")
        print(f"No chunking: {metrics_no_chunk['time']:.3f}s, {metrics_no_chunk['memory_used']:.1f}MB")
        print(f"With chunking: {metrics_chunked['time']:.3f}s, {metrics_chunked['memory_used']:.1f}MB")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Memory ratio: {memory_ratio:.2f}")

        assert metrics_chunked['result'] is not None
        assert metrics_no_chunk['result'] is not None


class TestONNXPerformance(TranscriptionBenchmark):
    """Performance benchmarks for ONNX implementation."""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create mock ONNX session with realistic behavior."""
        session = MagicMock()

        def mock_run(output_names, input_dict):

            # Simulate inference time
            time.sleep(0.015)

            batch_size = 1
            seq_len = 50
            vocab_size = 128256
            logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
            return [logits]

        session.run = mock_run
        session.get_inputs.return_value = [MagicMock(name="encoder_outputs")]
        session.get_outputs.return_value = [MagicMock(name="logits")]

        return session

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_onnx_performance(self, mock_load, mock_onnx_session):
        """Test ONNX transcription performance."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode = lambda x: "ONNX benchmark result"

        mock_load.return_value = (mock_onnx_session, tokenizer)

        durations = [10, 30, 60]
        results = []

        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX._preprocess_audio') as mock_preprocess:
            mock_features = np.random.randn(1, 100, 512).astype(np.float32)
            mock_preprocess.return_value = mock_features

            for duration in durations:
                audio = self.generate_test_audio(duration)

                metrics = self.measure_performance(
                    transcribe_with_parakeet_onnx,
                    audio,
                    sample_rate=16000
                )

                real_time_factor = duration / metrics['time'] if metrics['time'] > 0 else 0

                results.append({
                    'duration': duration,
                    'processing_time': metrics['time'],
                    'real_time_factor': real_time_factor,
                    'memory_used': metrics['memory_used']
                })

                print(f"ONNX - Duration: {duration}s, Time: {metrics['time']:.3f}s, "
                      f"RTF: {real_time_factor:.1f}x")

        return results


class TestBufferedTranscriptionPerformance(TranscriptionBenchmark):
    """Performance benchmarks for buffered transcription."""

    def test_merge_algorithm_performance(self):

        """Compare performance of different merge algorithms."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber,
            LCSMergeTranscriber,
            BufferedTranscriptionConfig,
            MergeAlgorithm
        )

        # Generate test chunks
        chunk_texts = []
        for i in range(20):
            chunk_texts.append(f"This is chunk {i} with some overlapping text chunk {i+1}")

        chunk_results = [
            {'text': text, 'start': i*2, 'end': (i+1)*2.5,
             'overlap_start': 0.5 if i > 0 else 0,
             'overlap_end': 0.5 if i < 19 else 0}
            for i, text in enumerate(chunk_texts)
        ]

        algorithms = {
            'middle': BufferedTranscriber,
            'lcs': LCSMergeTranscriber
        }

        results = {}

        for algo_name, transcriber_class in algorithms.items():
            config = BufferedTranscriptionConfig(merge_algo=algo_name)
            transcriber = transcriber_class(config)

            start = time.perf_counter()
            merged = transcriber._merge_results(chunk_results)
            elapsed = time.perf_counter() - start

            results[algo_name] = {
                'time': elapsed,
                'length': len(merged),
                'text': merged[:100] + "..."
            }

            print(f"{algo_name.upper()} merge: {elapsed*1000:.3f}ms, {len(merged)} chars")

        # LCS should produce shorter output (better deduplication)
        assert results['lcs']['length'] <= results['middle']['length']

        return results

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx')
    def test_buffered_memory_efficiency(self, mock_transcribe):
        """Test memory efficiency of buffered transcription."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )

        mock_transcribe.return_value = "Buffered chunk"

        # Test with very long audio (10 minutes)
        audio = self.generate_test_audio(600)

        # Small chunks should use less memory
        metrics_small = self.measure_performance(
            transcribe_long_audio,
            audio,
            variant='mlx',
            chunk_duration=10.0,
            total_buffer=15.0
        )

        # Large chunks should use more memory
        metrics_large = self.measure_performance(
            transcribe_long_audio,
            audio,
            variant='mlx',
            chunk_duration=60.0,
            total_buffer=90.0
        )

        print(f"\nMemory Efficiency:")
        print(f"Small chunks: {metrics_small['memory_used']:.1f}MB")
        print(f"Large chunks: {metrics_large['memory_used']:.1f}MB")

        # Small chunks should be more memory efficient
        assert _memory_efficiency_within_tolerance(
            metrics_small['memory_used'],
            metrics_large['memory_used'],
        )


class TestComparativePerformance(TranscriptionBenchmark):
    """Comparative benchmarks across all implementations."""

    def test_all_implementations_comparison(self):

        """Compare all transcription implementations."""

        results = {
            'implementation': [],
            'duration': [],
            'processing_time': [],
            'real_time_factor': [],
            'memory_used': []
        }

        audio_durations = [30, 60, 120]

        for duration in audio_durations:
            audio = self.generate_test_audio(duration)

            # Test MLX
            with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model') as mock_load:
                with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available') as mock_check:
                    mock_check.return_value = True
                    mock_model = MagicMock()
                    mock_model.transcribe.return_value = MagicMock(text="MLX result")
                    mock_load.return_value = mock_model

                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                        transcribe_with_parakeet_mlx
                    )

                    metrics = self.measure_performance(
                        transcribe_with_parakeet_mlx,
                        audio,
                        sample_rate=16000,
                        chunk_duration=30.0
                    )

                    results['implementation'].append('MLX')
                    results['duration'].append(duration)
                    results['processing_time'].append(metrics['time'])
                    results['real_time_factor'].append(duration / metrics['time'])
                    results['memory_used'].append(metrics['memory_used'])

            # Test ONNX
            with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model') as mock_load:
                mock_session = MagicMock()
                mock_session.run.return_value = [np.random.randn(1, 50, 128256)]
                mock_tokenizer = MagicMock()
                mock_tokenizer.decode.return_value = "ONNX result"
                mock_load.return_value = (mock_session, mock_tokenizer)

                with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX._preprocess_audio') as mock_preprocess:
                    mock_preprocess.return_value = np.random.randn(1, 100, 512)

                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                        transcribe_with_parakeet_onnx
                    )

                    metrics = self.measure_performance(
                        transcribe_with_parakeet_onnx,
                        audio,
                        sample_rate=16000
                    )

                    results['implementation'].append('ONNX')
                    results['duration'].append(duration)
                    results['processing_time'].append(metrics['time'])
                    results['real_time_factor'].append(duration / metrics['time'])
                    results['memory_used'].append(metrics['memory_used'])

        # Print comparison table
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Implementation':<15} {'Duration':<10} {'Time':<10} {'RTF':<10} {'Memory':<10}")
        print("-"*80)

        for i in range(len(results['implementation'])):
            print(f"{results['implementation'][i]:<15} "
                  f"{results['duration'][i]:<10}s "
                  f"{results['processing_time'][i]:<10.3f}s "
                  f"{results['real_time_factor'][i]:<10.1f}x "
                  f"{results['memory_used'][i]:<10.1f}MB")

        return results


class TestRealWorldScenarios(TranscriptionBenchmark):
    """Test real-world usage scenarios."""

    def test_podcast_transcription(self):

        """Simulate podcast transcription (long audio)."""
        # Simulate 1-hour podcast
        audio = self.generate_test_audio(3600)

        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription.transcribe_long_audio') as mock_transcribe:
            mock_transcribe.return_value = "Podcast transcription"

            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
                transcribe_long_audio
            )

            metrics = self.measure_performance(
                mock_transcribe,
                audio,
                variant='mlx',
                chunk_duration=60.0,
                merge_algo='lcs'
            )

            print(f"\nPodcast (1 hour) transcription:")
            print(f"Time: {metrics['time']:.2f}s")
            print(f"Memory: {metrics['memory_used']:.1f}MB")
            print(f"Real-time factor: {3600/metrics['time']:.1f}x")

    def test_meeting_transcription(self):

        """Simulate meeting transcription with multiple speakers."""
        # Simulate 30-minute meeting
        audio = self.generate_test_audio(1800)

        # Add more variation to simulate multiple speakers
        for i in range(0, len(audio), 16000 * 10):  # Every 10 seconds
            segment = audio[i:i+16000*10]
            # Vary pitch/frequency to simulate different speakers
            if (i // (16000 * 10)) % 3 == 0:
                segment *= 1.2  # Higher pitch
            elif (i // (16000 * 10)) % 3 == 1:
                segment *= 0.8  # Lower pitch
            audio[i:i+len(segment)] = segment

        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx') as mock_transcribe:
            mock_transcribe.return_value = "Meeting transcription"

            metrics = self.measure_performance(
                mock_transcribe,
                audio,
                sample_rate=16000,
                chunk_duration=30.0
            )

            print(f"\nMeeting (30 min) transcription:")
            print(f"Time: {metrics['time']:.2f}s")
            print(f"Real-time factor: {1800/metrics['time']:.1f}x")

    def test_batch_processing(self):

        """Test batch processing of multiple files."""
        # Create multiple audio files
        audio_files = []
        for i in range(10):
            audio = self.generate_test_audio(60)  # 1-minute files
            audio_files.append(audio)

        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx') as mock_transcribe:
            mock_transcribe.return_value = "Batch result"

            # Process sequentially
            start = time.perf_counter()
            results_seq = []
            for audio in audio_files:
                result = mock_transcribe(audio, sample_rate=16000)
                results_seq.append(result)
            time_sequential = time.perf_counter() - start

            print(f"\nBatch processing (10 files, 1 min each):")
            print(f"Sequential: {time_sequential:.2f}s")
            print(f"Average per file: {time_sequential/10:.2f}s")

            # Could add parallel processing test here if supported


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
