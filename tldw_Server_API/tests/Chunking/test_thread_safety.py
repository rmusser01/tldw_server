# test_thread_safety.py
"""
Thread safety tests for Chunking module singletons and shared state.

Tests verify that:
1. security_logger singleton is thread-safe
2. metrics singleton is thread-safe
3. TokenChunkingStrategy._failed_tokenizers set is thread-safe
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

import pytest


class TestSecurityLoggerThreadSafety:
    """Tests for thread-safe security logger singleton."""

    def test_get_security_logger_returns_same_instance(self):
        """Verify get_security_logger returns the same instance."""
        from tldw_Server_API.app.core.Chunking.security_logger import (
            get_security_logger,
            configure_security_logging,
            _security_logger_lock,
        )

        # Reset the singleton for test isolation
        configure_security_logging()

        logger1 = get_security_logger()
        logger2 = get_security_logger()

        assert logger1 is logger2

    def test_get_security_logger_concurrent_access(self):
        """Test concurrent calls to get_security_logger return same instance."""
        from tldw_Server_API.app.core.Chunking.security_logger import (
            get_security_logger,
            configure_security_logging,
        )

        # Reset singleton
        configure_security_logging()

        instances = []
        errors = []

        def get_logger():

            try:
                instance = get_security_logger()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        # Run 100 concurrent accesses
        threads = [threading.Thread(target=get_logger) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(instances) == 100
        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)

    def test_configure_security_logging_thread_safe(self):
        """Test configure_security_logging is thread-safe."""
        from tldw_Server_API.app.core.Chunking.security_logger import (
            get_security_logger,
            configure_security_logging,
        )

        errors = []

        def reconfigure():

            try:
                configure_security_logging()
            except Exception as e:
                errors.append(e)

        # Run concurrent reconfigurations
        threads = [threading.Thread(target=reconfigure) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Should still be able to get a valid logger
        logger = get_security_logger()
        assert logger is not None


class TestMetricsThreadSafety:
    """Tests for thread-safe metrics singleton."""

    def test_get_metrics_returns_same_instance(self):
        """Verify get_metrics returns the same instance."""
        from tldw_Server_API.app.core.Chunking.utils.metrics import get_metrics

        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2

    def test_get_metrics_concurrent_access(self):
        """Test concurrent calls to get_metrics return same instance."""
        from tldw_Server_API.app.core.Chunking.utils.metrics import get_metrics

        instances = []
        errors = []

        def get_instance():

            try:
                instance = get_metrics()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        # Run 100 concurrent accesses
        threads = [threading.Thread(target=get_instance) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(instances) == 100
        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


class TestTokenizerFailedCacheThreadSafety:
    """Tests for thread-safe TokenChunkingStrategy._failed_tokenizers."""

    def test_failed_tokenizers_concurrent_access(self):
        """Test concurrent access to _failed_tokenizers set is thread-safe."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import (
            TokenChunkingStrategy,
        )

        # Clear the failed tokenizers for test isolation
        with TokenChunkingStrategy._failed_tokenizers_lock:
            TokenChunkingStrategy._failed_tokenizers.clear()

        errors = []

        def add_failed_tokenizer(name: str):
            try:
                with TokenChunkingStrategy._failed_tokenizers_lock:
                    TokenChunkingStrategy._failed_tokenizers.add(name)
            except Exception as e:
                errors.append(e)

        def check_failed_tokenizer(name: str):
            try:
                with TokenChunkingStrategy._failed_tokenizers_lock:
                    _ = name in TokenChunkingStrategy._failed_tokenizers
            except Exception as e:
                errors.append(e)

        # Run concurrent reads and writes
        threads = []
        for i in range(50):
            threads.append(threading.Thread(target=add_failed_tokenizer, args=(f"tokenizer_{i}",)))
            threads.append(threading.Thread(target=check_failed_tokenizer, args=(f"tokenizer_{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all tokenizers were added
        with TokenChunkingStrategy._failed_tokenizers_lock:
            assert len(TokenChunkingStrategy._failed_tokenizers) == 50

    def test_tokenizer_property_concurrent_initialization(self):
        """Test concurrent tokenizer property access is thread-safe."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import (
            TokenChunkingStrategy,
        )

        # Clear the failed tokenizers for test isolation
        with TokenChunkingStrategy._failed_tokenizers_lock:
            TokenChunkingStrategy._failed_tokenizers.clear()

        errors = []
        tokenizers = []

        def get_tokenizer():

            try:
                # Use a standard tokenizer name - we're testing thread safety, not fallback behavior
                strategy = TokenChunkingStrategy(tokenizer_name="gpt2")
                tok = strategy.tokenizer
                tokenizers.append(tok)
            except Exception as e:
                errors.append(e)

        # Run concurrent tokenizer initializations
        threads = [threading.Thread(target=get_tokenizer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(tokenizers) == 20
        # All tokenizers should have been successfully initialized (any type is fine)
        assert all(tok is not None for tok in tokenizers)


class TestRaceConditionScenarios:
    """Tests for specific race condition scenarios."""

    def test_no_duplicate_initialization(self):
        """Verify singletons don't get initialized multiple times under load."""
        from tldw_Server_API.app.core.Chunking.security_logger import (
            SecurityLogger,
            configure_security_logging,
            get_security_logger,
        )

        # Track initialization calls
        init_count = 0
        original_init = SecurityLogger.__init__

        def counting_init(self, *args, **kwargs):

            nonlocal init_count
            init_count += 1
            return original_init(self, *args, **kwargs)

        # Reset and patch
        configure_security_logging()  # Reset to known state

        with patch.object(SecurityLogger, "__init__", counting_init):
            # Reconfigure to use patched init
            configure_security_logging()
            init_count = 0  # Reset after configure

            # Now concurrent access should NOT reinitialize
            threads = [threading.Thread(target=get_security_logger) for _ in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # init should not have been called after the initial configure
            assert init_count == 0

    def test_chunk_text_generator_does_not_hold_shared_lock_while_paused(self, monkeypatch):
        """A paused generator consumer should not block unrelated chunk_text calls."""
        from tldw_Server_API.app.core.Chunking.chunker import Chunker
        from tldw_Server_API.app.core.Chunking.base import ChunkerConfig
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        started = threading.Event()
        completed = threading.Event()
        result_holder = {}

        def controlled_chunk_generator(self, text, max_size, overlap=0, **options):
            yield "chunk-1"
            yield "chunk-2"

        monkeypatch.setattr(WordChunkingStrategy, "chunk_generator", controlled_chunk_generator)

        chunker = Chunker(ChunkerConfig(strategy_cache_mode="shared"))
        generator = chunker.chunk_text_generator("one two three", method="words", max_size=2, overlap=0)

        assert next(generator) == "chunk-1"

        def run_competing_call():
            started.set()
            result_holder["chunks"] = chunker.chunk_text("alpha beta gamma", method="words", max_size=2, overlap=0)
            completed.set()

        thread = threading.Thread(target=run_competing_call)
        thread.start()

        try:
            assert started.wait(0.2)
            assert completed.wait(0.2), "chunk_text blocked behind paused generator"
        finally:
            generator.close()
            thread.join(timeout=1)

        assert result_holder["chunks"]


class TestThreadPoolExecutorAccess:
    """Tests using ThreadPoolExecutor for more controlled concurrency."""

    def test_security_logger_with_executor(self):
        """Test security logger access with ThreadPoolExecutor."""
        from tldw_Server_API.app.core.Chunking.security_logger import (
            get_security_logger,
            configure_security_logging,
            SecurityEventType,
        )

        configure_security_logging()

        def get_and_log():

            logger = get_security_logger()
            logger.log_event(SecurityEventType.INVALID_INPUT, "test message")  # Use proper enum value
            return id(logger)

        # Use executor for concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_and_log) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        # All should return the same instance ID
        assert len(set(results)) == 1

    def test_metrics_with_executor(self):
        """Test metrics access with ThreadPoolExecutor."""
        from tldw_Server_API.app.core.Chunking.utils.metrics import get_metrics

        def get_and_record():

            metrics = get_metrics()
            metrics.record_request("test_method", "success")
            return id(metrics)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_and_record) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        # All should return the same instance ID
        assert len(set(results)) == 1
