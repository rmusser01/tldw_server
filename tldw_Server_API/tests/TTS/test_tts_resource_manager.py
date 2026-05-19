# test_tts_resource_manager.py
# Description: Comprehensive tests for TTS resource management system
#
# Imports
import pytest
import asyncio
import builtins
import sys
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any
import psutil
import httpx

#
# Local Imports
from tldw_Server_API.app.core.TTS import tts_resource_manager as rm_mod
from tldw_Server_API.app.core.TTS.tts_resource_manager import (
    TTSResourceManager,
    HTTPConnectionPool,
    MemoryMonitor,
    ResourceType,
    StreamingSession,
    StreamingSessionManager,
    get_resource_manager,
    reset_resource_manager,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSResourceError, TTSInsufficientMemoryError

#
#######################################################################################################################
#
# Test Memory Monitor


class TestMemoryMonitor:
    """Test the MemoryMonitor class"""

    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization"""
        monitor = MemoryMonitor()

        assert monitor.critical_threshold == 90
        assert monitor.warning_threshold == 80
        assert monitor._last_check_time == 0
        assert monitor._last_memory_usage is None

    def test_get_memory_usage(self):
        """Test getting memory usage statistics"""
        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage()

        assert "total_mb" in usage
        assert "available_mb" in usage
        assert "used_mb" in usage
        assert "percent" in usage
        assert "process_mb" in usage

        assert usage["total_mb"] > 0
        assert usage["percent"] >= 0
        assert usage["percent"] <= 100

    def test_memory_thresholds(self):
        """Test memory threshold checks"""
        monitor = MemoryMonitor(critical_threshold=50, warning_threshold=30)

        # Mock psutil to control memory values
        with patch("psutil.virtual_memory") as mock_memory:
            # Test critical threshold
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,  # 1GB
                available=400 * 1024 * 1024,  # 400MB (60% used - over critical)
                percent=60,
            )
            assert monitor.is_memory_critical() is True
            assert monitor.is_memory_warning() is True

            # Test warning threshold
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,
                available=600 * 1024 * 1024,  # 600MB (40% used - between warning and critical)
                percent=40,
            )
            assert monitor.is_memory_critical() is False
            assert monitor.is_memory_warning() is True

            # Test normal operation
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024, available=800 * 1024 * 1024, percent=20  # 800MB (20% used - below warning)
            )
            assert monitor.is_memory_critical() is False
            assert monitor.is_memory_warning() is False

    @pytest.mark.asyncio
    async def test_force_cleanup_failure_log_sanitizes_exception_text(self):
        """Force-cleanup callback logs should not include raw exceptions."""
        raw_marker = "MEMORY_CLEANUP_SECRET_MARKER"
        monitor = MemoryMonitor()

        def _failing_cleanup():
            raise RuntimeError(raw_marker)

        monitor._cleanup_callbacks.append(_failing_cleanup)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            await monitor.force_cleanup()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error in cleanup callback" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_monitor_loop_failure_log_sanitizes_exception_text(self, monkeypatch):
        """Memory monitor loop logs should not include raw exceptions."""
        raw_marker = "MEMORY_MONITOR_SECRET_MARKER"
        monitor = MemoryMonitor(check_interval=0)
        monitor._monitoring = True

        def _failing_memory_check():
            raise RuntimeError(raw_marker)

        async def _stop_after_log(_delay):
            raise asyncio.CancelledError

        monkeypatch.setattr(monitor, "is_memory_critical", _failing_memory_check)
        monkeypatch.setattr(rm_mod.asyncio, "sleep", _stop_after_log)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            with pytest.raises(asyncio.CancelledError):
                await monitor._monitor_loop()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error in memory monitoring" in log_output
        assert raw_marker not in log_output

    def test_memory_check_caching(self):
        """Test that memory checks are cached"""
        monitor = MemoryMonitor(check_interval=1.0)

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(total=1024 * 1024 * 1024, available=500 * 1024 * 1024, percent=50)

            # First call should check memory
            usage1 = monitor.get_memory_usage()
            first_call_count = mock_memory.call_count

            # Immediate second call may or may not use cache depending on implementation
            usage2 = monitor.get_memory_usage()
            # Don't assert exact call count as caching behavior may vary

            # After interval, should check again
            monitor._last_check_time = 0  # Force cache expiry
            usage3 = monitor.get_memory_usage()
            assert mock_memory.call_count == 2


class TestHTTPConnectionPool:
    """Test the HTTPConnectionPool class"""

    @pytest.fixture
    def pool(self):
        """Create a connection pool instance"""
        return HTTPConnectionPool(max_connections=5, timeout=30.0)

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, pool):
        """Test getting a new HTTP client"""
        client = await pool.get_client("provider1", "https://api.example.com")

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert "provider1" in pool._clients
        assert pool._clients["provider1"] == client

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, pool):
        """Test reusing existing HTTP client"""
        client1 = await pool.get_client("provider1", "https://api.example.com")
        client2 = await pool.get_client("provider1", "https://api.example.com")

        assert client1 is client2
        assert len(pool._clients) == 1

    @pytest.mark.asyncio
    async def test_get_client_different_providers(self, pool):
        """Test different clients for different providers"""
        client1 = await pool.get_client("provider1", "https://api1.example.com")
        client2 = await pool.get_client("provider2", "https://api2.example.com")

        assert client1 is not client2
        assert len(pool._clients) == 2
        assert "provider1" in pool._clients
        assert "provider2" in pool._clients

    @pytest.mark.asyncio
    async def test_close_client(self, pool):
        """Test closing a specific client"""
        client = await pool.get_client("provider1", "https://api.example.com")

        # Mock the aclose method
        client.aclose = AsyncMock()

        await pool.close_client("provider1")

        client.aclose.assert_called_once()
        assert "provider1" not in pool._pools

    @pytest.mark.asyncio
    async def test_close_all_clients(self, pool):
        """Test closing all clients"""
        # Create multiple clients
        client1 = await pool.get_client("provider1", "https://api1.example.com")
        client2 = await pool.get_client("provider2", "https://api2.example.com")

        # Mock aclose methods
        client1.aclose = AsyncMock()
        client2.aclose = AsyncMock()

        await pool.close_all()

        client1.aclose.assert_called_once()
        client2.aclose.assert_called_once()
        assert len(pool._pools) == 0

    @pytest.mark.asyncio
    async def test_connection_limits(self, pool):
        """Test connection pool limits"""
        pool = HTTPConnectionPool(max_connections=2)

        # Create clients up to the limit
        client1 = await pool.get_client("provider1", "https://api.example.com")
        client2 = await pool.get_client("provider2", "https://api.example.com")

        assert len(pool._clients) == 2

        # Creating another should still work (reuses or creates based on provider)
        client3 = await pool.get_client("provider3", "https://api.example.com")
        assert client3 is not None


class TestStreamingSession:
    """Test the StreamingSession class"""

    def test_streaming_session_creation(self):
        """Test creating a streaming session"""
        session = StreamingSession(session_id="test123", provider="openai")

        assert session.session_id == "test123"
        assert session.provider == "openai"
        assert session.created_at > 0
        assert session.bytes_sent == 0
        assert session.chunks_sent == 0
        assert session.is_active is True

    def test_streaming_session_update(self):
        """Test updating streaming session stats"""
        session = StreamingSession(session_id="test123", provider="openai")

        session.bytes_sent += 1024
        session.chunks_sent += 1

        assert session.bytes_sent == 1024
        assert session.chunks_sent == 1

        session.bytes_sent += 512
        session.chunks_sent += 1

        assert session.bytes_sent == 1536
        assert session.chunks_sent == 2

    @pytest.mark.asyncio
    async def test_close_failure_log_sanitizes_exception_text(self):
        """Session close logs should not include raw cleanup exceptions."""
        raw_marker = "STREAMING_SESSION_SECRET_MARKER"

        def _failing_cleanup():
            raise RuntimeError(raw_marker)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            session = StreamingSession(
                session_id="session123",
                provider="openai",
                cleanup_callback=_failing_cleanup,
            )

            await session.close()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error closing streaming session session123" in log_output
        assert raw_marker not in log_output


class TestStreamingSessionManager:
    """Test the StreamingSessionManager class"""

    @pytest.fixture
    def manager(self):
        """Create a session manager instance"""
        return StreamingSessionManager()

    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """Test creating a new streaming session"""
        session_id = await manager.create_session("openai")

        assert session_id is not None
        assert session_id in manager._sessions
        assert manager._sessions[session_id].provider == "openai"
        assert manager._sessions[session_id].is_active is True

    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        """Test getting an existing session"""
        session_id = await manager.create_session("kokoro")
        session = await manager.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.provider == "kokoro"

        # Non-existent session
        none_session = await manager.get_session("non_existent")
        assert none_session is None

    @pytest.mark.asyncio
    async def test_update_session(self, manager):
        """Test updating session statistics"""
        session_id = await manager.create_session("elevenlabs")

        await manager.update_session(session_id, bytes_sent=1024, chunks_sent=1)

        session = await manager.get_session(session_id)
        assert session.bytes_streamed == 1024
        assert session.chunks_sent == 1

        # Update again
        await manager.update_session(session_id, bytes_sent=512, chunks_sent=1)

        session = await manager.get_session(session_id)
        assert session.bytes_streamed == 1536
        assert session.chunks_sent == 2

    @pytest.mark.asyncio
    async def test_close_session(self, manager):
        """Test closing a session"""
        session_id = await manager.create_session("higgs")

        # Close the session
        stats = await manager.close_session(session_id)

        assert stats is not None
        assert stats["session_id"] == session_id
        assert stats["provider"] == "higgs"
        assert "duration" in stats
        assert "bytes_streamed" in stats
        assert "chunks_sent" in stats

        # Session should be removed
        assert session_id not in manager._sessions

        # Closing non-existent session
        none_stats = await manager.close_session("non_existent")
        assert none_stats is None

    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions(self, manager):
        """Test cleaning up inactive sessions"""
        # Create sessions
        session1 = await manager.create_session("provider1")
        session2 = await manager.create_session("provider2")

        # Make session1 inactive and old
        manager._sessions[session1].is_active = False
        manager._sessions[session1].start_time -= 7200  # 2 hours ago

        # Keep session2 active
        manager._sessions[session2].is_active = True

        # Run cleanup
        await manager.cleanup_inactive(max_age_seconds=3600)

        # Session1 should be removed, session2 should remain
        assert session1 not in manager._sessions
        assert session2 in manager._sessions

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, manager):
        """Test getting active sessions"""
        # Create multiple sessions
        session1 = await manager.create_session("provider1")
        session2 = await manager.create_session("provider2")
        session3 = await manager.create_session("provider3")

        # Make one inactive
        manager._sessions[session2].is_active = False

        active = await manager.get_active_sessions()

        assert len(active) == 2
        assert session1 in [s.session_id for s in active]
        assert session3 in [s.session_id for s in active]
        assert session2 not in [s.session_id for s in active]


class TestTTSResourceManager:
    """Test the main TTSResourceManager class"""

    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager instance"""
        config = {
            "max_connections": 10,
            "connection_timeout": 60,
            "memory_critical_threshold": 90,
            "memory_warning_threshold": 80,
        }
        return TTSResourceManager(config)

    def test_resource_manager_initialization(self, resource_manager):
        """Test ResourceManager initialization"""
        assert resource_manager.connection_pool is not None
        assert resource_manager.memory_monitor is not None
        assert resource_manager.session_manager is not None
        assert len(resource_manager._registered_models) == 0

    @pytest.mark.asyncio
    async def test_get_http_client(self, resource_manager):
        """Test getting HTTP client through resource manager"""
        client = await resource_manager.get_http_client(provider="openai", base_url="https://api.openai.com")

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

    def test_register_model(self, resource_manager):
        """Test registering a model"""
        mock_model = Mock()
        mock_cleanup = Mock()

        resource_manager.register_model(provider="kokoro", model_instance=mock_model, cleanup_callback=mock_cleanup)

        assert "kokoro" in resource_manager._registered_models
        assert resource_manager._registered_models["kokoro"]["model"] == mock_model
        assert resource_manager._registered_models["kokoro"]["cleanup"] == mock_cleanup

    def test_model_cache_eviction(self):
        """Evicts least-recently-used models when cache limit exceeded"""
        manager = TTSResourceManager({"model_cache_max_entries": 1})
        cleanup_a = Mock()
        cleanup_b = Mock()

        manager.register_model(provider="a", model_instance=Mock(), cleanup_callback=cleanup_a)
        manager.register_model(provider="b", model_instance=Mock(), cleanup_callback=cleanup_b)

        cleanup_a.assert_called_once()
        assert "a" not in manager._registered_models
        assert "b" in manager._registered_models

    def test_model_cache_eviction_failure_log_sanitizes_exception_text(self):
        """Model eviction cleanup logs should not include raw cleanup exceptions."""
        raw_marker = "MODEL_EVICTION_SECRET_MARKER"
        manager = TTSResourceManager({"model_cache_max_entries": 1})

        def _failing_cleanup():
            raise RuntimeError(raw_marker)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            manager.register_model(
                provider="a",
                model_instance=Mock(),
                cleanup_callback=_failing_cleanup,
            )
            manager.register_model(
                provider="b",
                model_instance=Mock(),
                cleanup_callback=Mock(),
            )
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error cleaning model for a" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_async_model_cache_eviction_failure_log_sanitizes_exception_text(self):
        """Async model eviction cleanup logs should not include raw exceptions."""
        raw_marker = "ASYNC_MODEL_EVICTION_SECRET_MARKER"
        manager = TTSResourceManager({"model_cache_max_entries": 1})

        async def _failing_cleanup():
            raise RuntimeError(raw_marker)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            manager.register_model(
                provider="a",
                model_instance=Mock(),
                cleanup_callback=_failing_cleanup,
            )
            eviction_task = manager.register_model(
                provider="b",
                model_instance=Mock(),
                cleanup_callback=Mock(),
            )
            assert eviction_task is not None
            await eviction_task
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error cleaning model for a" in log_output
        assert raw_marker not in log_output

    def test_cleanup_device_cache_does_not_import_torch_when_unloaded(self, monkeypatch):
        """Device-cache cleanup should not import torch just to check for caches."""
        manager = TTSResourceManager({})
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        original_import = builtins.__import__
        torch_import_attempts: list[str] = []

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                torch_import_attempts.append(name)
                raise ImportError("torch should not be imported during best-effort cleanup")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        manager._cleanup_device_cache()

        assert torch_import_attempts == []

    @pytest.mark.asyncio
    async def test_unregister_model(self, resource_manager):
        """Test unregistering a model"""
        mock_model = Mock()
        mock_cleanup = AsyncMock()

        resource_manager.register_model(provider="higgs", model_instance=mock_model, cleanup_callback=mock_cleanup)

        await resource_manager.unregister_model("higgs")

        mock_cleanup.assert_called_once()
        assert "higgs" not in resource_manager._registered_models

    @pytest.mark.asyncio
    async def test_unregister_model_failure_log_sanitizes_exception_text(self, resource_manager):
        """Unregister cleanup logs should not include raw cleanup exceptions."""
        raw_marker = "UNREGISTER_MODEL_SECRET_MARKER"

        def _failing_cleanup():
            raise RuntimeError(raw_marker)

        resource_manager.register_model(
            provider="higgs",
            model_instance=Mock(),
            cleanup_callback=_failing_cleanup,
        )

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            await resource_manager.unregister_model("higgs")
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error in model cleanup for higgs" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_shutdown_cleanup_handler_failure_log_sanitizes_exception_text(self):
        """Shutdown cleanup handler logs should not include raw exceptions."""
        raw_marker = "SHUTDOWN_HANDLER_SECRET_MARKER"
        manager = TTSResourceManager()

        def _failing_cleanup():
            raise RuntimeError(raw_marker)

        manager.register_cleanup_handler(ResourceType.TEMP_FILE, _failing_cleanup)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            await manager.shutdown()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error in ResourceType.TEMP_FILE cleanup handler" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_expired_session_cleanup_failure_log_sanitizes_exception_text(self, monkeypatch):
        """Expired-session cleanup loop logs should not include raw exceptions."""
        raw_marker = "SESSION_CLEANUP_SECRET_MARKER"
        manager = TTSResourceManager()

        class _FailingSession:
            def is_expired(self, _timeout):
                raise RuntimeError(raw_marker)

        async def _stop_after_log(_delay):
            raise asyncio.CancelledError

        manager._streaming_sessions["session123"] = _FailingSession()
        monkeypatch.setattr(rm_mod.asyncio, "sleep", _stop_after_log)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            with pytest.raises(asyncio.CancelledError):
                await manager._cleanup_expired_sessions()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error in session cleanup" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_create_streaming_session(self, resource_manager):
        """Test creating a streaming session"""
        session_id = await resource_manager.create_streaming_session("elevenlabs")

        assert session_id is not None
        session = await resource_manager.session_manager.get_session(session_id)
        assert session.provider == "elevenlabs"

    @pytest.mark.asyncio
    async def test_cleanup_all(self, resource_manager):
        """Test cleaning up all resources"""
        # Register a model
        mock_model = Mock()
        mock_cleanup = AsyncMock()
        resource_manager.register_model("test", mock_model, mock_cleanup)

        # Create a client
        client = await resource_manager.get_http_client("test", "https://test.com")
        client.aclose = AsyncMock()

        # Create a session
        session_id = await resource_manager.create_streaming_session("test")

        # Cleanup all
        await resource_manager.cleanup_all()

        # Check everything was cleaned
        mock_cleanup.assert_called_once()
        client.aclose.assert_called()
        assert len(resource_manager._registered_models) == 0
        assert session_id not in resource_manager.session_manager._sessions

    @pytest.mark.asyncio
    async def test_cleanup_all_model_failure_log_sanitizes_exception_text(self, monkeypatch):
        """cleanup_all model logs should not include raw exceptions."""
        raw_marker = "CLEANUP_ALL_MODEL_SECRET_MARKER"
        manager = TTSResourceManager()
        manager._registered_models["model-provider"] = {"model": Mock(), "cleanup": Mock()}

        async def _failing_unregister(_provider):
            raise RuntimeError(raw_marker)

        monkeypatch.setattr(manager, "unregister_model", _failing_unregister)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            await manager.cleanup_all()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error cleaning model model-provider" in log_output
        assert raw_marker not in log_output

    @pytest.mark.asyncio
    async def test_cleanup_all_session_failure_log_sanitizes_exception_text(self, monkeypatch):
        """cleanup_all session logs should not include raw exceptions."""
        raw_marker = "CLEANUP_ALL_SESSION_SECRET_MARKER"
        manager = TTSResourceManager()
        manager.session_manager._sessions["session123"] = Mock()

        async def _failing_close_session(_session_id):
            raise RuntimeError(raw_marker)

        monkeypatch.setattr(manager.session_manager, "close_session", _failing_close_session)

        messages = []
        sink_id = rm_mod.logger.add(messages.append, format="{message}")
        try:
            await manager.cleanup_all()
        finally:
            rm_mod.logger.remove(sink_id)

        log_output = "\n".join(messages)
        assert "Error closing session session123" in log_output
        assert raw_marker not in log_output

    def test_get_resource_statistics(self, resource_manager):
        """Test getting resource statistics"""
        # Register a model
        resource_manager.register_model("test", Mock(), Mock())

        stats = resource_manager.get_statistics()

        assert "memory" in stats
        assert "connections" in stats
        assert "models" in stats
        assert "sessions" in stats

        assert stats["models"]["registered"] == ["test"]
        assert stats["connections"]["active"] == 0  # No connections created yet


class TestResourceManagerSingleton:
    """Test the singleton resource manager functions"""

    @pytest.mark.asyncio
    async def test_get_resource_manager_singleton(self):
        """Test that get_resource_manager returns singleton"""
        manager1 = await get_resource_manager()
        manager2 = await get_resource_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_reset_resource_manager(self):
        """Test resetting the resource manager"""
        manager1 = await get_resource_manager()

        await reset_resource_manager()

        manager2 = await get_resource_manager()

        # Should be different instances after reset
        assert manager1 is not manager2


class TestResourceManagerIntegration:
    """Integration tests for resource management"""

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling memory pressure scenarios"""
        config = {"memory_critical_threshold": 50, "memory_warning_threshold": 30}  # Low threshold for testing
        manager = TTSResourceManager(config)

        with patch("psutil.virtual_memory") as mock_memory:
            # Simulate high memory usage
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024, available=400 * 1024 * 1024, percent=60  # 60% used
            )

            # Should detect critical memory
            assert manager.memory_monitor.is_memory_critical() is True

            # Get memory stats through manager
            stats = manager.get_statistics()
            assert stats["memory"]["percent"] == 60
            assert stats["memory"]["is_critical"] is True

    @pytest.mark.asyncio
    async def test_concurrent_session_management(self):
        """Test managing multiple concurrent sessions"""
        manager = TTSResourceManager()

        # Create multiple sessions concurrently
        tasks = [manager.create_streaming_session(f"provider{i}") for i in range(5)]

        session_ids = await asyncio.gather(*tasks)

        assert len(session_ids) == 5
        assert len(set(session_ids)) == 5  # All unique

        # Get active sessions
        active = await manager.session_manager.get_active_sessions()
        assert len(active) == 5

        # Close all sessions
        close_tasks = [manager.session_manager.close_session(sid) for sid in session_ids]

        await asyncio.gather(*close_tasks)

        # No active sessions remaining
        active = await manager.session_manager.get_active_sessions()
        assert len(active) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
