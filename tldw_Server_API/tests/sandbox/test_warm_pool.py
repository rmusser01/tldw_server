"""Tests for DockerWarmPool -- no real Docker required (subprocess mocked)."""
from __future__ import annotations

from unittest import mock

import pytest

from tldw_Server_API.app.core.Sandbox.pool import DockerWarmPool, get_warm_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(pool_size: int = 2, images: list[str] | None = None) -> DockerWarmPool:
    """Create a pool without starting the background thread."""
    pool = DockerWarmPool(
        pool_size=pool_size,
        images=images or ["python:3.12-slim"],
        replenish_interval=999,  # effectively disable auto-replenish
    )
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaimReturnsContainerFromPool:
    """Pre-populate pool, claim succeeds."""

    def test_claim_returns_container_from_pool(self) -> None:
        pool = _make_pool()
        image = "python:3.12-slim"
        # Manually seed the pool
        pool._pool[image] = ["abc123", "def456"]

        cid = pool.claim(image)

        assert cid == "abc123"
        # Pool should have one container left
        assert pool._pool[image] == ["def456"]


class TestClaimEmptyPoolReturnsNone:
    """Empty pool returns None."""

    def test_claim_empty_pool_returns_none(self) -> None:
        pool = _make_pool()

        cid = pool.claim("python:3.12-slim")

        assert cid is None


class TestReleaseTaintedDestroys:
    """Tainted release calls docker rm -f."""

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    def test_release_tainted_destroys(self, mock_check_call: mock.MagicMock) -> None:
        pool = _make_pool()

        pool.release("container_xyz", tainted=True)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args
        cmd = args[0][0]
        assert cmd == ["docker", "rm", "-f", "container_xyz"]

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    def test_release_untainted_also_destroys(self, mock_check_call: mock.MagicMock) -> None:
        """Containers are single-use after exec, so untainted also destroys."""
        pool = _make_pool()

        pool.release("container_xyz", tainted=False)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args
        cmd = args[0][0]
        assert cmd == ["docker", "rm", "-f", "container_xyz"]


class TestReplenishCreatesContainers:
    """Replenish fills pool to target size."""

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_output")
    def test_replenish_creates_containers(
        self,
        mock_check_output: mock.MagicMock,
        mock_check_call: mock.MagicMock,
    ) -> None:
        pool = _make_pool(pool_size=3)
        pool._running = True  # simulate started state
        image = "python:3.12-slim"

        # Each docker create returns a unique container id
        mock_check_output.side_effect = ["cid_1\n", "cid_2\n", "cid_3\n"]

        pool._replenish_image(image)

        # Should have created 3 containers
        assert mock_check_output.call_count == 3
        # And started each one
        assert mock_check_call.call_count == 3
        # Pool should now have 3 containers
        assert pool._pool[image] == ["cid_1", "cid_2", "cid_3"]

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_output")
    def test_replenish_tops_up_partial_pool(
        self,
        mock_check_output: mock.MagicMock,
        mock_check_call: mock.MagicMock,
    ) -> None:
        pool = _make_pool(pool_size=3)
        pool._running = True
        image = "python:3.12-slim"
        pool._pool[image] = ["existing_1"]

        mock_check_output.side_effect = ["cid_new_1\n", "cid_new_2\n"]

        pool._replenish_image(image)

        assert mock_check_output.call_count == 2
        assert pool._pool[image] == ["existing_1", "cid_new_1", "cid_new_2"]

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_output")
    def test_replenish_stops_on_error(
        self,
        mock_check_output: mock.MagicMock,
        mock_check_call: mock.MagicMock,
    ) -> None:
        pool = _make_pool(pool_size=3)
        pool._running = True
        image = "python:3.12-slim"

        mock_check_output.side_effect = [
            "cid_1\n",
            RuntimeError("docker daemon not running"),
        ]

        pool._replenish_image(image)

        # Only one container should have been created
        assert pool._pool[image] == ["cid_1"]


class TestPoolStatusReportsCounts:
    """pool_status returns correct counts."""

    def test_pool_status_reports_counts(self) -> None:
        pool = _make_pool(images=["img_a", "img_b"])
        pool._pool["img_a"] = ["c1", "c2", "c3"]
        pool._pool["img_b"] = ["c4"]

        status = pool.pool_status()

        assert status == {"img_a": 3, "img_b": 1}

    def test_pool_status_empty(self) -> None:
        pool = _make_pool()

        status = pool.pool_status()

        assert status == {}


class TestShutdownDestroysAll:
    """shutdown cleans up all containers."""

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    def test_shutdown_destroys_all(self, mock_check_call: mock.MagicMock) -> None:
        pool = _make_pool()
        pool._pool["img_a"] = ["c1", "c2"]
        pool._pool["img_b"] = ["c3"]

        pool.shutdown()

        # Should have called docker rm -f for each container
        assert mock_check_call.call_count == 3
        destroyed_ids = [
            call[0][0][-1]  # last element of the cmd list
            for call in mock_check_call.call_args_list
        ]
        assert set(destroyed_ids) == {"c1", "c2", "c3"}
        # Pool should be empty
        assert pool._pool["img_a"] == []
        assert pool._pool["img_b"] == []


class TestCreateIdleContainerUsesSleepInfinity:
    """Verify docker create args include sleep infinity."""

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_output")
    def test_create_idle_container_uses_sleep_infinity(
        self,
        mock_check_output: mock.MagicMock,
        mock_check_call: mock.MagicMock,
    ) -> None:
        pool = _make_pool()
        mock_check_output.return_value = "new_cid_123\n"

        cid = pool._create_idle_container("python:3.12-slim")

        assert cid == "new_cid_123"

        # Verify the docker create command
        create_cmd = mock_check_output.call_args[0][0]
        assert create_cmd == [
            "docker",
            "create",
            "--entrypoint",
            "/bin/sh",
            "python:3.12-slim",
            "-c",
            "sleep infinity",
        ]

        # Verify docker start was called
        start_cmd = mock_check_call.call_args[0][0]
        assert start_cmd == ["docker", "start", "new_cid_123"]


class TestGetWarmPoolSingleton:
    """get_warm_pool returns a singleton."""

    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_call")
    @mock.patch("tldw_Server_API.app.core.Sandbox.pool.subprocess.check_output")
    def test_get_warm_pool_returns_singleton(
        self,
        mock_check_output: mock.MagicMock,
        mock_check_call: mock.MagicMock,
    ) -> None:
        import tldw_Server_API.app.core.Sandbox.pool as pool_mod

        # Reset module-level singleton
        pool_mod._warm_pool = None

        try:
            p1 = pool_mod.get_warm_pool()
            p2 = pool_mod.get_warm_pool()
            assert p1 is p2
            assert p1._running is True
        finally:
            p1.shutdown()
            pool_mod._warm_pool = None
