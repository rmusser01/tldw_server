"""Docker warm container pool for sub-second sandbox startup.

Pre-creates idle containers with 'sleep infinity' entrypoint.
On claim, uses 'docker exec' to run the actual command.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
from collections import defaultdict
from typing import Any

from loguru import logger


class DockerWarmPool:
    """Maintains a pool of pre-created Docker containers.

    Containers are created with ``/bin/sh -c 'sleep infinity'`` as their
    entrypoint so they start immediately and idle until claimed.  When a
    caller claims a container the pool pops one off its internal list and
    returns the container ID.  The caller is then expected to use
    ``docker exec`` to run the real workload inside the already-running
    container.

    Containers are **single-use** after exec -- ``release()`` always
    destroys the container regardless of the *tainted* flag.
    """

    def __init__(
        self,
        pool_size: int | None = None,
        images: list[str] | None = None,
        replenish_interval: float = 10.0,
    ) -> None:
        self._pool_size = pool_size or int(os.getenv("SANDBOX_WARM_POOL_SIZE", "3"))
        self._images = images or os.getenv(
            "SANDBOX_WARM_POOL_IMAGES", "python:3.12-slim"
        ).split(",")
        self._replenish_interval = replenish_interval

        # image -> [container_id, ...]
        self._pool: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Register shutdown hook
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background replenishment thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._replenish_loop, daemon=True, name="warm-pool"
        )
        self._thread.start()
        logger.info(
            "Docker warm pool started: size={}, images={}",
            self._pool_size,
            self._images,
        )

    def shutdown(self) -> None:
        """Stop the pool and destroy all idle containers."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            for image, containers in self._pool.items():
                for cid in containers:
                    self._destroy_container(cid)
                containers.clear()
        logger.info("Docker warm pool shut down")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def claim(self, image: str) -> str | None:
        """Claim a pre-created container for the given image.

        Returns the container ID or ``None`` if the pool is empty for
        that image.
        """
        with self._lock:
            pool = self._pool.get(image, [])
            if pool:
                cid = pool.pop(0)
                logger.debug(
                    "Claimed container {} from warm pool (image={})",
                    cid[:12],
                    image,
                )
                return cid
        return None

    def release(self, container_id: str, tainted: bool = False) -> None:
        """Return a container to the pool, or destroy if tainted.

        Because containers are single-use after ``docker exec``, this
        always destroys the container regardless of the *tainted* flag.
        """
        if tainted:
            self._destroy_container(container_id)
            return
        # Don't return to pool -- containers are single-use after exec
        self._destroy_container(container_id)

    def pool_status(self) -> dict[str, int]:
        """Return current pool sizes per image."""
        with self._lock:
            return {img: len(cids) for img, cids in self._pool.items()}

    # ------------------------------------------------------------------
    # Background replenishment
    # ------------------------------------------------------------------

    def _replenish_loop(self) -> None:
        """Background thread that maintains pool levels."""
        while self._running:
            for image in self._images:
                self._replenish_image(image)
            time.sleep(self._replenish_interval)

    def _replenish_image(self, image: str) -> None:
        """Top up the pool for a single image."""
        with self._lock:
            current = len(self._pool.get(image, []))
            needed = self._pool_size - current

        for _ in range(needed):
            if not self._running:
                break
            try:
                cid = self._create_idle_container(image)
                with self._lock:
                    self._pool[image].append(cid)
            except Exception as exc:
                logger.warning(
                    "Failed to pre-create container for {}: {}", image, exc
                )
                break

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    def _create_idle_container(self, image: str) -> str:
        """Create a container with ``sleep infinity`` entrypoint."""
        cmd = [
            "docker",
            "create",
            "--cap-drop=ALL",
            "--read-only",
            "--network=none",
            "--memory=256m",
            "--pids-limit=64",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "sleep infinity",
        ]
        cid = subprocess.check_output(cmd, text=True, timeout=30).strip()
        subprocess.check_call(
            ["docker", "start", cid],
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return cid

    @staticmethod
    def _destroy_container(container_id: str) -> None:
        """Force-remove a container."""
        try:
            subprocess.check_call(
                ["docker", "rm", "-f", container_id],
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# Module-level singleton (lazy-initialized)
# ------------------------------------------------------------------

_warm_pool: DockerWarmPool | None = None


def get_warm_pool() -> DockerWarmPool:
    """Get or create the global warm pool instance."""
    global _warm_pool
    if _warm_pool is None:
        _warm_pool = DockerWarmPool()
        _warm_pool.start()
    return _warm_pool
