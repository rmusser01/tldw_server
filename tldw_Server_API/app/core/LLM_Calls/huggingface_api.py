# huggingface_api.py
"""
HuggingFace API client for browsing and downloading GGUF models.
"""

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.http_client import create_async_client

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover - optional in some test environments
    _httpx = None  # type: ignore[assignment]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_HF_HTTP_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if _httpx is not None:
    _HF_HTTP_EXCEPTIONS = (
        _httpx.HTTPError,
        _httpx.TimeoutException,
    )

_HF_API_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
) + _HF_HTTP_EXCEPTIONS


def _validate_gguf_model_filename(model_file: str) -> str:
    filename = str(model_file or "").strip()
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).name != filename
        or not filename.lower().endswith(".gguf")
    ):
        raise ValueError("model_file must be a GGUF filename without path components")
    return filename


async def _async_retry_sleep(delay: float, attempt: int) -> None:
    if delay > 0:
        await asyncio.sleep(delay * (attempt + 1))


def _is_retryable_status_code(status: Optional[int]) -> bool:
    return status in _RETRYABLE_STATUS


class HuggingFaceAPI:
    """Client for interacting with HuggingFace API."""

    BASE_URL = "https://huggingface.co"
    API_BASE = f"{BASE_URL}/api"

    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace API client.

        Args:
            token: Optional HuggingFace API token for private repos
        """
        cfg = load_and_log_configs() or {}
        hf_cfg = cfg.get("huggingface_api", {})
        # Prefer config.txt, fallback to env if not provided
        self.token = token or hf_cfg.get("api_key") or os.environ.get("HUGGINGFACE_API_KEY", "")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
        # Retry/timeout settings (config.txt only as requested)
        try:
            self.api_retries = int(hf_cfg.get("api_retries", 2))
        except (TypeError, ValueError):
            self.api_retries = 2
        try:
            self.api_retry_delay = float(hf_cfg.get("api_retry_delay", 0.5))
        except (TypeError, ValueError):
            self.api_retry_delay = 0.5
        try:
            self.api_timeout = float(hf_cfg.get("api_timeout", 30.0))
        except (TypeError, ValueError):
            self.api_timeout = 30.0

    async def search_models(
        self,
        query: str = "",
        filter_tags: Optional[list[str]] = None,
        sort: str = "downloads",
        limit: int = 50,
        full_search: bool = False
    ) -> list[dict[str, Any]]:
        """
        Search for models on HuggingFace.

        Args:
            query: Search query string
            filter_tags: List of tags to filter by (e.g., ["gguf", "llama"])
            sort: Sort by "downloads", "likes", "lastModified"
            limit: Maximum number of results
            full_search: If True, search in model card content too

        Returns:
            List of model information dictionaries
        """
        params = {
            "limit": limit,
            "sort": sort,
            "direction": -1,  # Descending order
            "full": full_search
        }

        # Add search query
        if query:
            params["search"] = query

        # Build filter string
        filters = []
        if filter_tags:
            for tag in filter_tags:
                filters.append(tag)

        # Always filter for GGUF models
        filters.append("gguf")

        if filters:
            params["filter"] = filters

        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)
        async with create_async_client(timeout=self.api_timeout) as client:
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = await client.get(
                        f"{self.API_BASE}/models",
                        headers=self.headers,
                        params=params,
                    )
                    resp.raise_for_status()
                    return resp.json()
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    last_exc = e
                    if attempt + 1 >= attempts:
                        break
                    # simple decorrelated backoff
                    delay = max(0.001, (backoff_ms / 1000.0))
                    await asyncio.sleep(delay)
            logger.error(f"Error searching models: {last_exc}")
            return []

    async def get_model_info(self, repo_id: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            repo_id: Repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")

        Returns:
            Model information dictionary or None if error
        """
        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)
        async with create_async_client(timeout=self.api_timeout) as client:
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = await client.get(
                        f"{self.API_BASE}/models/{repo_id}",
                        headers=self.headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    last_exc = e
                    if attempt + 1 >= attempts:
                        break
                    await asyncio.sleep(max(0.001, (backoff_ms / 1000.0)))
            logger.error(f"Error getting model info for {repo_id}: {last_exc}")
            return None

    async def list_model_files(self, repo_id: str, path: str = "") -> list[dict[str, Any]]:
        """
        List files in a model repository.

        Args:
            repo_id: Repository ID
            path: Path within repository (default is root)

        Returns:
            List of file information dictionaries
        """
        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)
        async with create_async_client(timeout=self.api_timeout) as client:
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = await client.get(
                        f"{self.API_BASE}/models/{repo_id}/tree/main",
                        headers=self.headers,
                    )
                    resp.raise_for_status()
                    files = resp.json()
                    if path:
                        files = [f for f in files if f.get("path", "").startswith(path)]
                    gguf_files = [f for f in files if f.get("path", "").endswith(".gguf")]
                    return gguf_files
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    last_exc = e
                    if attempt + 1 >= attempts:
                        break
                    await asyncio.sleep(max(0.001, (backoff_ms / 1000.0)))
            logger.error(f"Error listing files for {repo_id}: {last_exc}")
            return []

    async def get_download_url(self, repo_id: str, filename: str, revision: str = "main") -> Optional[str]:
        """
        Get the download URL for a specific file.

        Args:
            repo_id: Repository ID
            filename: Name of the file to download
            revision: Git revision (branch, tag, or commit)

        Returns:
            Download URL or None if error
        """
        return f"{self.BASE_URL}/{repo_id}/resolve/{revision}/{filename}"

    async def download_file(
        self,
        repo_id: str,
        filename: str,
        destination: Path,
        revision: str = "main",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from a HuggingFace repository.

        Args:
            repo_id: Repository ID
            filename: Name of the file to download
            destination: Destination path for the downloaded file
            revision: Git revision
            progress_callback: Optional callback for progress updates (downloaded_bytes, total_bytes)
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        url = await self.get_download_url(repo_id, filename, revision)
        if not url:
            return False

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_file = destination.with_suffix(".tmp")

        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)

        try:
            async with create_async_client(timeout=self.api_timeout) as client:
                # HEAD for size
                total_size = 0
                last_exc: Optional[Exception] = None
                for attempt in range(attempts):
                    try:
                        head_resp = await client.head(url, headers=self.headers)
                        total_size = int(head_resp.headers.get("content-length", 0))
                        break
                    except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                        last_exc = e
                        if attempt + 1 >= attempts:
                            logger.error(f"HEAD failed for {filename}: {last_exc}")
                            return False
                        await asyncio.sleep(max(0.001, (backoff_ms / 1000.0)))

                if destination.exists() and total_size > 0 and destination.stat().st_size == total_size:
                    logger.info(f"File {filename} already exists with correct size, skipping download")
                    return True

                # Stream download
                try:
                    async with client.stream("GET", url, headers=self.headers) as resp:
                        resp.raise_for_status()
                        with open(temp_file, "wb") as f:
                            downloaded = 0
                            async for chunk in resp.aiter_bytes(chunk_size):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                downloaded += len(chunk)
                                if progress_callback and total_size:
                                    with contextlib.suppress(_HF_API_NONCRITICAL_EXCEPTIONS):
                                        progress_callback(min(downloaded, total_size), total_size)
                        temp_file.replace(destination)
                        # Final progress callback to ensure completion state
                        if progress_callback and total_size and downloaded < total_size:
                            with contextlib.suppress(_HF_API_NONCRITICAL_EXCEPTIONS):
                                progress_callback(total_size, total_size)
                        logger.info(f"Successfully downloaded {filename} to {destination}")
                        return True
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Error downloading {filename}: {e}")
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except _HF_API_NONCRITICAL_EXCEPTIONS:
                        pass
                    return False
        except _HF_API_NONCRITICAL_EXCEPTIONS as e:
            # Catch any unexpected errors outside inner blocks
            logger.error(f"Unexpected error downloading {filename} from {repo_id}: {e}")
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except _HF_API_NONCRITICAL_EXCEPTIONS:
                pass
            return False

    async def get_model_readme(self, repo_id: str) -> Optional[str]:
        """
        Get the README content for a model.

        Args:
            repo_id: Repository ID

        Returns:
            README content as string or None if not found
        """
        url = f"{self.BASE_URL}/{repo_id}/raw/main/README.md"

        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)
        async with create_async_client(timeout=self.api_timeout) as client:
            # Try README.md
            try:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code < 400:
                    return resp.text
            except _HF_API_NONCRITICAL_EXCEPTIONS:
                pass
            # Fallback README (no extension)
            alt = f"{self.BASE_URL}/{repo_id}/raw/main/README"
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = await client.get(alt, headers=self.headers)
                    if resp.status_code < 400:
                        return resp.text
                    else:
                        last_exc = Exception(f"status={resp.status_code}")
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    last_exc = e
                if attempt + 1 < attempts:
                    await asyncio.sleep(max(0.001, (backoff_ms / 1000.0)))
            logger.debug(f"No README found for {repo_id}: {last_exc}")
            return None

    async def get_model_config(self, repo_id: str) -> Optional[dict[str, Any]]:
        """
        Get the config.json for a model.

        Args:
            repo_id: Repository ID

        Returns:
            Config dictionary or None if not found
        """
        url = f"{self.BASE_URL}/{repo_id}/raw/main/config.json"

        attempts = max(1, int(self.api_retries)) + 1
        backoff_ms = int(self.api_retry_delay * 1000)
        async with create_async_client(timeout=self.api_timeout) as client:
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = await client.get(url, headers=self.headers)
                    resp.raise_for_status()
                    return resp.json()
                except _HF_API_NONCRITICAL_EXCEPTIONS as e:
                    last_exc = e
                    if attempt + 1 < attempts:
                        await asyncio.sleep(max(0.001, (backoff_ms / 1000.0)))
            logger.debug(f"No config.json found for {repo_id}: {last_exc}")
            return None

    async def search_gguf_models(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        size_range: Optional[tuple[int, int]] = None,
        quantization: Optional[str] = None,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Search specifically for GGUF models with additional filters.

        Args:
            query: Search query
            model_type: Filter by model type (e.g., "llama", "mistral", "mixtral")
            size_range: Tuple of (min_size_gb, max_size_gb)
            quantization: Quantization type (e.g., "Q4_K_M", "Q5_K_S")
            limit: Maximum results

        Returns:
            List of matching GGUF models
        """
        # Build filter tags
        filter_tags = ["gguf"]
        if model_type:
            filter_tags.append(model_type.lower())

        # Search models
        models = await self.search_models(
            query=query,
            filter_tags=filter_tags,
            sort="downloads",
            limit=limit * 2  # Get more to filter
        )

        # Further filter results
        filtered_models = []
        for model in models:
            # Get model files to check sizes and quantization
            if quantization or size_range:
                files = await self.list_model_files(model["modelId"])

                # Check quantization
                if quantization:
                    has_quant = any(quantization.lower() in f.get("path", "").lower() for f in files)
                    if not has_quant:
                        continue

                # Check size range
                if size_range and files:
                    # Get total size of GGUF files
                    total_size_bytes = sum(f.get("size", 0) for f in files)
                    total_size_gb = total_size_bytes / (1024 ** 3)

                    if not (size_range[0] <= total_size_gb <= size_range[1]):
                        continue

            filtered_models.append(model)

            if len(filtered_models) >= limit:
                break

        return filtered_models


# Utility functions for common operations
async def find_best_gguf_model(
    model_name: str,
    max_size_gb: float = 10.0,
    preferred_quant: Optional[str] = "Q4_K_M"
) -> Optional[dict[str, Any]]:
    """
    Find the best GGUF version of a model based on criteria.

    Args:
        model_name: Name of the model to search for
        max_size_gb: Maximum model size in GB
        preferred_quant: Preferred quantization type

    Returns:
        Best matching model info or None
    """
    api = HuggingFaceAPI()

    # Search for the model
    models = await api.search_gguf_models(
        query=model_name,
        size_range=(0, max_size_gb),
        quantization=preferred_quant,
        limit=5
    )

    if not models:
        # Try without quantization preference
        models = await api.search_gguf_models(
            query=model_name,
            size_range=(0, max_size_gb),
            limit=5
        )

    # Return the most downloaded one
    return models[0] if models else None


async def download_gguf_model(
    repo_id: str,
    model_file: str,
    destination_dir: Path,
    show_progress: bool = True
) -> bool:
    """
    Download a GGUF model file with progress indication.

    Args:
        repo_id: HuggingFace repository ID
        model_file: Name of the GGUF file
        destination_dir: Directory to save the model
        show_progress: Whether to show download progress

    Returns:
        True if successful
    """
    api = HuggingFaceAPI()

    model_file = _validate_gguf_model_filename(model_file)
    destination_root = Path(destination_dir).resolve()
    destination = (destination_root / model_file).resolve()
    if destination.parent != destination_root:
        raise ValueError("model_file destination must remain inside destination_dir")

    last_pct = {"v": -10.0}

    def progress_callback(downloaded: int, total: int):
        if show_progress and total > 0:
            percent = (downloaded / total) * 100
            if percent - last_pct["v"] >= 10.0 or downloaded >= total:
                last_pct["v"] = percent
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                logger.info(
                    f"Downloading {model_file}: {percent:.0f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                )

    success = await api.download_file(
        repo_id=repo_id,
        filename=model_file,
        destination=destination,
        progress_callback=progress_callback if show_progress else None
    )

    return success
