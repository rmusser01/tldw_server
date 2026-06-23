# Llamafile_Handler.py
#
# Imports
#
# Third-party imports
import asyncio
import contextlib
import os
import platform
import shutil
import signal
# Used without shell for managed process flags, exceptions, and cleanup.
import subprocess  # nosec B404
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from tldw_Server_API.app.core.Local_LLM import handler_utils, http_utils

#
# Local imports
from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import (
    InferenceError,
    ModelDownloadError,
    ModelNotFoundError,
    ServerError,
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamafileConfig
from tldw_Server_API.app.core.Utils.Utils import verify_checksum


def redact_cmd_args(*args, **kwargs):
    """Proxy command redaction for easier monkeypatching in tests."""
    return http_utils.redact_cmd_args(*args, **kwargs)


def create_async_client(*args, **kwargs):
    """Proxy AsyncClient factory to respect patched targets."""
    return http_utils.create_async_client(*args, **kwargs)


async def request_json(*args, **kwargs):
    """Proxy JSON request helper."""
    return await http_utils.request_json(*args, **kwargs)


async def wait_for_http_ready(*args, **kwargs):
    """Proxy readiness poller preserving test expectations."""
    return await http_utils.wait_for_http_ready(*args, **kwargs)

_LLAMAFILE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    subprocess.SubprocessError,
    TimeoutError,
    TypeError,
    ValueError,
)

########################################################################################################################
#
# Functions:


class LlamafileHandler(BaseLLMHandler):
    def __init__(self, config: LlamafileConfig, global_app_config: dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: LlamafileConfig  # For type hinting

        self.llamafile_exe_path = self.config.llamafile_dir / ("llamafile.exe" if os.name == "nt" else "llamafile")
        self.models_dir = Path(self.config.models_dir)

        self.config.llamafile_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Corrected type hint for asyncio.subprocess.Process
        self._active_servers: dict[int, asyncio.subprocess.Process] = {}
        self._stream_tasks: dict[int, list[asyncio.Task]] = {}
        self._lifecycle_lock = asyncio.Lock()

        # Apply environment overrides
        handler_utils.apply_env_overrides(self.config)

        self._setup_signal_handlers()  # For cleaning up on exit
        self._cleanup_done = False  # Guard against duplicate cleanup
        self.metrics = {"starts": 0, "stops": 0, "start_errors": 0, "readiness_time_sum": 0.0, "readiness_count": 0}

    def _safe_log(self, level: str, msg: str, *args):
        """Log defensively to avoid errors when sinks are closed during atexit."""
        handler_utils.safe_log(self.logger, level, msg, *args)

    def _signal_managed_process_group(self, process: Any, sig: signal.Signals) -> int:
        """Signal a process group only when it is owned by a real managed subprocess."""
        if not isinstance(process, asyncio.subprocess.Process):
            raise ProcessLookupError("Refusing to signal process group for non-asyncio process")

        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 1:
            raise ProcessLookupError(f"Refusing unsafe process group signal for PID {pid!r}")

        pgid = os.getpgid(pid)
        if pgid != pid:
            raise PermissionError(f"Refusing to signal non-leader process group {pgid} for PID {pid}")
        if pgid == os.getpgrp():
            raise PermissionError("Refusing to signal current process group")

        os.killpg(pgid, sig)
        return pgid

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        is_windows = platform.system().lower().startswith("win")
        if is_windows:
            if hasattr(process, "terminate"):
                process.terminate()
        else:
            try:
                await asyncio.to_thread(self._signal_managed_process_group, process, signal.SIGTERM)
            except ProcessLookupError:
                if hasattr(process, "terminate"):
                    process.terminate()
            except _LLAMAFILE_NONCRITICAL_EXCEPTIONS:
                if hasattr(process, "terminate"):
                    process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            if is_windows:
                if hasattr(process, "kill"):
                    process.kill()
            else:
                try:
                    await asyncio.to_thread(self._signal_managed_process_group, process, signal.SIGKILL)
                except _LLAMAFILE_NONCRITICAL_EXCEPTIONS:
                    if hasattr(process, "kill"):
                        process.kill()
            with contextlib.suppress(_LLAMAFILE_NONCRITICAL_EXCEPTIONS):
                await process.wait()

    async def _drain_stream(self, stream, label: str) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    break
        except asyncio.CancelledError:
            return
        except _LLAMAFILE_NONCRITICAL_EXCEPTIONS:
            # Best-effort drain; ignore errors
            return

    def _start_stream_drainers(self, port: int, process: asyncio.subprocess.Process) -> None:
        tasks: list[asyncio.Task] = []
        if getattr(process, "stdout", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stdout, "stdout")))
        if getattr(process, "stderr", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stderr, "stderr")))
        if tasks:
            self._stream_tasks[port] = tasks

    def _stop_stream_drainers(self, port: int) -> None:
        tasks = self._stream_tasks.pop(port, [])
        for task in tasks:
            if not task.done():
                task.cancel()

    def _extract_llamafile_from_zip(self, zip_path: Path, output_path: Path) -> None:
        def _is_safe_member(member_name: str) -> bool:
            member_path = Path(member_name)
            return not member_path.is_absolute() and ".." not in member_path.parts

        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and _is_safe_member(m.filename)]
            if not members:
                raise ModelDownloadError("No suitable files found in llamafile zip archive.")

            def _pick_member():
                for member in members:
                    name = Path(member.filename).name.lower()
                    if name in {"llamafile", "llamafile.exe"}:
                        return member
                for member in members:
                    name = Path(member.filename).name.lower()
                    if "llamafile" in name:
                        return member
                if len(members) == 1:
                    return members[0]
                return None

            chosen = _pick_member()
            if not chosen:
                raise ModelDownloadError("No llamafile binary found in zip archive.")

            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extract(chosen, tmp_dir)
                extracted_path = Path(tmp_dir) / chosen.filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extracted_path), str(output_path))

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)

    def _is_port_free(self, host: str, port: int) -> bool:
        return handler_utils.is_port_free(host, port)

    def _pick_port(self, host: str, start_port: int) -> int:
        """Find an available port. Uses self._is_port_free for testability."""
        if not getattr(self.config, "port_autoselect", True):
            return start_port
        max_probe = int(getattr(self.config, "port_probe_max", 10) or 0)
        for i in range(max_probe + 1):
            candidate = start_port + i
            if self._is_port_free(host, candidate):
                return candidate
        return start_port  # Fallback

    def _denylist_check(self, args: dict[str, Any]):
        try:
            handler_utils.check_denylist(
                args,
                allow_secrets=getattr(self.config, "allow_cli_secrets", False),
            )
        except ValueError as e:
            raise ServerError(str(e)) from e

    def _is_path_allowed(self, p: Path) -> bool:
        """Check if path is under allowed directories."""
        base_dirs = handler_utils.build_allowed_paths(
            self.models_dir,
            getattr(self.config, "allowed_paths", None),
        )
        return handler_utils.is_path_allowed(p, base_dirs)

    # --- Llamafile Executable Management ---
    async def download_latest_llamafile_executable(self, force_download: bool = False) -> Path:
        """Downloads the latest llamafile binary if not present or if force_download is True."""
        output_path = self.llamafile_exe_path
        expected_sha256 = str(getattr(self.config, "executable_sha256", "") or "").strip().lower()
        self.logger.info(f"Checking for llamafile executable at {output_path}...")
        if output_path.exists() and not force_download:
            if expected_sha256 and not await asyncio.to_thread(verify_checksum, str(output_path), expected_sha256):
                raise ModelDownloadError("Existing llamafile executable failed checksum verification.")
            self.logger.debug(f"{output_path.name} already exists. Skipping download.")
            await asyncio.to_thread(os.chmod, output_path, 0o755)  # Ensure executable
            return output_path

        if not getattr(self.config, "allow_executable_auto_download", False):
            raise ModelDownloadError(
                "Llamafile executable auto-download is disabled. Configure a local executable or enable it explicitly."
            )
        if not expected_sha256:
            raise ModelDownloadError("Llamafile executable auto-download requires executable_sha256.")

        repo = "Mozilla-Ocho/llamafile"
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"

        from tldw_Server_API.app.core.Local_LLM.http_utils import async_stream_download, request_json

        async with create_async_client(timeout=60.0) as client:  # Increased timeout for fetching release info
            try:
                self.logger.debug(f"Fetching latest release info from {latest_release_url}")
                latest_release_data = await request_json(client, "GET", latest_release_url)
                tag_name = latest_release_data['tag_name']
                self.logger.debug(f"Latest release tag: {tag_name}")

                assets = latest_release_data.get('assets', [])
                asset_url = None
                chosen_asset_name = None
                system = platform.system().lower()
                is_windows = system.startswith("win")
                tag_hint = tag_name.lower().lstrip("v")
                machine = platform.machine().lower()

                os_tokens = {
                    "windows": {"windows", "win32", "win64"},
                    "darwin": {"darwin", "macos", "osx", "mac"},
                    "linux": {"linux", "ubuntu", "debian", "glibc", "musl"},
                }
                if system.startswith("win"):
                    os_positive = os_tokens["windows"]
                    os_negative = os_tokens["darwin"] | os_tokens["linux"]
                elif system == "darwin":
                    os_positive = os_tokens["darwin"]
                    os_negative = os_tokens["windows"] | os_tokens["linux"]
                else:
                    os_positive = os_tokens["linux"]
                    os_negative = os_tokens["windows"] | os_tokens["darwin"]

                arch_positive: set[str] = set()
                arch_negative: set[str] = set()
                if machine in {"x86_64", "amd64"} or "x86_64" in machine or "amd64" in machine:
                    arch_positive = {"x86_64", "amd64", "x64"}
                    arch_negative = {"arm64", "aarch64"}
                elif machine in {"arm64", "aarch64"} or "arm64" in machine or "aarch64" in machine:
                    arch_positive = {"arm64", "aarch64"}
                    arch_negative = {"x86_64", "amd64", "x64"}

                def _asset_score(asset_name: str) -> Optional[int]:
                    name = asset_name.lower()
                    if "debug" in name:
                        return None
                    if name.endswith((".sha256", ".sig", ".txt", ".tar.gz", ".tgz", ".tar")):
                        return None

                    score = 100
                    if "llamafile" in name:
                        score -= 30
                    if name.startswith("llamafile"):
                        score -= 10
                    if tag_hint and tag_hint in name:
                        score -= 5
                    if os_positive and any(token in name for token in os_positive):
                        score -= 25
                    elif os_negative and any(token in name for token in os_negative):
                        score += 200
                    if arch_positive and any(token in name for token in arch_positive):
                        score -= 15
                    elif arch_negative and any(token in name for token in arch_negative):
                        score += 200
                    if name.endswith(".exe"):
                        score += -40 if is_windows else 80
                    if name.endswith(".zip"):
                        score += -20 if is_windows else 40
                    if not is_windows and "." not in Path(name).name:
                        score -= 20
                    if "src" in name or "source" in name:
                        score += 200
                    return score

                candidates = []
                for asset in assets:
                    name = asset.get("name", "")
                    score = _asset_score(name)
                    if score is None:
                        continue
                    candidates.append((score, asset))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    asset_to_download = candidates[0][1]
                    asset_url = asset_to_download["browser_download_url"]
                    chosen_asset_name = asset_to_download["name"]

                if not asset_url:
                    self.logger.error(
                        f"No suitable llamafile asset found in release {tag_name}. Assets: {[a['name'] for a in assets]}")
                    raise ModelDownloadError(f"No suitable llamafile asset found for tag {tag_name}.")

                self.logger.info(
                    f"Found asset: {chosen_asset_name}. Downloading Llamafile from {http_utils.redacted_url(asset_url)} to {output_path}...")

                # Ensure the target directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Streaming download with retries
                if chosen_asset_name and chosen_asset_name.lower().endswith(".zip"):
                    tmp_zip_path = output_path.with_suffix(".zip")
                    try:
                        await async_stream_download(asset_url, str(tmp_zip_path))
                        await asyncio.to_thread(self._extract_llamafile_from_zip, tmp_zip_path, output_path)
                    finally:
                        tmp_zip_path.unlink(missing_ok=True)
                else:
                    await async_stream_download(asset_url, str(output_path))
                if not await asyncio.to_thread(verify_checksum, str(output_path), expected_sha256):
                    output_path.unlink(missing_ok=True)
                    raise ModelDownloadError("Checksum verification failed for downloaded llamafile executable.")
                await asyncio.to_thread(os.chmod, output_path, 0o755)
                self.logger.info(f"Downloaded {output_path.name} successfully.")
                return output_path

            except Exception as e:
                status = http_utils.get_http_status_from_exception(e)
                if status is not None:
                    error_text = http_utils.get_http_error_text(e)
                    self.logger.error(
                        f"Failed to fetch llamafile release info/download: {status} - {error_text}",
                        exc_info=True,
                    )
                    raise ModelDownloadError(f"Failed to fetch/download llamafile: {status}") from e
                self.logger.error(f"Unexpected error downloading llamafile: {e}", exc_info=True)
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise ModelDownloadError(f"Unexpected error downloading llamafile: {e}") from e

    # --- Model Management ---
    async def download_model_file(self, model_name: str, model_url: str, model_filename: Optional[str] = None,
                                  expected_hash: Optional[str] = None, force_download: bool = False) -> Path:
        """Downloads the specified LLM model file (.llamafile or .gguf)."""
        filename = model_filename or model_url.split('/')[-1].split('?')[0]
        if not filename or Path(str(filename)).name in {"", ".", ".."}:
            raise ModelDownloadError("Invalid model filename.")

        candidate = Path(filename)
        model_path = candidate if candidate.is_absolute() else (self.models_dir / candidate)
        if not self._is_path_allowed(model_path):
            raise ModelDownloadError("Model filename must resolve under allowed directories.")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Checking availability of model: {model_name} at {model_path}")
        if model_path.exists() and not force_download:
            if expected_hash:
                # project_utils.verify_checksum needs to be a real function
                is_valid = await asyncio.to_thread(verify_checksum, str(model_path), expected_hash)
                if not is_valid:
                    self.logger.warning(f"Checksum mismatch for existing model {model_path}. Re-downloading.")
                    model_path.unlink()
                else:
                    self.logger.debug(f"Model '{model_name}' ({filename}) already exists and checksum verified.")
                    return model_path
            else:
                self.logger.debug(
                    f"Model '{model_name}' ({filename}) already exists. Skipping download (no hash check).")
                return model_path

        self.models_dir.mkdir(parents=True, exist_ok=True)  # Ensure models dir exists
        self.logger.info(f"Downloading model: {model_name} from {http_utils.redacted_url(model_url)} to {model_path}")
        try:
            from tldw_Server_API.app.core.Local_LLM.http_utils import async_stream_download
            await async_stream_download(model_url, str(model_path))
            if expected_hash:
                is_valid = await asyncio.to_thread(verify_checksum, str(model_path), expected_hash)
                if not is_valid:
                    model_path.unlink(missing_ok=True)
                    raise ModelDownloadError("Checksum verification failed for downloaded model.")
            self.logger.info(f"Downloaded model '{model_name}' ({filename}) successfully.")
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}", exc_info=True)
            if model_path.exists():
                model_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"Failed to download model {model_name}: {e}") from e

    async def list_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []

        def _scan_dir():
            files = []
            for ext in ("*.gguf", "*.llamafile"):
                files.extend(self.models_dir.glob(ext))
            return [f.name for f in files]

        return await asyncio.to_thread(_scan_dir)

    async def is_model_available(self, model_filename: str) -> bool:
        return (self.models_dir / model_filename).is_file()  # Check if it's a file

    # --- Server Management ---
    async def start_server(self, model_filename: str, server_args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        async with self._lifecycle_lock:
            return await self._start_server_unlocked(model_filename, server_args=server_args)

    async def _start_server_unlocked(self, model_filename: str, server_args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        llamafile_exe = await self.download_latest_llamafile_executable()
        if not llamafile_exe or not llamafile_exe.exists():
            raise ServerError("Llamafile executable not found or could not be downloaded.")

        model_path = self.models_dir / model_filename
        if not self._is_path_allowed(model_path):
            raise ServerError("Model path must be under allowed directories.")
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file {model_filename} not found in {self.models_dir}.")

        args = {k: v for k, v in (server_args or {}).items() if v is not None and v != ""}
        host = args.get("host", self.config.default_host)
        if not host:
            host = self.config.default_host or "127.0.0.1"
        host = handler_utils.strip_host_brackets(str(host))
        client_host = handler_utils.resolve_client_host(host)

        def _coerce_port(value: Any, fallback: Any) -> int:
            if value is None or value == "":
                return int(fallback)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ServerError(f"Invalid port value: {value!r}") from exc

        port = self._pick_port(host, _coerce_port(args.get("port"), self.config.default_port))

        # Corrected check using .returncode for asyncio.subprocess.Process
        if port in self._active_servers and self._active_servers[port].returncode is None:
            active_pid = self._active_servers[port].pid
            self.logger.warning(
                f"Llamafile server already managed on port {port} with PID {active_pid}.")
            return {"status": "already_managed", "pid": active_pid, "port": port, "host": host,
                    "model": model_filename}

        # Allowlist of supported args
        allowed_formatters: dict[str, Any] = {
            "port": lambda v: ["--port", str(int(v))],
            "host": lambda v: ["--host", str(v)],
            "threads": lambda v: ["-t", str(int(v))],
            "threads_batch": lambda v: ["-tb", str(int(v))],
            "ctx_size": lambda v: ["-c", str(int(v))],
            "c": lambda v: ["-c", str(int(v))],
            "ngl": lambda v: ["-ngl", str(int(v))],
            "gpu_layers": lambda v: ["-ngl", str(int(v))],
            "batch_size": lambda v: ["-b", str(int(v))],
            "b": lambda v: ["-b", str(int(v))],
            "verbose": lambda v: (["-v"] if v else []),
            "api_key": lambda v: (["--api-key", str(v)] if v else []),
            # Boolean toggles
            "log_disable": lambda v: (["--log-disable"] if v else []),
            "memory_f32": lambda v: (["--memory-f32"] if v else []),
            "numa": lambda v: (["--numa"] if v else []),
            "sane_defaults": lambda v: (["--sane-defaults"] if v else []),
            # Extended safe flags commonly supported by llama.cpp-compatible CLIs
            "no_mmap": lambda v: (["--no-mmap"] if v else []),
            "mlock": lambda v: (["--mlock"] if v else []),
            "main_gpu": lambda v: ["--main-gpu", str(int(v))],
            "tensor_split": lambda v: ["--tensor-split", ",".join(map(str, v))] if isinstance(v, (list, tuple)) else ["--tensor-split", str(v)],
            "rope_freq_base": lambda v: ["--rope-freq-base", str(float(v))],
            "rope_freq_scale": lambda v: ["--rope-freq-scale", str(float(v))],
            # Parity with llama.cpp handler
            "main_kv": lambda v: ["--main-kv", str(int(v))],
            "no_kv_offload": lambda v: (["--no-kv-offload"] if v else []),
            "rope_scaling": lambda v: ["--rope-scaling", str(v)],
            "flash_attn": lambda v: (["--flash-attn"] if v else []),
            "cont_batching": lambda v: (["--cont-batching"] if v else []),
            "lora": lambda v: sum((["--lora", str(x)] for x in (v if isinstance(v, (list, tuple)) else [v])), []),
            "lora_base": lambda v: ["--lora-base", str(v)],
            "cache_type_k": lambda v: ["--cache-type-k", str(v)],
            "cache_type_v": lambda v: ["--cache-type-v", str(v)],
            "hf_token": lambda v: ["--hf-token", str(v)],
            "token": lambda v: ["--token", str(v)],
            # Paths
            "grammar_file": lambda v: ["--grammar-file", str(v)],
            "json_schema_file": lambda v: ["--json-schema-file", str(v)],
            "chat_template_file": lambda v: ["--chat-template-file", str(v)],
            "prompt_cache": lambda v: ["--prompt-cache", str(v)],
            "log_file": lambda v: ["--log-file", str(v)],
        }

        self._denylist_check(args)
        command = [str(llamafile_exe), "-m", str(model_path)]
        command += ["--port", str(port)]
        if host:
            command += ["--host", host]

        # Validate keys
        invalid = [k for k in args if k not in allowed_formatters]
        if invalid and not getattr(self.config, "allow_unvalidated_args", False):
            raise ServerError(f"Unsupported llamafile server args: {sorted(invalid)}")

        # Apply allowlisted flags (skip ones already included explicitly)
        for k, v in args.items():
            if k in ("port", "host"):
                continue
            if k in allowed_formatters:
                if k in {"grammar_file", "json_schema_file", "chat_template_file", "prompt_cache", "log_file", "lora_base"}:
                    if not self._is_path_allowed(Path(v)):
                        raise ServerError(f"File path for '{k}' must be under allowed directories.")
                if k == "lora":
                    vals = v if isinstance(v, (list, tuple)) else [v]
                    for item in vals:
                        if not self._is_path_allowed(Path(item)):
                            raise ServerError("LoRA path must be under allowed directories.")
                command += allowed_formatters[k](v)
            elif getattr(self.config, "allow_unvalidated_args", False):
                flag = f"--{k.replace('_', '-')}"
                if v is True:
                    command.append(flag)
                elif v is False or v is None:
                    pass
                else:
                    command += [flag, str(v)]

        redacted_cmd = redact_cmd_args(command)
        self.logger.info(f"Starting llamafile server for {model_filename} with command: {' '.join(redacted_cmd)}")

        try:
            # Using create_subprocess_exec for better security with list of args
            cpe_kwargs = {"stdout": asyncio.subprocess.PIPE, "stderr": asyncio.subprocess.PIPE}
            if platform.system() != "Windows":
                cpe_kwargs["preexec_fn"] = os.setsid
            else:
                cpe_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            t0 = time.perf_counter()
            process = await asyncio.create_subprocess_exec(*command, **cpe_kwargs)
            # Poll HTTP readiness instead of fixed sleep
            base_url = handler_utils.build_base_url(client_host, port)
            readiness_timeout = getattr(self.config, "readiness_timeout", 30.0) or 30.0
            is_ready = await wait_for_http_ready(
                base_url,
                timeout_total=readiness_timeout,
                interval=0.5,
                accept_any_non_5xx=True,
            )

            if process.returncode is not None or not is_ready:
                stderr_output = ""
                if process.stderr:
                    try:
                        # Use timeout to prevent blocking indefinitely if server is still writing
                        stderr_timeout = getattr(self.config, "stderr_read_timeout", 5.0) or 5.0
                        err_bytes = await asyncio.wait_for(process.stderr.read(), timeout=stderr_timeout)
                        stderr_output = err_bytes.decode(errors='ignore').strip()
                    except asyncio.TimeoutError:
                        stderr_output = "(stderr read timed out after 5s)"
                        self.logger.warning("stderr read timed out during startup failure diagnosis")
                self.logger.error(
                    f"Llamafile server failed to start or become ready for {model_filename}. Exit code: {process.returncode}. Stderr: {stderr_output}")
                stdout_output = ""
                if process.stdout:
                    try:
                        stderr_timeout = getattr(self.config, "stderr_read_timeout", 5.0) or 5.0
                        out_bytes = await asyncio.wait_for(process.stdout.read(), timeout=stderr_timeout)
                        stdout_output = out_bytes.decode(errors='ignore').strip()
                    except asyncio.TimeoutError:
                        stdout_output = "(stdout read timed out after 5s)"
                if stdout_output:
                    self.logger.error(f"Llamafile server stdout: {stdout_output}")
                await self._terminate_process(process)
                self.metrics["start_errors"] += 1
                raise ServerError(f"Llamafile server failed to start. Stderr: {stderr_output}")

            self._active_servers[port] = process
            t1 = time.perf_counter()
            self.metrics["starts"] += 1
            self.metrics["readiness_time_sum"] += max(0.0, t1 - t0)
            self.metrics["readiness_count"] += 1
            self._start_stream_drainers(port, process)
            self.logger.info(f"Llamafile server started for {model_filename} on port {port} with PID {process.pid}.")
            return {"status": "started", "pid": process.pid, "port": port, "host": host, "model": model_filename,
                    "command": ' '.join(redacted_cmd)}  # Return redacted command
        except Exception as e:
            self.logger.error(f"Exception starting llamafile server for {model_filename}: {e}", exc_info=True)
            raise ServerError(f"Exception starting llamafile: {e}") from e

    async def stop_server(self, port: Optional[int] = None, pid: Optional[int] = None) -> str:
        async with self._lifecycle_lock:
            return await self._stop_server_unlocked(port=port, pid=pid)

    async def _stop_server_unlocked(self, port: Optional[int] = None, pid: Optional[int] = None) -> str:
        process_to_stop: Optional[asyncio.subprocess.Process] = None
        port_to_clear = None

        if pid:
            for p, proc_obj in self._active_servers.items():
                if proc_obj.pid == pid:
                    process_to_stop = proc_obj
                    port_to_clear = p
                    break
            if not process_to_stop:
                return f"No managed llamafile server found with PID {pid}."
        elif port:
            if port in self._active_servers:
                process_to_stop = self._active_servers[port]
                port_to_clear = port
            else:
                return f"No managed llamafile server found on port {port}."
        else:
            return "Please provide a port or PID to stop a llamafile server."

        if not process_to_stop:
            return "No server matching criteria to stop."

        current_pid = process_to_stop.pid
        self.logger.info(f"Stopping llamafile server (PID: {current_pid}, Port: {port_to_clear or 'N/A'}).")
        try:
            # Check if still running using returncode
            if process_to_stop.returncode is None:
                await self._terminate_process(process_to_stop)
                self.logger.info(
                    f"Llamafile server PID {current_pid} terminated (return code: {process_to_stop.returncode}).")
            else:
                self.logger.info(
                    f"Llamafile server PID {current_pid} was already stopped (return code: {process_to_stop.returncode}).")

            if port_to_clear and port_to_clear in self._active_servers:
                self._stop_stream_drainers(port_to_clear)
                del self._active_servers[port_to_clear]
            return f"Llamafile server PID {current_pid} stopped."
        except Exception as e:
            self.logger.error(f"Error stopping llamafile server PID {current_pid}: {e}", exc_info=True)
            if port_to_clear and port_to_clear in self._active_servers:
                self._stop_stream_drainers(port_to_clear)
                del self._active_servers[port_to_clear]
            raise ServerError(f"Error stopping llamafile server: {e}") from e

    async def inference(self,
                        prompt: str,
                        port: Optional[int] = None,
                        host: Optional[str] = None,
                        system_prompt: Optional[str] = None,
                        n_predict: int = -1,
                        temperature: float = 0.8,
                        top_k: int = 40,
                        top_p: float = 0.95,
                        api_key: Optional[str] = None,
                        **kwargs) -> dict[str, Any]:
        if port is None:
            port = self.config.default_port
        if port is None:
            raise ServerError("Port is required for Llamafile inference.")
        target_host = host or self.config.default_host
        client_host = handler_utils.resolve_client_host(target_host)
        api_url = f"{handler_utils.build_base_url(client_host, port)}/v1/chat/completions"

        if port not in self._active_servers or self._active_servers[port].returncode is not None:
            self.logger.debug(
                f"Port {port} not in _active_servers or process terminated. Checking for external server responsiveness.")
            conn_made = False
            try:
                _, writer = await asyncio.open_connection(client_host, port)
                writer.close()
                await writer.wait_closed()
                conn_made = True
                self.logger.debug(f"Successfully connected to {client_host}:{port}. Assuming external server.")
            except ConnectionRefusedError:
                self.logger.exception(
                    f"No managed llamafile server on port {port} (or it terminated), and connection refused to {client_host}:{port}.")
                raise ServerError(f"Llamafile server not found or not responding on {client_host}:{port}.") from None
            except Exception as e:
                self.logger.error(f"Error checking connection to {client_host}:{port}: {e}", exc_info=True)
                raise ServerError(f"Error connecting to Llamafile server at {client_host}:{port}: {e}") from e
            if not conn_made:
                raise ServerError(f"Llamafile server not found/responding on {client_host}:{port} (conn test failed).")
        else:
            self.logger.debug(f"Using managed llamafile server on port {port}.")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        timeout = kwargs.pop("timeout", None)
        if "stream" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "stream"}
        payload = {
            "messages": messages, "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "n_predict": n_predict, **kwargs
        }
        payload["stream"] = False
        payload = {k: v for k, v in payload.items() if v is not None}

        msg_count = len(messages)
        self.logger.debug(
            f"Sending llamafile inference request to {api_url} (messages={msg_count}, n_predict={payload.get('n_predict')})"
        )
        http_timeout = timeout if timeout is not None else getattr(self.config, "http_timeout", 120.0)
        async with create_async_client(timeout=http_timeout) as client:
            try:
                result = await request_json(client, "POST", api_url, json=payload, headers=headers)
                self.logger.debug("Llamafile inference successful.")
                return result
            except Exception as e:
                status = http_utils.get_http_status_from_exception(e)
                if status is not None:
                    error_text = http_utils.get_http_error_text(e)
                    self.logger.error(
                        f"Llamafile API error ({status}) from {api_url}: {error_text}",
                        exc_info=True
                    )
                    raise InferenceError(f"Llamafile API error ({status}): {error_text}") from e
                if http_utils.is_network_error(e):
                    self.logger.error(
                        f"Could not connect or communicate with Llamafile server at {api_url}: {e}",
                        exc_info=True
                    )
                    raise ServerError(f"Could not connect/communicate with Llamafile server at {api_url}: {e}") from e
                self.logger.error(f"Unexpected error during llamafile inference to {api_url}: {e}", exc_info=True)
                raise InferenceError(f"Unexpected error during llamafile inference: {e}") from e

    def _cleanup_all_managed_servers_sync(self):
        """Synchronous cleanup for signal handlers or app shutdown.

        Uses _safe_log to avoid errors when logging sinks are closed during shutdown.
        """
        # Guard against duplicate cleanup
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self._safe_log("info", "Cleaning up all managed llamafile servers (sync)...")
        ports_to_remove = list(self._active_servers.keys())
        for port in ports_to_remove:
            proc = self._active_servers.get(port)
            # Check if proc exists and if its returncode is None (meaning it might be running)
            if proc and proc.returncode is None:
                pid = proc.pid
                self._safe_log("info", f"Stopping server on port {port}, PID {pid}...")
                try:
                    if platform.system() == "Windows":
                        proc.terminate()
                    else:
                        # For processes started with os.setsid, kill the process group
                        try:
                            pgid = self._signal_managed_process_group(proc, signal.SIGTERM)
                            self._safe_log("debug", f"Sent SIGTERM to process group {pgid} (leader PID {pid}).")
                        except ProcessLookupError:
                            self._safe_log("warning",
                                f"Process {pid} (or group) not found during SIGTERM, likely already terminated.")
                            if hasattr(proc, "terminate"):
                                proc.terminate()

                    # Best-effort wait: try to reap using running event loop if available
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(proc.wait(), loop)
                    except RuntimeError:
                        # No running event loop - best effort, OS will reap eventually
                        pass
                    self._safe_log("debug", f"Termination signal sent to PID {pid}. OS will handle reaping.")

                except ProcessLookupError:
                    self._safe_log("warning", f"Process {pid} not found during termination, likely already exited.")
                except _LLAMAFILE_NONCRITICAL_EXCEPTIONS as e:
                    self._safe_log("error", f"Error during cleanup of PID {pid}: {e}. Attempting kill.")
                    if proc.returncode is None:
                        if platform.system() == "Windows":
                            if hasattr(proc, "kill"):
                                with contextlib.suppress(_LLAMAFILE_NONCRITICAL_EXCEPTIONS):
                                    proc.kill()
                        else:
                            try:
                                self._signal_managed_process_group(proc, signal.SIGKILL)
                            except _LLAMAFILE_NONCRITICAL_EXCEPTIONS:
                                if hasattr(proc, "kill"):
                                    with contextlib.suppress(_LLAMAFILE_NONCRITICAL_EXCEPTIONS):
                                        proc.kill()
            if port in self._active_servers:
                self._stop_stream_drainers(port)
                del self._active_servers[port]
        self._safe_log("info", "Managed llamafile server synchronous cleanup complete.")

    def _signal_handler(self, sig, frame):
        self._safe_log("info", f'Signal handler called with signal: {sig}')
        self._cleanup_all_managed_servers_sync()
        sys.exit(0)

    def _setup_signal_handlers(self):
        import atexit
        atexit.register(self._cleanup_all_managed_servers_sync)
        self.logger.info("Registered atexit synchronous cleanup for LlamafileHandler.")
        # Signal handling can be tricky with asyncio and web servers.
        # Relying on atexit for this synchronous cleanup part is simpler for now.
        # For FastAPI, its own startup/shutdown events are better for managing async resources.

#
# # End of Llamafile_Handler.py
#########################################################################################################################
