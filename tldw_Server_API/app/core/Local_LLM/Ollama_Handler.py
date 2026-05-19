# Ollama_Handler.py
# Description: Handler for Ollama LLM backend.
#
# Imports
#
# Third-party Imports
import asyncio
import os
import shutil
from typing import Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.Local_LLM import handler_utils, http_utils
from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError, ModelDownloadError, ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import OllamaConfig

_OLLAMA_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)

_OLLAMA_PROCESS_CONTROL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    ProcessLookupError,
    asyncio.TimeoutError,
)

_OLLAMA_REQUEST_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    NetworkError,
    RetryExhaustedError,
)

_OLLAMA_PSUTIL_EXCEPTIONS = _OLLAMA_NONCRITICAL_EXCEPTIONS
if PSUTIL_AVAILABLE:
    _OLLAMA_PSUTIL_EXCEPTIONS = _OLLAMA_NONCRITICAL_EXCEPTIONS + (psutil.Error,)


def create_async_client(*args, **kwargs):
    """Proxy AsyncClient factory so tests can patch either module."""
    return http_utils.create_async_client(*args, **kwargs)


async def request_json(*args, **kwargs):
    """Proxy request_json helper for graceful monkeypatching."""
    return await http_utils.request_json(*args, **kwargs)


async def wait_for_http_ready(base_url: str, timeout_total: float = 30.0, interval: float = 0.5):
    """Ollama readiness uses general HTTP readiness with Ollama-friendly paths."""
    return await http_utils.wait_for_http_ready(
        base_url,
        paths=("/api/version", "/api/tags"),
        timeout_total=timeout_total,
        interval=interval,
    )


#######################################################################################################################
#
# Functions:

class OllamaHandler(BaseLLMHandler):
    def __init__(self, config: OllamaConfig, global_app_config: dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: OllamaConfig  # For type hinting
        self._serve_process: Optional[asyncio.subprocess.Process] = None
        self._serve_stream_tasks: list[asyncio.Task] = []

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
        except _OLLAMA_NONCRITICAL_EXCEPTIONS:
            # Best-effort drain; ignore errors
            return

    def _start_stream_drainers(self, process: asyncio.subprocess.Process) -> None:
        tasks: list[asyncio.Task] = []
        if getattr(process, "stdout", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stdout, "stdout")))
        if getattr(process, "stderr", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stderr, "stderr")))
        if tasks:
            self._serve_stream_tasks = tasks

    def _stop_stream_drainers(self) -> None:
        for task in self._serve_stream_tasks:
            if not task.done():
                task.cancel()
        self._serve_stream_tasks = []

    def _clear_serve_process(self, pid: Optional[int] = None) -> None:
        if self._serve_process and (pid is None or self._serve_process.pid == pid):
            self._stop_stream_drainers()
            self._serve_process = None
    async def is_ollama_installed(self) -> bool:
        """Checks if the 'ollama' executable is available."""
        return await asyncio.to_thread(shutil.which, 'ollama') is not None

    async def list_models(self) -> list[str]:
        """Retrieves available Ollama models."""
        if not await self.is_ollama_installed():
            self.logger.error("Ollama executable not found.")
            return []
        try:
            stdout, _ = await self._run_subprocess(['ollama', 'list'])
            models = stdout.strip().split('\n')
            if not models or models[0].strip().upper().startswith("NAME"):  # Skip header
                models = models[1:]
            model_names = [model.split()[0] for model in models if model.strip()]
            self.logger.debug(f"Available Ollama models: {model_names}")
            return model_names
        except ServerError as e:  # Catching generic server error from _run_subprocess
            self.logger.exception(f"Error executing Ollama 'list': {e}")
            return []
        except _OLLAMA_NONCRITICAL_EXCEPTIONS as e:
            self.logger.exception(f"Unexpected error in get_ollama_models: {e}")
            return []

    async def is_model_available(self, model_name: str) -> bool:
        models = await self.list_models()
        return model_name in models

    async def pull_model(self, model_name: str, timeout: int = 300) -> str:
        """Pulls the specified Ollama model."""
        if not await self.is_ollama_installed():
            msg = "Ollama is not installed."
            self.logger.error(msg)
            raise ModelDownloadError(msg)

        self.logger.info(f"Pulling Ollama model: {model_name}")
        try:
            # subprocess.run with timeout is blocking, use asyncio.create_subprocess_exec for better async
            process = await asyncio.create_subprocess_exec(
                'ollama', 'pull', model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Stream output or just wait
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

            if process.returncode == 0:
                self.logger.info(f"Successfully pulled model: {model_name}")
                return f"Successfully pulled model: {model_name}"
            else:
                err_msg = stderr.decode().strip() if stderr else "Unknown error during pull."
                self.logger.error(f"Failed to pull model '{model_name}': {err_msg}")
                raise ModelDownloadError(f"Failed to pull model '{model_name}': {err_msg}")

        except asyncio.TimeoutError:
            self.logger.exception(f"Pulling model '{model_name}' timed out after {timeout}s.")
            # Attempt to terminate the process if it's still running
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except _OLLAMA_PROCESS_CONTROL_EXCEPTIONS as e_term:
                    self.logger.exception(f"Error terminating timed-out ollama pull process: {e_term}")
                    process.kill()  # Force kill if terminate fails
                    await process.wait()
            raise ModelDownloadError(f"Failed to pull model '{model_name}': Operation timed out.") from None
        except _OLLAMA_NONCRITICAL_EXCEPTIONS as e:
            self.logger.exception(f"Unexpected error in pull_ollama_model: {e}")
            raise ModelDownloadError(f"Failed to pull model '{model_name}': {e}") from e

    async def serve_model(self, model_name: str, port: Optional[int] = None, host: str = "127.0.0.1") -> dict[str, Any]:
        """
        Serves the specified Ollama model.
        Ollama's `ollama serve` command starts a general server, not specific to one model.
        It will load models on demand. The `ollama run <model>` or API calls will use the served models.
        This function will ensure `ollama serve` is running.
        Returns a dictionary with server status and PID if started.
        """
        if not await self.is_ollama_installed():
            msg = "Ollama is not installed."
            self.logger.error(msg)
            raise ServerError(msg)

        port = port or self.config.default_port
        client_host = handler_utils.resolve_client_host(host)
        ollama_env = os.environ.copy()
        ollama_env["OLLAMA_HOST"] = f"{host}:{port}"

        # Check if ollama serve is already running with the specified host/port
        # Use retry loop to handle race conditions in port detection
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available; skipping port check before starting server")
        else:
            max_port_check_retries = getattr(self.config, "port_check_retries", 3)
            for attempt in range(max_port_check_retries):
                try:
                    port_in_use = await asyncio.to_thread(psutil.net_connections)
                    for conn in port_in_use:
                        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == int(port):
                            # Port is in use - check if it's ollama or another process
                            try:
                                proc = psutil.Process(conn.pid)
                                if "ollama" in proc.name().lower():
                                    self.logger.warning(
                                        f"Ollama server seems to be already running on port {port} (PID: {conn.pid}).")
                                    return {"status": "already_running", "pid": conn.pid, "host": host, "port": port}
                                else:
                                    # Non-ollama process is using the port
                                    self.logger.warning(f"Port {port} is already in use by another process ({proc.name()}).")
                                    return {"status": "port_in_use", "pid": conn.pid, "host": host, "port": port}
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                # Couldn't get process info, but port is still in use
                                self.logger.warning(f"Port {port} is already in use (couldn't identify process).")
                                return {"status": "port_in_use",
                                        "pid": conn.pid if hasattr(conn, 'pid') else None, "host": host,
                                        "port": port}
                    # Port appears free, break out of retry loop
                    break
                except _OLLAMA_PSUTIL_EXCEPTIONS as e:
                    self.logger.warning(f"Could not check port status (attempt {attempt + 1}): {e}")
                    if attempt < max_port_check_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Backoff

        self.logger.info(f"Starting Ollama server on {host}:{port}. Models will be loaded on demand.")
        try:
            # Start the Ollama server. It daemonizes by default on some systems.
            # For library use, running it explicitly and capturing PID is better if possible.
            # `ollama serve` itself often detaches.
            # We might need a wrapper or expect it to be run via systemd.
            # For now, we'll launch it. If it detaches, stopping it by PID here is hard.
            cmd = ['ollama', 'serve']
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=ollama_env,
                stdout=asyncio.subprocess.PIPE,  # Or DEVNULL if we don't care about its output after start
                stderr=asyncio.subprocess.PIPE
            )
            # Confirm server is up by polling common Ollama endpoints
            ready = await wait_for_http_ready(
                handler_utils.build_base_url(client_host, port),
                timeout_total=30.0,
                interval=0.5,
            )

            if process.returncode is not None and process.returncode != 0:
                stderr_output = (await process.stderr.read()).decode() if process.stderr else "Unknown error"
                self.logger.error(f"Failed to start Ollama server: {stderr_output}")
                raise ServerError(f"Failed to start Ollama server: {stderr_output}")

            if not ready:
                stderr_output = ""
                if process.returncode is not None and process.stderr:
                    stderr_output = (await process.stderr.read()).decode() or "Unknown error"
                self.logger.error(
                    f"Ollama server did not become ready on {host}:{port}. {stderr_output}".strip()
                )
                if process.returncode is None:
                    try:
                        process.terminate()
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except _OLLAMA_PROCESS_CONTROL_EXCEPTIONS:
                        try:
                            process.kill()
                            await process.wait()
                        except _OLLAMA_PROCESS_CONTROL_EXCEPTIONS:
                            pass
                raise ServerError("Ollama server did not become ready in time.")

            # Try to find the PID if it daemonized
            # This is OS-dependent and fragile.
            # Best if `ollama serve` had a `--no-daemon` and `--pidfile` option.
            pid = process.pid
            self.logger.info(f"Ollama server process started with PID {pid} on {host}:{port}. May run in background.")
            self._serve_process = process
            self._start_stream_drainers(process)
            return {"status": "started", "pid": pid, "host": host, "port": port}

        except FileNotFoundError:
            msg = "Ollama executable not found."
            self.logger.exception(msg)
            raise ServerError(msg) from None
        except _OLLAMA_NONCRITICAL_EXCEPTIONS as e:
            self.logger.exception(f"Error starting Ollama server: {e}")
            raise ServerError(f"Error starting Ollama server: {e}") from e

    async def stop_server(self, pid: Optional[int] = None, port: Optional[int] = None) -> str:
        """
        Stops the Ollama server.
        If PID is given, it attempts to terminate that specific process.
        If port is given, it attempts to find and terminate the process listening on that port.
        Stopping `ollama serve` can be tricky as it might be managed by systemd.
        This function primarily targets processes started by this library or manually.
        """
        if not await self.is_ollama_installed():
            return "Ollama is not installed."

        if pid:
            self.logger.info(f"Attempting to stop Ollama server with PID {pid}")
            try:
                await asyncio.to_thread(self._terminate_process, pid)
                self._clear_serve_process(pid)
                return f"Attempted to stop Ollama server with PID {pid}"
            except ProcessLookupError:
                self.logger.warning(f"No process found with PID {pid}")
                return f"No process found with PID {pid}"
            except _OLLAMA_NONCRITICAL_EXCEPTIONS as e:
                self.logger.error(f"Error stopping Ollama server PID {pid}")
                return f"Error stopping Ollama server PID {pid}"
        elif port:
            if not PSUTIL_AVAILABLE:
                return "Cannot stop by port: psutil not available. Please provide PID instead."
            self.logger.info(f"Attempting to stop Ollama server listening on port {port}")
            found_pid = None
            try:
                for conn in await asyncio.to_thread(psutil.net_connections):
                    if conn.status == psutil.CONN_LISTEN and conn.laddr.port == int(port):
                        if conn.pid:
                            proc_info = await asyncio.to_thread(psutil.Process, conn.pid)
                            if "ollama" in proc_info.name().lower():
                                found_pid = conn.pid
                                break
                if found_pid:
                    await asyncio.to_thread(self._terminate_process, found_pid)
                    self._clear_serve_process(found_pid)
                    return f"Attempted to stop Ollama server (PID {found_pid}) on port {port}"
                else:
                    return f"No Ollama server found listening on port {port}"
            except _OLLAMA_PSUTIL_EXCEPTIONS as e:
                self.logger.error(f"Error stopping Ollama server on port {port}")
                return f"Error stopping Ollama server on port {port}"
        else:
            # General stop command `ollama stop` - this might not exist or work as expected for `ollama serve`
            # `ollama ps` and then finding the server PID might be more reliable if `ollama stop` isn't for `serve`
            self.logger.info("Attempting to stop the main Ollama application/service (if running).")
            # The 'ollama stop' command usually refers to stopping a model being run, not 'ollama serve'
            # For 'ollama serve', typically pkill or systemctl is used if managed.
            # This is a best-effort and might not stop a detached `ollama serve`.
            # stdout, stderr = await self._run_subprocess(['ollama', 'stop']) # This command does not exist
            # self.logger.info(f"Ollama stop command output: {stdout}")
            return "General 'ollama stop' for a server is not a standard command. Please provide PID or manage via system services."

    def _terminate_process(self, pid: int):
        """Helper to terminate a process by PID. Requires psutil."""
        if not PSUTIL_AVAILABLE:
            raise ServerError("psutil not available; cannot terminate process by PID")
        try:
            proc = psutil.Process(pid)
            proc.terminate()  # Send SIGTERM
            self.logger.info(f"Sent SIGTERM to process {pid}")
            try:
                proc.wait(timeout=5)  # Wait for termination
                self.logger.info(f"Process {pid} terminated gracefully.")
            except psutil.TimeoutExpired:
                self.logger.warning(f"Process {pid} did not terminate after SIGTERM, sending SIGKILL.")
                proc.kill()  # Send SIGKILL
                proc.wait(timeout=5)
                self.logger.info(f"Process {pid} killed.")
        except psutil.NoSuchProcess:
            raise ProcessLookupError(f"No process found with PID {pid}") from None
        except _OLLAMA_PSUTIL_EXCEPTIONS as e:
            raise ServerError(f"Failed to terminate process {pid}: {e}") from e

    async def inference(self, model_name: str, prompt: str, system_message: Optional[str] = None,
                        port: Optional[int] = None, host: str = "127.0.0.1",
                        options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Performs inference using a model served by Ollama.
        Assumes `ollama serve` is running or starts it.
        Uses the Ollama REST API.
        """
        port = port or self.config.default_port
        client_host = handler_utils.resolve_client_host(host)
        api_url = f"{handler_utils.build_base_url(client_host, port)}/api/generate"

        # Pull model if not available, with configurable retry logic
        if not await self.is_model_available(model_name):
            max_pull_retries = getattr(self.config, "max_pull_retries", 2)
            last_error = None
            for attempt in range(max_pull_retries):
                try:
                    self.logger.info(f"Model {model_name} not available locally. Pulling (attempt {attempt + 1}/{max_pull_retries})...")
                    await self.pull_model(model_name)
                    last_error = None
                    break
                except ModelDownloadError as e:
                    last_error = e
                    if attempt < max_pull_retries - 1:
                        backoff_time = 1.0 * (attempt + 1)
                        self.logger.warning(f"Pull attempt {attempt + 1} failed: {e}. Retrying in {backoff_time}s...")
                        await asyncio.sleep(backoff_time)
            if last_error:
                raise InferenceError(f"Model {model_name} not found and could not be pulled after {max_pull_retries} attempts: {last_error}")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False  # For non-streaming response
        }
        if system_message:
            payload["system"] = system_message
        if options:
            payload["options"] = options

        prompt_len = len(prompt) if isinstance(prompt, str) else 0
        options_keys = list(options.keys()) if isinstance(options, dict) else []
        self.logger.debug(
            f"Sending Ollama inference request to {api_url} (model={model_name}, prompt_len={prompt_len}, "
            f"system={'yes' if system_message else 'no'}, options_keys={options_keys})"
        )
        async with create_async_client() as client:
            try:
                result = await request_json(client, "POST", api_url, json=payload)
                self.logger.debug(f"Ollama inference successful for {model_name}.")
                return result
            except _OLLAMA_REQUEST_EXCEPTIONS as e:
                status = http_utils.get_http_status_from_exception(e)
                if status is not None:
                    text = http_utils.get_http_error_text(e)
                    self.logger.exception(f"Ollama API error ({status}): {text}")
                    # Attempt model pull on 404 or message indicating not found
                    if status == 404 or (isinstance(text, str) and "model not found" in text.lower()):
                        self.logger.info(f"Model {model_name} not found on server, attempting to pull.")
                        try:
                            await self.pull_model(model_name)
                            # retry once
                            return await request_json(client, "POST", api_url, json=payload)
                        except ModelDownloadError as e_pull:
                            raise InferenceError(
                                f"Model {model_name} could not be pulled: {e_pull}. Original API error: {text}") from e_pull
                    raise InferenceError(f"Ollama API error ({status}): {text}") from e
                # Likely connection error; attempt to start server then retry
                self.logger.exception(f"Could not connect to Ollama server at {api_url}: {e}")
                self.logger.info("Attempting to start Ollama server...")
                try:
                    await self.serve_model(model_name, port=port, host=host)
                    ready = await wait_for_http_ready(
                        handler_utils.build_base_url(client_host, port),
                        timeout_total=30.0,
                        interval=0.5,
                    )
                    if not ready:
                        raise InferenceError("Ollama server did not become ready in time")
                    return await request_json(client, "POST", api_url, json=payload)
                except ServerError as se:
                    raise InferenceError(f"Could not start or connect to Ollama server: {se}") from se
                except _OLLAMA_REQUEST_EXCEPTIONS as e_retry:
                    raise InferenceError(f"Failed to perform inference after server start attempt: {e_retry}") from e_retry

#
# End of Ollama_Handler.py
#######################################################################################################################
