# /tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py
# Description: Handler for Llama.cpp models, managing server processes and inference.
#
import asyncio
import os
import platform
import signal
import subprocess  # For synchronous fallback if needed
from pathlib import Path
from typing import List, Optional, Dict, Any
#
# Third-party imports
import httpx  # For making API calls to the Llama.cpp server
# from loguru import logger # Assuming you have a global logger or pass one
#
# Local imports
from .LLM_Base_Handler import BaseLLMHandler
from .LLM_Inference_Exceptions import ModelNotFoundError, ServerError, InferenceError
from .LLM_Inference_Schemas import LlamaCppConfig
# from .Utils import download_file, verify_checksum # If you need model downloading later
#########################################################################################################################
#
# Functions:

class LlamaCppHandler(BaseLLMHandler):
    def __init__(self, config: LlamaCppConfig, global_app_config: Dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: LlamaCppConfig  # For type hinting
        # self.logger = logger # Or use self.logger from BaseLLMHandler if already set

        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # For Llama.cpp, we usually manage one server instance that can have its model swapped.
        # If you need multiple concurrent Llama.cpp servers, this would be a Dict like in LlamafileHandler
        self._active_server_process: Optional[asyncio.subprocess.Process] = None
        self._active_server_model: Optional[str] = None
        self._active_server_port: Optional[int] = None
        self._active_server_host: Optional[str] = None

        self._setup_signal_handlers()  # For cleaning up on exit

    async def list_models(self) -> List[str]:
        """Lists locally available GGUF models."""
        if not self.models_dir.exists():
            return []

        def _scan_dir():
            return [f.name for f in self.models_dir.glob("*.gguf")]

        return await asyncio.to_thread(_scan_dir)

    async def is_model_available(self, model_filename: str) -> bool:
        """Checks if a GGUF model file is available locally."""
        return (self.models_dir / model_filename).is_file()

    # --- Server Management (Core of the swapping logic) ---
    async def start_server(self, model_filename: str, server_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Starts the Llama.cpp server with the specified model.
        If a server is already running managed by this handler, it will be stopped first (model swap).
        """
        if not Path(self.config.executable_path).is_file():
            raise ServerError(f"Llama.cpp server executable not found at {self.config.executable_path}")

        model_path = self.models_dir / model_filename
        if not model_path.is_file():
            raise ModelNotFoundError(f"Model file {model_filename} not found in {self.models_dir}.")

        # --- Model Swapping Logic ---
        if self._active_server_process and self._active_server_process.returncode is None:
            self.logger.info(
                f"Stopping existing Llama.cpp server (PID: {self._active_server_process.pid}) to swap model.")
            await self.stop_server()  # stop_server will clear _active_server_process etc.

        args = server_args or {}
        port = args.get("port", self.config.default_port)
        host = args.get("host", self.config.default_host)
        n_gpu_layers = args.get("n_gpu_layers", args.get("ngl", self.config.default_n_gpu_layers))
        ctx_size = args.get("ctx_size", args.get("c", self.config.default_ctx_size))
        threads = args.get("threads", args.get("t", self.config.default_threads))

        command = [
            str(self.config.executable_path),
            "-m", str(model_path),
            "--host", host,
            "--port", str(port),
            "-c", str(ctx_size),
            "-ngl", str(n_gpu_layers)
        ]
        if threads is not None:
            command.extend(["-t", str(threads)])

        # Add other boolean flags or key-value pairs from server_args
        # Example: --log-disable, --verbose, etc.
        if args.get("verbose", False): command.append("--verbose")  # or -v
        if args.get("log_disable", False): command.append("--log-disable")
        # ... add more custom args as needed based on Llama.cpp server options

        # For extra parameters not explicitly handled:
        extra_params = args.get("extra_params", [])  # e.g. ["--cont-batching", "--flash-attn"]
        command.extend(extra_params)

        self.logger.info(
            f"Starting Llama.cpp server for {model_filename} on {host}:{port} with command: {' '.join(command)}")

        stdout_redir = asyncio.subprocess.PIPE
        stderr_redir = asyncio.subprocess.PIPE
        log_file_handle = None

        if self.config.log_output_file:
            try:
                log_file_handle = open(self.config.log_output_file, "ab")  # Append binary
                stdout_redir = log_file_handle
                stderr_redir = log_file_handle
                self.logger.info(f"Llama.cpp server logs will be written to: {self.config.log_output_file}")
            except Exception as e:
                self.logger.error(f"Could not open log file {self.config.log_output_file}: {e}. Logging to PIPE.")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=stdout_redir,
                stderr=stderr_redir,
                preexec_fn=os.setsid if platform.system() != "Windows" else None
            )

            # Brief pause to let the server initialize. A better way is to poll a health endpoint.
            # Llama.cpp server prints "server listening at http://0.0.0.0:8080"
            # You could try to read its stdout/stderr for this message if not redirecting to file.
            await asyncio.sleep(5)  # Adjust as needed, or implement polling

            if process.returncode is not None:
                # If logging to file, error might not be in stderr pipe here.
                # Consider reading last few lines of log_output_file if it exists.
                stderr_output = ""
                if stderr_redir == asyncio.subprocess.PIPE and process.stderr:  # Check if stderr was piped
                    err_bytes = await process.stderr.read()  # This might hang if server is still writing
                    stderr_output = err_bytes.decode(errors='ignore').strip()
                self.logger.error(
                    f"Llama.cpp server failed to start for {model_filename}. Exit code: {process.returncode}. Stderr: {stderr_output}"
                )
                if log_file_handle: log_file_handle.close()
                raise ServerError(f"Llama.cpp server failed to start. Stderr: {stderr_output}")

            self._active_server_process = process
            self._active_server_model = model_filename
            self._active_server_port = port
            self._active_server_host = host
            if log_file_handle:  # Store handle to close it later
                self._active_server_log_handle = log_file_handle
            else:
                self._active_server_log_handle = None

            self.logger.info(f"Llama.cpp server started for {model_filename} on {host}:{port} with PID {process.pid}.")
            return {"status": "started", "pid": process.pid, "model": model_filename, "port": port, "host": host,
                    "command": ' '.join(command)}
        except Exception as e:
            if log_file_handle: log_file_handle.close()
            self.logger.error(f"Exception starting Llama.cpp server for {model_filename}: {e}", exc_info=True)
            raise ServerError(f"Exception starting Llama.cpp server: {e}")

    async def stop_server(self) -> str:
        if not self._active_server_process:
            return "No Llama.cpp server managed by this handler is currently running."

        process_to_stop = self._active_server_process
        pid = process_to_stop.pid
        model_name = self._active_server_model
        self.logger.info(f"Stopping Llama.cpp server (PID: {pid}, Model: {model_name}).")

        try:
            if process_to_stop.returncode is None:  # Still running
                if platform.system() == "Windows":
                    process_to_stop.terminate()
                else:
                    try:
                        pgid = await asyncio.to_thread(os.getpgid, pid)
                        await asyncio.to_thread(os.killpg, pgid, signal.SIGTERM)
                        self.logger.info(f"Sent SIGTERM to process group {pgid} (leader PID: {pid}).")
                    except ProcessLookupError:
                        self.logger.warning(f"Process {pid} not found for SIGTERM, likely already terminated.")
                        process_to_stop.terminate()  # Fallback
                    except Exception as e_pg:
                        self.logger.warning(
                            f"Failed to send SIGTERM to process group {pid}: {e_pg}. Falling back to PID.")
                        process_to_stop.terminate()

                try:
                    await asyncio.wait_for(process_to_stop.wait(), timeout=10)
                    self.logger.info(f"Llama.cpp server PID {pid} (Model: {model_name}) terminated gracefully.")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Llama.cpp server PID {pid} (Model: {model_name}) did not terminate gracefully. Killing.")
                    if platform.system() == "Windows":
                        process_to_stop.kill()
                    else:
                        try:
                            pgid = await asyncio.to_thread(os.getpgid, pid)  # Re-fetch pgid in case
                            await asyncio.to_thread(os.killpg, pgid, signal.SIGKILL)
                        except:  # Broad except for kill fallback
                            process_to_stop.kill()
                    await process_to_stop.wait()
            else:
                self.logger.info(
                    f"Llama.cpp server PID {pid} (Model: {model_name}) was already stopped (return code: {process_to_stop.returncode}).")

            if self._active_server_log_handle:
                self._active_server_log_handle.close()
                self._active_server_log_handle = None

            self._active_server_process = None
            self._active_server_model = None
            self._active_server_port = None
            self._active_server_host = None
            return f"Llama.cpp server PID {pid} (Model: {model_name}) stopped."

        except Exception as e:
            self.logger.error(f"Error stopping Llama.cpp server PID {pid}: {e}", exc_info=True)
            # Clear state even on error to allow trying to start a new one
            if self._active_server_log_handle:
                self._active_server_log_handle.close()
                self._active_server_log_handle = None
            self._active_server_process = None
            self._active_server_model = None
            self._active_server_port = None
            self._active_server_host = None
            raise ServerError(f"Error stopping Llama.cpp server: {e}")

    async def get_server_status(self) -> Dict[str, Any]:
        if self._active_server_process and self._active_server_process.returncode is None:
            return {
                "status": "running",
                "pid": self._active_server_process.pid,
                "model": self._active_server_model,
                "port": self._active_server_port,
                "host": self._active_server_host,
                "log_file": str(
                    self.config.log_output_file) if self.config.log_output_file and self._active_server_log_handle else None
            }
        return {"status": "stopped", "model": None, "pid": None, "port": None, "host": None}

    async def inference(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None,
                        api_endpoint: str = "/v1/chat/completions",  # or /completion
                        **kwargs) -> Dict[str, Any]:
        if not self._active_server_process or self._active_server_process.returncode is not None:
            raise ServerError("Llama.cpp server is not running or not managed by this handler.")

        base_url = f"http://{self._active_server_host}:{self._active_server_port}"
        target_url = f"{base_url}{api_endpoint.lstrip('/')}"  # Ensure single slash

        # Prepare payload (OpenAI compatible)
        payload = kwargs.copy()  # n_predict, temperature, top_k, top_p, stop, etc.
        if messages:
            payload["messages"] = messages
        elif prompt:  # Convert simple prompt to messages for chat completions endpoint
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            raise InferenceError("Either 'prompt' or 'messages' must be provided for inference.")

        if "stream" not in payload:  # Default to non-streaming for this method
            payload["stream"] = False

        self.logger.debug(f"Sending Llama.cpp inference request to {target_url} with payload: {payload}")

        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 120.0)) as client:
            try:
                response = await client.post(target_url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                result = response.json()
                self.logger.debug("Llama.cpp inference successful.")
                return result
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                self.logger.error(f"Llama.cpp API error ({e.response.status_code}) from {target_url}: {error_text}",
                                  exc_info=True)
                raise InferenceError(f"Llama.cpp API error ({e.response.status_code}): {error_text}")
            except httpx.RequestError as e:
                self.logger.error(f"Could not connect or communicate with Llama.cpp server at {target_url}: {e}",
                                  exc_info=True)
                raise ServerError(f"Could not connect/communicate with Llama.cpp server at {target_url}: {e}")

    # --- Cleanup ---
    def _cleanup_managed_server_sync(self):
        self.logger.info("Cleaning up managed Llama.cpp server (sync)...")
        if self._active_server_process and self._active_server_process.returncode is None:
            proc = self._active_server_process
            pid = proc.pid
            self.logger.info(
                f"Stopping Llama.cpp server (PID: {pid}, Model: {self._active_server_model}) synchronously...")
            try:
                if platform.system() == "Windows":
                    # Use subprocess for synchronous Popen if asyncio process is tricky here
                    # Or just call terminate on the asyncio process if it works in sync context
                    proc.terminate()
                else:
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        self.logger.warning(f"Process {pid} (or group) not found during sync SIGTERM.")
                        proc.terminate()  # Fallback
                    except Exception:  # Broad except for pgid issues
                        proc.terminate()

                # Synchronous wait is tricky with asyncio.subprocess.Process.
                # For atexit, sending TERM/KILL is often the best effort.
                # If you need guaranteed wait, you might need to use `subprocess.Popen` for the server.
                self.logger.info(f"Termination signal sent to Llama.cpp server PID {pid}. OS will handle reaping.")

            except Exception as e:
                self.logger.error(f"Error during synchronous cleanup of PID {pid}: {e}. Attempting kill.")
                if proc.returncode is None:  # Check again before kill
                    if platform.system() == "Windows":
                        proc.kill()
                    else:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        except:
                            proc.kill()
            if self._active_server_log_handle:
                self._active_server_log_handle.close()

        self._active_server_process = None  # Clear state
        self.logger.info("Managed Llama.cpp server synchronous cleanup attempt complete.")

    def _signal_handler(self, sig, frame):
        self.logger.info(f'Signal handler called with signal: {sig} for LlamaCppHandler')
        self._cleanup_managed_server_sync()
        # Let other handlers run, don't sys.exit here unless it's the main app's job
        # sys.exit(0)

    def _setup_signal_handlers(self):
        import atexit
        # Register for process exit
        atexit.register(self._cleanup_managed_server_sync)
        self.logger.info("Registered atexit synchronous cleanup for LlamaCppHandler.")
        # Signal handling for Ctrl+C etc.
        # Note: Multiple signal handlers can be tricky. Ensure this doesn't conflict.
        # signal.signal(signal.SIGINT, self._signal_handler)
        # signal.signal(signal.SIGTERM, self._signal_handler)

#
# End of LlamaCpp_Handler.py
########################################################################################################################
