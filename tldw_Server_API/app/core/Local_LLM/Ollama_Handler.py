# Ollamafile_Handler.py
# Description:
#
# Imports
import platform
import subprocess
import psutil
import os
import signal
import shutil
from typing import List, Optional, Dict, Any
#
# Third-party Imports
import asyncio

from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import (
    ServerError,
    ModelDownloadError,
    InferenceError
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import OllamaConfig


#
# Local Imports

# from .base_handler import BaseLLMHandler # Use if BaseLLMHandler is in a separate file
# from .exceptions import ModelNotFoundError, ModelDownloadError, ServerError # Use if exceptions are separate
# from .utils_loader import logging, project_utils # From the loader
#
#######################################################################################################################
#
# Functions:

class OllamaHandler(BaseLLMHandler):
    def __init__(self, config: OllamaConfig, global_app_config: Dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: OllamaConfig  # For type hinting

    async def is_ollama_installed(self) -> bool:
        """Checks if the 'ollama' executable is available."""
        return await asyncio.to_thread(shutil.which, 'ollama') is not None

    async def list_models(self) -> List[str]:
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
            self.logger.error(f"Error executing Ollama 'list': {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in get_ollama_models: {e}")
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
            self.logger.error(f"Pulling model '{model_name}' timed out after {timeout}s.")
            # Attempt to terminate the process if it's still running
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except Exception as e_term:
                    self.logger.error(f"Error terminating timed-out ollama pull process: {e_term}")
                    process.kill()  # Force kill if terminate fails
                    await process.wait()
            raise ModelDownloadError(f"Failed to pull model '{model_name}': Operation timed out.")
        except Exception as e:
            self.logger.error(f"Unexpected error in pull_ollama_model: {e}")
            raise ModelDownloadError(f"Failed to pull model '{model_name}': {e}")

    async def serve_model(self, model_name: str, port: Optional[int] = None, host: str = "127.0.0.1") -> Dict[str, Any]:
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
        ollama_env = os.environ.copy()
        ollama_env["OLLAMA_HOST"] = f"{host}:{port}"

        # Check if ollama serve is already running with the specified host/port
        # This is a bit tricky as `ollama serve` might be managed by systemd or run manually.
        # For simplicity, we'll check if the port is in use.
        # A more robust check would involve inspecting running 'ollama serve' processes.
        try:
            port_in_use = await asyncio.to_thread(psutil.net_connections)
            for conn in port_in_use:
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == int(port):
                    # Potentially check if the process is an ollama process
                    try:
                        proc = psutil.Process(conn.pid)
                        if "ollama" in proc.name().lower():
                            self.logger.warning(
                                f"Ollama server seems to be already running on port {port} (PID: {conn.pid}).")
                            return {"status": "already_running", "pid": conn.pid, "host": host, "port": port}
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # Process might have ended or we don't have permission
                    self.logger.warning(f"Port {port} is already in use. Assuming Ollama server or other service.")
                    # Not raising an error, as it might be an externally managed Ollama server
                    return {"status": "port_in_use",
                            "pid": conn.pid if 'conn' in locals() and hasattr(conn, 'pid') else None, "host": host,
                            "port": port}

        except Exception as e:
            self.logger.warning(f"Could not check port status: {e}")

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
            # `ollama serve` might run in the background quickly.
            # We need a way to confirm it's up, e.g., by polling its health endpoint.
            # This is a simplified start; a robust version would poll health.
            await asyncio.sleep(2)  # Give it a moment to start

            if process.returncode is not None and process.returncode != 0:
                stderr_output = (await process.stderr.read()).decode() if process.stderr else "Unknown error"
                self.logger.error(f"Failed to start Ollama server: {stderr_output}")
                raise ServerError(f"Failed to start Ollama server: {stderr_output}")

            # Try to find the PID if it daemonized
            # This is OS-dependent and fragile.
            # Best if `ollama serve` had a `--no-daemon` and `--pidfile` option.
            pid = process.pid
            self.logger.info(f"Ollama server process started with PID {pid} on {host}:{port}. May run in background.")
            return {"status": "started", "pid": pid, "host": host, "port": port}

        except FileNotFoundError:
            msg = "Ollama executable not found."
            self.logger.error(msg)
            raise ServerError(msg)
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            raise ServerError(f"Error starting Ollama server: {e}")

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
                return f"Attempted to stop Ollama server with PID {pid}"
            except ProcessLookupError:
                self.logger.warning(f"No process found with PID {pid}")
                return f"No process found with PID {pid}"
            except Exception as e:
                self.logger.error(f"Error stopping Ollama server PID {pid}: {e}")
                return f"Error stopping Ollama server PID {pid}: {e}"
        elif port:
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
                    return f"Attempted to stop Ollama server (PID {found_pid}) on port {port}"
                else:
                    return f"No Ollama server found listening on port {port}"
            except Exception as e:
                self.logger.error(f"Error stopping Ollama server on port {port}: {e}")
                return f"Error stopping Ollama server on port {port}: {e}"
        else:
            # General stop command `ollama stop` - this might not exist or work as expected for `ollama serve`
            # `ollama ps` and then finding the server PID might be more reliable if `ollama stop` isn't for `serve`
            self.logger.info("Attempting to stop the main Ollama application/service (if running).")
            try:
                # The 'ollama stop' command usually refers to stopping a model being run, not 'ollama serve'
                # For 'ollama serve', typically pkill or systemctl is used if managed.
                # This is a best-effort and might not stop a detached `ollama serve`.
                # stdout, stderr = await self._run_subprocess(['ollama', 'stop']) # This command does not exist
                # self.logger.info(f"Ollama stop command output: {stdout}")
                return "General 'ollama stop' for a server is not a standard command. Please provide PID or manage via system services."
            except Exception as e:
                self.logger.error(f"Error sending general stop signal to Ollama: {e}")
                return f"Error sending general stop signal to Ollama: {e}"

    def _terminate_process(self, pid: int):
        """Helper to terminate a process by PID."""
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
            raise ProcessLookupError(f"No process found with PID {pid}")
        except Exception as e:
            raise ServerError(f"Failed to terminate process {pid}: {e}")

    async def inference(self, model_name: str, prompt: str, system_message: Optional[str] = None,
                        port: Optional[int] = None, host: str = "127.0.0.1",
                        options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Performs inference using a model served by Ollama.
        Assumes `ollama serve` is running or starts it.
        Uses the Ollama REST API.
        """
        port = port or self.config.default_port
        api_url = f"http://{host}:{port}/api/generate"

        if not await self.is_model_available(model_name):
            try:
                await self.pull_model(model_name)
            except ModelDownloadError:
                raise InferenceError(f"Model {model_name} not found and could not be pulled.")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False  # For non-streaming response
        }
        if system_message:
            payload["system"] = system_message
        if options:
            payload["options"] = options

        self.logger.debug(f"Sending inference request to {api_url} with payload: {payload}")

        # Use an HTTP client like aiohttp for async requests
        try:
            import aiohttp
        except ImportError:
            self.logger.error("aiohttp is not installed. Please install it: pip install aiohttp")
            raise ImportError("aiohttp is required for Ollama inference.")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.debug(f"Ollama inference successful for {model_name}.")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama API error ({response.status}): {error_text}")
                        # Check if it's a model not found error from the server
                        if "model not found" in error_text.lower() or response.status == 404:
                            # Attempt to pull if not found, then retry (once)
                            self.logger.info(f"Model {model_name} not found on server, attempting to pull.")
                            try:
                                await self.pull_model(model_name)
                                async with session.post(api_url, json=payload) as retry_response:
                                    if retry_response.status == 200:
                                        return await retry_response.json()
                                    else:
                                        retry_error_text = await retry_response.text()
                                        raise InferenceError(
                                            f"Ollama API error after pull ({retry_response.status}): {retry_error_text}")
                            except ModelDownloadError as e_pull:
                                raise InferenceError(
                                    f"Model {model_name} could not be pulled: {e_pull}. Original API error: {error_text}")
                        raise InferenceError(f"Ollama API error ({response.status}): {error_text}")
            except aiohttp.ClientConnectorError as e:
                self.logger.error(f"Could not connect to Ollama server at {api_url}: {e}")
                self.logger.info("Attempting to start Ollama server...")
                try:
                    await self.serve_model(model_name, port=port, host=host)  # model_name is just for logging here
                    await asyncio.sleep(5)  # Wait for server to be ready
                    # Retry inference
                    async with session.post(api_url, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise InferenceError(
                                f"Ollama API error after server start ({response.status}): {error_text}")
                except ServerError as se:
                    raise InferenceError(f"Could not start or connect to Ollama server: {se}")
                except Exception as e_retry:
                    raise InferenceError(f"Failed to perform inference after server start attempt: {e_retry}")

#
# End of Ollama_Handler.py
#######################################################################################################################
