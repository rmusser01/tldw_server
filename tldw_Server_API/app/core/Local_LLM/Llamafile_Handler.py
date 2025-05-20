# Llamafile_Handler.py
#
# Imports
import os
import platform
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
#
# Third-party imports
import asyncio

from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelDownloadError, ServerError, \
    ModelNotFoundError, InferenceError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamafileConfig


#
# Local imports
# from .base_handler import BaseLLMHandler
# from .exceptions import ModelNotFoundError, ModelDownloadError, ServerError, InferenceError
# from .utils_loader import logging, project_utils # From the loader
# from .config_model import LlamafileConfig
#
########################################################################################################################
#
# Functions:


class LlamafileHandler(BaseLLMHandler):
    def __init__(self, config: LlamafileConfig, global_app_config: Dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: LlamafileConfig  # For type hinting

        self.llamafile_exe_path = self.config.llamafile_dir / ("llamafile.exe" if os.name == "nt" else "llamafile")
        self.models_dir = Path(self.config.models_dir)

        self.config.llamafile_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # To keep track of managed llamafile processes
        self._active_servers: Dict[int, subprocess.Popen] = {}  # port -> process
        self._setup_signal_handlers()  # For cleaning up on exit

    # --- Llamafile Executable Management ---
    async def download_latest_llamafile_executable(self, force_download: bool = False) -> Path:
        """Downloads the latest llamafile binary if not present or if force_download is True."""
        output_path = self.llamafile_exe_path
        self.logger.info(f"Checking for llamafile executable at {output_path}...")
        if output_path.exists() and not force_download:
            self.logger.debug(f"{output_path.name} already exists. Skipping download.")
            await asyncio.to_thread(os.chmod, output_path, 0o755)  # Ensure executable
            return output_path

        repo = "Mozilla-Ocho/llamafile"  # TODO: Make this configurable if needed
        asset_name_prefix = "llamafile-"
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"

        try:
            import httpx  # Using httpx for async HTTP requests
        except ImportError:
            self.logger.error("httpx is not installed. Please install it: pip install httpx")
            raise ImportError("httpx is required for downloading llamafile.")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(latest_release_url)
                response.raise_for_status()
                latest_release_data = response.json()
                tag_name = latest_release_data['tag_name']

                # The direct asset URL might be in the main latest release, or we might need to find it by OS
                assets = latest_release_data.get('assets', [])
                asset_url = None

                # Try to find a platform-specific one first, or a generic one
                # llamafile releases are often like llamafile-0.X or llamafile-server-0.X
                # Example: llamafile-0.8.6 , llamafile-compatibility-0.8.3 (older style)
                # Need to determine if we need a server-specific build or the general one.
                # The original script used a generic prefix.
                platform_suffix_map = {
                    "linux": "linux-x86_64",  # This might vary, check actual asset names
                    "darwin": "macos",  # This might vary
                    "win32": ".exe"  # Or llamafile-win.zip etc.
                }
                # The actual asset name seems to be just "llamafile-X.Y.Z" and it's multi-platform
                # Or "llamafile-server-X.Y.Z"
                # Let's prioritize simple "llamafile-<version>"

                preferred_assets = []
                for asset in assets:
                    if asset['name'].startswith(asset_name_prefix) and not "debug" in asset['name']:
                        # Basic llamafile should work for serving.
                        # We might need to be more specific if there are server-only builds.
                        preferred_assets.append(asset)

                if not preferred_assets:  # Fallback: check for assets with version tag if prefix fails
                    for asset in assets:
                        if tag_name in asset['name'] and 'llamafile' in asset['name']:
                            preferred_assets.append(asset)

                if preferred_assets:
                    # Simplistic choice: take the first one. Might need refinement based on OS.
                    # Often the main 'llamafile-X.Y.Z' is the universal one.
                    asset_url = preferred_assets[0]['browser_download_url']
                    self.logger.info(f"Found asset: {preferred_assets[0]['name']}")

                if not asset_url:
                    self.logger.error(
                        f"No suitable asset found with prefix '{asset_name_prefix}' or tag '{tag_name}' in the latest release.")
                    raise ModelDownloadError(f"No llamafile asset found.")

                self.logger.info(f"Downloading Llamafile from {asset_url} to {output_path}...")
                # Use project_utils.download_file (which needs to be async or run in thread)
                # For now, direct download with httpx streaming
                async with client.stream("GET", asset_url, follow_redirects=True) as response_download:
                    response_download.raise_for_status()
                    with open(output_path, 'wb') as f:
                        async for chunk in response_download.aiter_bytes():
                            f.write(chunk)

                await asyncio.to_thread(os.chmod, output_path, 0o755)  # Make it executable
                self.logger.debug(f"Downloaded {output_path.name} from {asset_url}")
                return output_path

            except httpx.HTTPStatusError as e:
                self.logger.error(
                    f"Failed to fetch llamafile release info: {e.response.status_code} - {e.response.text}")
                raise ModelDownloadError(f"Failed to fetch llamafile release info: {e.response.status_code}")
            except Exception as e:
                self.logger.error(f"Unexpected error downloading llamafile: {e}")
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise ModelDownloadError(f"Unexpected error downloading llamafile: {e}")

    # --- Model Management ---
    async def download_model_file(self, model_name: str, model_url: str, model_filename: Optional[str] = None,
                                  expected_hash: Optional[str] = None, force_download: bool = False) -> Path:
        """Downloads the specified LLM model file (.llamafile or .gguf)."""
        filename = model_filename or model_url.split('/')[-1].split('?')[0]  # Basic filename extraction
        model_path = self.models_dir / filename

        self.logger.info(f"Checking availability of model: {model_name} at {model_path}")
        if model_path.exists() and not force_download:
            # Optionally, verify hash if present and not forced
            if expected_hash and not await asyncio.to_thread(project_utils.verify_checksum, str(model_path),
                                                             expected_hash):
                self.logger.warning(f"Checksum mismatch for existing model {model_path}. Re-downloading.")
                model_path.unlink()
            else:
                self.logger.debug(f"Model '{model_name}' ({filename}) already exists. Skipping download.")
                return model_path

        self.logger.info(f"Downloading model: {model_name} from {model_url} to {model_path}")
        try:
            # Assuming project_utils.download_file can handle large files and is thread-safe or async
            # If it's blocking, run it in a thread
            await asyncio.to_thread(
                project_utils.download_file,
                url=model_url,
                dest_path=str(model_path),
                expected_checksum=expected_hash
            )
            self.logger.debug(f"Downloaded model '{model_name}' ({filename}) successfully.")
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            if model_path.exists(): model_path.unlink(missing_ok=True)  # Clean up partial
            raise ModelDownloadError(f"Failed to download model {model_name}: {e}")

    async def list_models(self) -> List[str]:
        """Retrieves model files (.gguf or .llamafile) from the local models directory."""
        if not self.models_dir.exists():
            return []

        def _scan_dir():
            files = []
            for ext in ("*.gguf", "*.llamafile"):
                files.extend(self.models_dir.glob(ext))
            return [f.name for f in files]

        return await asyncio.to_thread(_scan_dir)

    async def is_model_available(self, model_filename: str) -> bool:
        return (self.models_dir / model_filename).exists()

    # --- Server Management ---
    async def start_server(self, model_filename: str, server_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Starts the llamafile server for the given model.
        model_filename: Name of the .llamafile or .gguf file in self.models_dir.
        server_args: Dictionary of command-line arguments for llamafile.
                     Example: {"port": 8080, "threads": 4, "ctx-size": 2048, "ngl": 100}
        Returns a dict with server info, including PID and port.
        """
        llamafile_exe = await self.download_latest_llamafile_executable()
        if not llamafile_exe or not llamafile_exe.exists():
            raise ServerError("Llamafile executable not found or could not be downloaded.")

        model_path = self.models_dir / model_filename
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file {model_filename} not found in {self.models_dir}.")

        args = server_args or {}
        port = args.get("port", self.config.default_port)
        host = args.get("host", self.config.default_host)  # Llamafile defaults to 0.0.0.0 usually

        # Check if a server is already managed on this port
        if port in self._active_servers and self._active_servers[port].poll() is None:
            self.logger.warning(
                f"Llamafile server already managed on port {port} with PID {self._active_servers[port].pid}.")
            return {"status": "already_managed", "pid": self._active_servers[port].pid, "port": port, "host": host,
                    "model": model_filename}

        # Construct command
        # Basic command: ./llamafile -m model.gguf --port 8080 --host 0.0.0.0
        # Server mode is often implied by --port or specific server flags
        command = [str(llamafile_exe), "-m", str(model_path)]

        # Add common server arguments from server_args
        command.extend(["--port", str(port)])
        if host: command.extend(["--host", host])  # Host might be optional if llamafile defaults well
        if args.get("threads"): command.extend(["-t", str(args["threads"])])
        if args.get("threads-batch"): command.extend(["-tb", str(args["threads-batch"])])  # Or --threads-batch
        if args.get("ctx-size") or args.get("c"): command.extend(["-c", str(args.get("ctx-size") or args.get("c"))])
        if args.get("ngl") or args.get("gpu-layers"): command.extend(
            ["-ngl", str(args.get("ngl") or args.get("gpu-layers"))])
        if args.get("batch-size") or args.get("b"): command.extend(["-b", str(args.get("batch-size") or args.get("b"))])
        if args.get("verbose"): command.append("-v")
        if args.get("api-key"): command.extend(["--api-key", str(args.get("api-key"))])
        # Add other flags as needed, converting True to just the flag, e.g. --log-disable
        for k, v in args.items():
            if k not in ["port", "host", "threads", "threads-batch", "ctx-size", "ngl", "batch-size", "verbose",
                         "api-key", "c", "gpu-layers", "b"] and v is True and f"--{k}" not in command:
                command.append(f"--{k}")
            elif k not in ["port", "host", "threads", "threads-batch", "ctx-size", "ngl", "batch-size", "verbose",
                           "api-key", "c", "gpu-layers", "b"] and v is not False and f"--{k}" not in command:
                # For key-value pairs not handled explicitly
                if f"--{k}" not in command and f"-{k}" not in command:  # Avoid duplicates
                    command.extend([f"--{k}", str(v)])

        self.logger.info(f"Starting llamafile server for {model_filename} with command: {' '.join(command)}")

        try:
            # Using subprocess.Popen directly and managing it, as it's a long-running server
            # Run in a way that doesn't block the main FastAPI thread if called directly from there
            # For asyncio, we'd use asyncio.create_subprocess_exec
            loop = asyncio.get_event_loop()
            process = await loop.subprocess_create_shell(
                ' '.join(command),  # shell=True needs careful command construction if user inputs are involved
                # Safer to use list of args with create_subprocess_exec
                # *command, # Use this if not using shell=True
                stdout=subprocess.PIPE,  # Or DEVNULL if we don't need to log output after start
                stderr=subprocess.PIPE,  # Or a file
                preexec_fn=os.setsid if platform.system() != "Windows" else None
                # Create new process group for easier cleanup on Unix
            )

            # Give it a moment to start or fail
            await asyncio.sleep(2)  # Adjust as needed

            if process.returncode is not None:  # Process exited quickly
                stderr_output = ""
                if process.stderr:
                    err_bytes = await process.stderr.read()
                    stderr_output = err_bytes.decode(errors='ignore')
                self.logger.error(
                    f"Llamafile server failed to start for {model_filename}. Exit code: {process.returncode}. Stderr: {stderr_output}")
                raise ServerError(f"Llamafile server failed to start. Stderr: {stderr_output}")

            self._active_servers[port] = process
            self.logger.info(f"Llamafile server started for {model_filename} on port {port} with PID {process.pid}.")
            return {"status": "started", "pid": process.pid, "port": port, "host": host, "model": model_filename,
                    "command": command}
        except Exception as e:
            self.logger.error(f"Exception starting llamafile server for {model_filename}: {e}", exc_info=True)
            raise ServerError(f"Exception starting llamafile: {e}")

    async def stop_server(self, port: Optional[int] = None, pid: Optional[int] = None) -> str:
        """Stops a managed llamafile server by port or PID."""
        process_to_stop = None
        port_to_clear = None

        if pid:
            for p, proc_obj in self._active_servers.items():
                if proc_obj.pid == pid:
                    process_to_stop = proc_obj
                    port_to_clear = p
                    break
            if not process_to_stop:
                # Try to stop an unmanaged process by PID (less safe)
                self.logger.warning(f"PID {pid} not in managed servers. Attempting to terminate externally.")
                try:
                    if platform.system() == "Windows":
                        await self._run_subprocess(['taskkill', '/F', '/PID', str(pid)])
                    else:
                        os.kill(pid, signal.SIGTERM)  # This is blocking, use asyncio.to_thread or send_signal
                        # await asyncio.to_thread(os.kill, pid, signal.SIGTERM)
                    return f"Attempted to stop unmanaged llamafile server with PID {pid}."
                except ProcessLookupError:
                    return f"No process found with PID {pid}."
                except Exception as e:
                    raise ServerError(f"Error stopping unmanaged PID {pid}: {e}")


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

        self.logger.info(f"Stopping llamafile server (PID: {process_to_stop.pid}, Port: {port_to_clear or 'N/A'}).")
        try:
            if process_to_stop.poll() is None:  # If still running
                if platform.system() == "Windows":
                    process_to_stop.terminate()  # or send_signal(signal.CTRL_C_EVENT) for graceful
                else:
                    # Send SIGTERM to the process group if preexec_fn=os.setsid was used
                    os.killpg(os.getpgid(process_to_stop.pid), signal.SIGTERM)

                try:
                    await asyncio.wait_for(process_to_stop.wait(), timeout=10)
                    self.logger.info(f"Llamafile server PID {process_to_stop.pid} terminated.")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Llamafile server PID {process_to_stop.pid} did not terminate gracefully. Killing.")
                    if platform.system() == "Windows":
                        process_to_stop.kill()
                    else:
                        os.killpg(os.getpgid(process_to_stop.pid), signal.SIGKILL)
                    process_to_stop.wait()  # Ensure it's reaped
            else:
                self.logger.info(f"Llamafile server PID {process_to_stop.pid} was already stopped.")

            if port_to_clear and port_to_clear in self._active_servers:
                del self._active_servers[port_to_clear]
            return f"Llamafile server PID {process_to_stop.pid} stopped."
        except Exception as e:
            self.logger.error(f"Error stopping llamafile server PID {process_to_stop.pid}: {e}", exc_info=True)
            # Clean up from active servers if error occurs during stop
            if port_to_clear and port_to_clear in self._active_servers:
                del self._active_servers[port_to_clear]
            raise ServerError(f"Error stopping llamafile server: {e}")

    async def inference(self,
                        prompt: str,
                        port: int,  # Port where the target llamafile server is running
                        host: Optional[str] = None,
                        system_prompt: Optional[str] = None,  # For chat completions
                        n_predict: int = -1,  # Max tokens
                        temperature: float = 0.8,
                        top_k: int = 40,
                        top_p: float = 0.95,
                        api_key: Optional[str] = None,
                        # Add other OpenAI compatible params: stop, presence_penalty, frequency_penalty, etc.
                        **kwargs) -> Dict[str, Any]:
        """
        Performs inference using a llamafile server (OpenAI compatible API).
        Requires the llamafile server to be already running on the specified port.
        """
        target_host = host or self.config.default_host
        api_url = f"http://{target_host}:{port}/v1/chat/completions"  # Llamafile uses this endpoint

        if port not in self._active_servers and pid is None:  # if pid is None, means we are not targeting an externally managed server by pid
            # Optional: Check if an unmanaged server is listening on the port
            conn_made = False
            try:
                _, writer = await asyncio.open_connection(target_host, port)
                writer.close()
                await writer.wait_closed()
                conn_made = True
            except ConnectionRefusedError:
                pass  # Server not there or not listening

            if not conn_made:
                self.logger.error(
                    f"No managed llamafile server on port {port}, and connection refused. Please start a server first.")
                raise ServerError(f"Llamafile server not found or not responding on port {port}.")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Construct OpenAI-like payload
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "n_predict": n_predict,  # Llama.cpp specific, maps to max_tokens somewhat
            "stream": False,
            **kwargs  # Pass other OpenAI compatible params
        }
        # Filter None values from payload for cleaner requests
        payload = {k: v for k, v in payload.items() if v is not None}

        self.logger.debug(f"Sending llamafile inference request to {api_url} with payload: {payload}")
        try:
            import httpx
        except ImportError:
            self.logger.error("httpx is not installed. Please install it: pip install httpx")
            raise ImportError("httpx is required for Llamafile inference.")

        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 120.0)) as client:  # Default timeout 120s
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
                result = response.json()
                self.logger.debug("Llamafile inference successful.")
                return result
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                self.logger.error(f"Llamafile API error ({e.response.status_code}): {error_text}")
                raise InferenceError(f"Llamafile API error ({e.response.status_code}): {error_text}")
            except httpx.RequestError as e:
                self.logger.error(f"Could not connect to Llamafile server at {api_url}: {e}")
                raise ServerError(f"Could not connect to Llamafile server at {api_url}: {e}")

    def _cleanup_all_managed_servers(self):
        """Synchronous cleanup for signal handlers or app shutdown."""
        self.logger.info("Cleaning up all managed llamafile servers...")
        ports_to_remove = list(self._active_servers.keys())
        for port in ports_to_remove:
            proc = self._active_servers.get(port)
            if proc and proc.poll() is None:  # If running
                self.logger.info(f"Stopping server on port {port}, PID {proc.pid}...")
                try:
                    if platform.system() == "Windows":
                        proc.terminate()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Terminate process group
                    proc.wait(timeout=5)  # Synchronous wait
                except Exception as e:
                    self.logger.error(f"Error during cleanup of PID {proc.pid}: {e}. Killing.")
                    if platform.system() == "Windows":
                        proc.kill()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    try:
                        proc.wait(timeout=2)
                    except:
                        pass  # Best effort
            if port in self._active_servers:
                del self._active_servers[port]
        self.logger.info("Managed llamafile server cleanup complete.")

    def _signal_handler(self, sig, frame):
        self.logger.info(f'Signal handler called with signal: {sig}')
        self._cleanup_all_managed_servers()
        sys.exit(0)

    def _setup_signal_handlers(self):
        # These handlers are process-wide. In a FastAPI app, this might conflict
        # if FastAPI itself sets up handlers. FastAPI's shutdown event is preferred
        # for cleaning up resources managed by the application.
        # For a standalone library usage, this is okay.
        # signal.signal(signal.SIGINT, self._signal_handler)
        # signal.signal(signal.SIGTERM, self._signal_handler)
        # Consider using atexit for a simpler cleanup if signals are too complex here
        import atexit
        atexit.register(self._cleanup_all_managed_servers)
        self.logger.info("Registered atexit cleanup for LlamafileHandler.")

#
# # End of Llamafile_Handler.py
#########################################################################################################################
