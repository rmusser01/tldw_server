# Llamafile_Handler.py
#
# Imports
import logging
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
import loguru as logger
#
# Local imports
from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelDownloadError, ServerError, \
    ModelNotFoundError, InferenceError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamafileConfig
from tldw_Server_API.app.core.Utils.Utils import download_file, verify_checksum
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
        self.logger = logging  # Ensure logger is assigned

        self.llamafile_exe_path = self.config.llamafile_dir / ("llamafile.exe" if os.name == "nt" else "llamafile")
        self.models_dir = Path(self.config.models_dir)

        self.config.llamafile_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Corrected type hint for asyncio.subprocess.Process
        self._active_servers: Dict[int, asyncio.subprocess.Process] = {}
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

        repo = "Mozilla-Ocho/llamafile"
        asset_name_prefix = "llamafile-"  # This needs to be accurate based on current releases
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"

        try:
            import httpx
        except ImportError:
            self.logger.error("httpx is not installed. Please install it: pip install httpx")
            raise ImportError("httpx is required for downloading llamafile.")

        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for fetching release info
            try:
                self.logger.debug(f"Fetching latest release info from {latest_release_url}")
                response = await client.get(latest_release_url)
                response.raise_for_status()
                latest_release_data = response.json()
                tag_name = latest_release_data['tag_name']
                self.logger.debug(f"Latest release tag: {tag_name}")

                assets = latest_release_data.get('assets', [])
                asset_url = None
                chosen_asset_name = None

                # Prioritize assets that are just "llamafile" or "llamafile-<version>"
                # As the universal executable is often simply named.
                simple_llamafile_asset = None
                for asset in assets:
                    if asset['name'] == "llamafile" or asset['name'].startswith(f"llamafile-{tag_name}") or asset[
                        'name'].startswith(f"{asset_name_prefix}{tag_name.lstrip('v')}"):
                        # Check if it's an executable type or no extension (common for Linux/macOS executables)
                        if '.' not in asset['name'].split('-')[-1] or asset['name'].endswith(
                                ('.exe', '.zip')) == False:  # Heuristic for executable
                            simple_llamafile_asset = asset
                            break

                if simple_llamafile_asset:
                    asset_url = simple_llamafile_asset['browser_download_url']
                    chosen_asset_name = simple_llamafile_asset['name']
                else:  # Fallback to previous broader search if specific one isn't found
                    preferred_assets = []
                    for asset in assets:
                        # More general check if the specific name isn't found
                        if asset['name'].startswith(asset_name_prefix) and "debug" not in asset['name'].lower():
                            preferred_assets.append(asset)

                    if not preferred_assets:
                        for asset in assets:  # Broader fallback if prefix fails
                            if tag_name in asset['name'] and 'llamafile' in asset['name'].lower() and "debug" not in \
                                    asset['name'].lower():
                                preferred_assets.append(asset)

                    if preferred_assets:
                        # Simplistic choice: take the first one. Might need refinement.
                        asset_to_download = preferred_assets[0]
                        # Prefer smaller, non-source files if multiple matches
                        preferred_assets.sort(key=lambda x: x.get('size', float('inf')))
                        for pa in preferred_assets:
                            if 'src' not in pa['name'].lower() and 'source' not in pa['name'].lower():
                                asset_to_download = pa
                                break
                        asset_url = asset_to_download['browser_download_url']
                        chosen_asset_name = asset_to_download['name']

                if not asset_url:
                    self.logger.error(
                        f"No suitable llamafile asset found in release {tag_name}. Assets: {[a['name'] for a in assets]}")
                    raise ModelDownloadError(f"No suitable llamafile asset found for tag {tag_name}.")

                self.logger.info(
                    f"Found asset: {chosen_asset_name}. Downloading Llamafile from {asset_url} to {output_path}...")

                # Ensure the target directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Streaming download
                async with client.stream("GET", asset_url, follow_redirects=True,
                                         timeout=300.0) as response_download:  # Long timeout for download
                    response_download.raise_for_status()
                    total_size = int(response_download.headers.get('content-length', 0))

                    # Using tqdm for progress if available, can be made optional
                    try:
                        from tqdm.asyncio import tqdm  # For async progress bar
                        
                        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name,
                                    disable=not self.logger.isEnabledFor(logging.DEBUG))  # Only show if debug
                    except ImportError:
                        pbar = None
                        self.logger.info(f"tqdm not found, downloading {output_path.name} without progress bar.")

                    with open(output_path, 'wb') as f:
                        async for chunk in response_download.aiter_bytes():
                            f.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()

                await asyncio.to_thread(os.chmod, output_path, 0o755)
                self.logger.info(f"Downloaded {output_path.name} successfully.")
                return output_path

            except httpx.HTTPStatusError as e:
                self.logger.error(
                    f"Failed to fetch llamafile release info/download: {e.response.status_code} - {e.response.text}",
                    exc_info=True)
                raise ModelDownloadError(f"Failed to fetch/download llamafile: {e.response.status_code}")
            except Exception as e:
                self.logger.error(f"Unexpected error downloading llamafile: {e}", exc_info=True)
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise ModelDownloadError(f"Unexpected error downloading llamafile: {e}")

    # --- Model Management ---
    async def download_model_file(self, model_name: str, model_url: str, model_filename: Optional[str] = None,
                                  expected_hash: Optional[str] = None, force_download: bool = False) -> Path:
        """Downloads the specified LLM model file (.llamafile or .gguf)."""
        filename = model_filename or model_url.split('/')[-1].split('?')[0]
        model_path = self.models_dir / filename

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
        self.logger.info(f"Downloading model: {model_name} from {model_url} to {model_path}")
        try:
            # project_utils.download_file needs to be a real function that can handle large files
            # and ideally offer progress. If it's blocking, to_thread is correct.
            await asyncio.to_thread(
                download_file,  # This function must exist in your Utils.py
                url=model_url,
                dest_path=str(model_path),
                expected_checksum=expected_hash  # Your download_file should handle this
            )
            self.logger.info(f"Downloaded model '{model_name}' ({filename}) successfully.")
            return model_path
        except NotImplementedError:  # If placeholder download_file is used
            self.logger.error("project_utils.download_file is not implemented. Cannot download model.")
            raise ModelDownloadError("Model download function not implemented.")
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}", exc_info=True)
            if model_path.exists(): model_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"Failed to download model {model_name}: {e}")

    async def list_models(self) -> List[str]:
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
    async def start_server(self, model_filename: str, server_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        llamafile_exe = await self.download_latest_llamafile_executable()
        if not llamafile_exe or not llamafile_exe.exists():
            raise ServerError("Llamafile executable not found or could not be downloaded.")

        model_path = self.models_dir / model_filename
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file {model_filename} not found in {self.models_dir}.")

        args = server_args or {}
        port = args.get("port", self.config.default_port)
        host = args.get("host", self.config.default_host)

        # Corrected check using .returncode for asyncio.subprocess.Process
        if port in self._active_servers and self._active_servers[port].returncode is None:
            active_pid = self._active_servers[port].pid
            self.logger.warning(
                f"Llamafile server already managed on port {port} with PID {active_pid}.")
            return {"status": "already_managed", "pid": active_pid, "port": port, "host": host,
                    "model": model_filename}

        command = [str(llamafile_exe), "-m", str(model_path)]
        command.extend(["--port", str(port)])
        if host: command.extend(["--host", host])
        if args.get("threads"): command.extend(["-t", str(args["threads"])])
        if args.get("threads_batch"): command.extend(["-tb", str(args["threads_batch"])])  # Common short flag
        if args.get("ctx_size") or args.get("c"): command.extend(["-c", str(args.get("ctx_size") or args.get("c"))])
        if args.get("ngl") or args.get("gpu_layers"): command.extend(
            ["-ngl", str(args.get("ngl") or args.get("gpu_layers"))])
        if args.get("batch_size") or args.get("b"): command.extend(["-b", str(args.get("batch_size") or args.get("b"))])
        if args.get("verbose"): command.append("-v")
        if args.get("api_key"): command.extend(["--api-key", str(args.get("api_key"))])

        # For boolean flags like --log-disable, or --memory-f32, --numa from your original script
        bool_flags_map = {
            "log_disable": "--log-disable",
            "memory_f32": "--memory-f32",
            "numa": "--numa",
            "sane_defaults": "--sane-defaults",  # from your start_llamafile
            # Add more as needed
        }
        for arg_key, flag_str in bool_flags_map.items():
            if args.get(arg_key) is True:
                if flag_str not in command: command.append(flag_str)

        # For other key-value pairs not explicitly handled
        explicitly_handled_keys = {
            "port", "host", "threads", "threads_batch", "ctx_size", "c", "ngl", "gpu_layers",
            "batch_size", "b", "verbose", "api_key"
        }.union(bool_flags_map.keys())

        for k, v in args.items():
            if k not in explicitly_handled_keys:
                # Skip if value is False (for flags that might be set to False to disable)
                if v is False: continue

                # Construct flag, try --k first, then -k if common
                flag = f"--{k.replace('_', '-')}"  # common convention for multi-word args
                alt_flag = f"-{k}"

                # Check if flag or its value is already part of the command by some other means
                # This is a bit tricky; the current explicit handling should cover most common cases.
                # This part is mainly for less common or new llamafile arguments.
                is_already_added = False
                for item in command:
                    if flag == item or alt_flag == item:
                        is_already_added = True
                        break

                if not is_already_added:
                    if v is True:  # Boolean flag
                        command.append(flag)
                    else:  # Key-value pair
                        command.extend([flag, str(v)])

        self.logger.info(f"Starting llamafile server for {model_filename} with command: {' '.join(command)}")

        try:
            # Using create_subprocess_exec for better security with list of args
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid if platform.system() != "Windows" else None
            )
            await asyncio.sleep(3)  # Increased sleep slightly

            if process.returncode is not None:
                stderr_output = ""
                if process.stderr:
                    err_bytes = await process.stderr.read()
                    stderr_output = err_bytes.decode(errors='ignore').strip()
                self.logger.error(
                    f"Llamafile server failed to start for {model_filename}. Exit code: {process.returncode}. Stderr: {stderr_output}")
                stdout_output = ""
                if process.stdout:
                    out_bytes = await process.stdout.read()
                    stdout_output = out_bytes.decode(errors='ignore').strip()
                if stdout_output: self.logger.error(f"Llamafile server stdout: {stdout_output}")
                raise ServerError(f"Llamafile server failed to start. Stderr: {stderr_output}")

            self._active_servers[port] = process
            self.logger.info(f"Llamafile server started for {model_filename} on port {port} with PID {process.pid}.")
            return {"status": "started", "pid": process.pid, "port": port, "host": host, "model": model_filename,
                    "command": ' '.join(command)}  # Return stringified command
        except Exception as e:
            self.logger.error(f"Exception starting llamafile server for {model_filename}: {e}", exc_info=True)
            raise ServerError(f"Exception starting llamafile: {e}")

    async def stop_server(self, port: Optional[int] = None, pid: Optional[int] = None) -> str:
        process_to_stop: Optional[asyncio.subprocess.Process] = None
        port_to_clear = None

        if pid:
            for p, proc_obj in self._active_servers.items():
                if proc_obj.pid == pid:
                    process_to_stop = proc_obj
                    port_to_clear = p
                    break
            if not process_to_stop:
                self.logger.warning(
                    f"PID {pid} not in managed servers. Attempting to terminate externally (best effort).")
                try:
                    # This part is synchronous and for unmanaged processes
                    target_pid = int(pid)
                    if platform.system() == "Windows":
                        subprocess.run(['taskkill', '/F', '/PID', str(target_pid)], check=True, capture_output=True)
                    else:
                        os.kill(target_pid, signal.SIGTERM)  # Can use os.killpg if it was started in a group
                    return f"Attempted to send SIGTERM to unmanaged llamafile server with PID {pid}."
                except ProcessLookupError:
                    return f"No process found with PID {pid}."
                except subprocess.CalledProcessError as e_taskkill:
                    self.logger.error(f"taskkill failed for PID {pid}: {e_taskkill.stderr.decode()}")
                    return f"Failed to stop unmanaged PID {pid} with taskkill."
                except Exception as e:
                    self.logger.error(f"Error stopping unmanaged PID {pid}: {e}", exc_info=True)
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

        current_pid = process_to_stop.pid
        self.logger.info(f"Stopping llamafile server (PID: {current_pid}, Port: {port_to_clear or 'N/A'}).")
        try:
            # Check if still running using returncode
            if process_to_stop.returncode is None:
                if platform.system() == "Windows":
                    process_to_stop.terminate()
                else:
                    try:
                        # Get process group ID (pgid) to terminate the entire group
                        pgid = await asyncio.to_thread(os.getpgid, current_pid)
                        await asyncio.to_thread(os.killpg, pgid, signal.SIGTERM)
                        self.logger.info(f"Sent SIGTERM to process group {pgid} (leader PID: {current_pid}).")
                    except ProcessLookupError:  # Process might have died just now
                        self.logger.warning(
                            f"Process {current_pid} not found during getpgid, likely already terminated.")
                        # Fallback to terminating just the PID if getpgid fails for other reasons
                        process_to_stop.terminate()
                        self.logger.info(f"Sent SIGTERM to process PID {current_pid} (fallback).")

                try:
                    await asyncio.wait_for(process_to_stop.wait(), timeout=10)
                    self.logger.info(
                        f"Llamafile server PID {current_pid} terminated gracefully (return code: {process_to_stop.returncode}).")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Llamafile server PID {current_pid} did not terminate gracefully after SIGTERM. Killing.")
                    if platform.system() == "Windows":
                        process_to_stop.kill()
                    else:
                        try:
                            pgid = await asyncio.to_thread(os.getpgid, current_pid)
                            await asyncio.to_thread(os.killpg, pgid, signal.SIGKILL)
                            self.logger.info(f"Sent SIGKILL to process group {pgid} (leader PID: {current_pid}).")
                        except ProcessLookupError:
                            self.logger.warning(f"Process {current_pid} not found during getpgid for SIGKILL.")
                            process_to_stop.kill()  # Fallback
                            self.logger.info(f"Sent SIGKILL to process PID {current_pid} (fallback).")

                    await process_to_stop.wait()  # Ensure it's reaped
                    self.logger.info(
                        f"Llamafile server PID {current_pid} killed (return code: {process_to_stop.returncode}).")
            else:
                self.logger.info(
                    f"Llamafile server PID {current_pid} was already stopped (return code: {process_to_stop.returncode}).")

            if port_to_clear and port_to_clear in self._active_servers:
                del self._active_servers[port_to_clear]
            return f"Llamafile server PID {current_pid} stopped."
        except Exception as e:
            self.logger.error(f"Error stopping llamafile server PID {current_pid}: {e}", exc_info=True)
            if port_to_clear and port_to_clear in self._active_servers:
                del self._active_servers[port_to_clear]
            raise ServerError(f"Error stopping llamafile server: {e}")

    async def inference(self,
                        prompt: str,
                        port: int,
                        host: Optional[str] = None,
                        system_prompt: Optional[str] = None,
                        n_predict: int = -1,
                        temperature: float = 0.8,
                        top_k: int = 40,
                        top_p: float = 0.95,
                        api_key: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        target_host = host or self.config.default_host
        api_url = f"http://{target_host}:{port}/v1/chat/completions"

        if port not in self._active_servers or self._active_servers[port].returncode is not None:
            self.logger.debug(
                f"Port {port} not in _active_servers or process terminated. Checking for external server responsiveness.")
            conn_made = False
            try:
                _, writer = await asyncio.open_connection(target_host, port)
                writer.close()
                await writer.wait_closed()
                conn_made = True
                self.logger.debug(f"Successfully connected to {target_host}:{port}. Assuming external server.")
            except ConnectionRefusedError:
                self.logger.error(
                    f"No managed llamafile server on port {port} (or it terminated), and connection refused to {target_host}:{port}.")
                raise ServerError(f"Llamafile server not found or not responding on {target_host}:{port}.")
            except Exception as e:
                self.logger.error(f"Error checking connection to {target_host}:{port}: {e}", exc_info=True)
                raise ServerError(f"Error connecting to Llamafile server at {target_host}:{port}: {e}")
            if not conn_made:
                raise ServerError(f"Llamafile server not found/responding on {target_host}:{port} (conn test failed).")
        else:
            self.logger.debug(f"Using managed llamafile server on port {port}.")

        headers = {"Content-Type": "application/json"}
        if api_key: headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages, "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "n_predict": n_predict, "stream": False, **kwargs
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        self.logger.debug(f"Sending llamafile inference request to {api_url} with payload: {payload}")
        try:
            import httpx
        except ImportError:
            self.logger.error("httpx is required for Llamafile inference. Please install it: pip install httpx")
            raise ImportError("httpx is required for Llamafile inference.")

        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 120.0)) as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                self.logger.debug("Llamafile inference successful.")
                return result
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                self.logger.error(f"Llamafile API error ({e.response.status_code}) from {api_url}: {error_text}",
                                  exc_info=True)
                raise InferenceError(f"Llamafile API error ({e.response.status_code}): {error_text}")
            except httpx.RequestError as e:
                self.logger.error(f"Could not connect or communicate with Llamafile server at {api_url}: {e}",
                                  exc_info=True)
                raise ServerError(f"Could not connect/communicate with Llamafile server at {api_url}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error during llamafile inference to {api_url}: {e}", exc_info=True)
                raise InferenceError(f"Unexpected error during llamafile inference: {e}")

    def _cleanup_all_managed_servers_sync(self):  # Renamed to indicate it's synchronous
        """Synchronous cleanup for signal handlers or app shutdown."""
        self.logger.info("Cleaning up all managed llamafile servers (sync)...")
        ports_to_remove = list(self._active_servers.keys())
        for port in ports_to_remove:
            proc = self._active_servers.get(port)
            # Check if proc exists and if its returncode is None (meaning it might be running)
            if proc and proc.returncode is None:
                pid = proc.pid
                self.logger.info(f"Stopping server on port {port}, PID {pid}...")
                try:
                    if platform.system() == "Windows":
                        proc.terminate()
                    else:
                        # For processes started with os.setsid, kill the process group
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGTERM)
                            self.logger.info(f"Sent SIGTERM to process group {pgid} (leader PID {pid}).")
                        except ProcessLookupError:
                            self.logger.warning(
                                f"Process {pid} (or group) not found during SIGTERM, likely already terminated.")
                            # Fallback just in case, or if it wasn't started with setsid somehow
                            proc.terminate()

                    # proc.wait() is a coroutine, cannot be called directly in sync func
                    # For a sync cleanup, we might need to use a different approach or
                    # acknowledge that immediate reaping might not happen here.
                    # A simple approach for atexit: send terminate and hope for the best.
                    # More robust would be to launch a small async task to await termination.
                    # For now, just send terminate/kill.
                    try:
                        # Python's subprocess module Popen has a wait() method, but asyncio.Process does not have a sync one.
                        # We are in a sync function (_cleanup_all_managed_servers_sync)
                        # We can't `await proc.wait()`.
                        # The OS will eventually reap, but for cleaner shutdown logging:
                        self.logger.debug(f"Termination signal sent to PID {pid}. OS will handle reaping.")
                    except Exception as e_wait:  # Catch any error from trying to wait
                        self.logger.warning(f"Error during proc.wait() for PID {pid} in sync cleanup: {e_wait}")


                except ProcessLookupError:  # If process died between check and action
                    self.logger.warning(f"Process {pid} not found during termination, likely already exited.")
                except Exception as e:
                    self.logger.error(f"Error during cleanup of PID {pid}: {e}. Attempting kill.")
                    if proc.returncode is None:  # Check again before kill
                        if platform.system() == "Windows":
                            proc.kill()
                        else:
                            try:
                                pgid = os.getpgid(pid)
                                os.killpg(pgid, signal.SIGKILL)
                            except:
                                proc.kill()  # fallback
            if port in self._active_servers:
                del self._active_servers[port]
        self.logger.info("Managed llamafile server synchronous cleanup attempt complete.")

    def _signal_handler(self, sig, frame):
        self.logger.info(f'Signal handler called with signal: {sig}')
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
