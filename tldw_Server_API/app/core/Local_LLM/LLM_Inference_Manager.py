# llm_inference_lib/manager.py
#
#
# Imports
from typing import Dict, Any, Optional, List
#
# Third-party imports
from loguru import logger as logging
from pathlib import Path

from tldw_Server_API.app.core.Local_LLM.Huggingface_Handler import HuggingFaceHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LLMManagerConfig
#
# Local imports
from tldw_Server_API.app.core.Local_LLM.Llamafile_Handler import LlamafileHandler
from tldw_Server_API.app.core.Local_LLM.Ollama_Handler import OllamaHandler
# from .ollama_handler import OllamaHandler # Relative imports for package structure
# from .huggingface_handler import HuggingFaceHandler
# from .llamafile_handler import LlamafileHandler
# from .config_model import LLMManagerConfig, OllamaConfig, HuggingFaceConfig, LlamafileConfig
# from .exceptions import InferenceError, ModelNotFoundError
########################################################################################################################
#
# Functions:

class LLMInferenceManager:
    def __init__(self, config: LLMManagerConfig):
        self.config = config
        self.logger = logging # Use the logger from utils_loader

        self.ollama: Optional[OllamaHandler] = None
        if self.config.ollama and self.config.ollama.enabled:
            self.ollama = OllamaHandler(self.config.ollama, self.config.app_config)
            self.logger.info("Ollama handler initialized.")

        self.huggingface: Optional[HuggingFaceHandler] = None
        if self.config.huggingface and self.config.huggingface.enabled:
            # Ensure models_dir is Path object and created
            hf_cfg = self.config.huggingface
            if isinstance(hf_cfg.models_dir, str):
                 hf_cfg.models_dir = Path(hf_cfg.models_dir)
            hf_cfg.models_dir.mkdir(parents=True, exist_ok=True)

            self.huggingface = HuggingFaceHandler(hf_cfg, self.config.app_config)
            self.logger.info(f"HuggingFace handler initialized. Models directory: {hf_cfg.models_dir}")


        self.llamafile: Optional[LlamafileHandler] = None
        if self.config.llamafile and self.config.llamafile.enabled:
            lf_cfg = self.config.llamafile
            if isinstance(lf_cfg.llamafile_dir, str):
                lf_cfg.llamafile_dir = Path(lf_cfg.llamafile_dir)
            if isinstance(lf_cfg.models_dir, str):
                lf_cfg.models_dir = Path(lf_cfg.models_dir)
            lf_cfg.llamafile_dir.mkdir(parents=True, exist_ok=True)
            lf_cfg.models_dir.mkdir(parents=True, exist_ok=True)

            self.llamafile = LlamafileHandler(lf_cfg, self.config.app_config)
            self.logger.info(f"Llamafile handler initialized. Executable dir: {lf_cfg.llamafile_dir}, Models dir: {lf_cfg.models_dir}")


    def get_handler(self, backend_name: str):
        if backend_name == "ollama" and self.ollama:
            return self.ollama
        elif backend_name == "huggingface" and self.huggingface:
            return self.huggingface
        elif backend_name == "llamafile" and self.llamafile:
            return self.llamafile
        else:
            self.logger.error(f"Backend '{backend_name}' not available or not enabled.")
            raise InferenceError(f"Backend '{backend_name}' not available.")

    async def list_local_models(self, backend: str) -> List[str]:
        handler = self.get_handler(backend)
        return await handler.list_models()

    async def download_model(self, backend: str, model_name: str, **kwargs) -> str:
        handler = self.get_handler(backend)
        if backend == "ollama":
            return await handler.pull_model(model_name, timeout=kwargs.get("timeout", 600))
        elif backend == "huggingface":
            return await handler.download_model(model_name, save_directory=kwargs.get("save_directory"))
        elif backend == "llamafile":
            # For llamafile, 'model_name' is the display name, we need URL and filename
            model_url = kwargs.get("model_url")
            if not model_url:
                raise InferenceError("model_url is required for downloading llamafile models.")
            model_filename = kwargs.get("model_filename")
            expected_hash = kwargs.get("expected_hash")
            force_download = kwargs.get("force_download", False)
            model_path = await handler.download_model_file(
                model_name, model_url, model_filename, expected_hash, force_download
            )
            return str(model_path)
        raise InferenceError(f"Download not implemented for backend {backend} via this generic method or backend unknown.")

    async def run_inference(self, backend: str, model_name_or_path: str, prompt: Any, **kwargs) -> Dict[str, Any]:
        handler = self.get_handler(backend)
        self.logger.info(f"Running inference with {backend} for model {model_name_or_path}")
        if backend == "ollama":
            # Ollama's inference uses its own client which expects a running server
            return await handler.inference(
                model_name=model_name_or_path,
                prompt=prompt,
                system_message=kwargs.get("system_message"),
                port=kwargs.get("port"), host=kwargs.get("host"),
                options=kwargs.get("options")
            )
        elif backend == "huggingface":
            # HuggingFace can do chat or pipeline. Default to chat for prompt object.
            # If prompt is a string, and messages not provided, could use text_generation_pipeline
            messages = kwargs.get("messages")
            if isinstance(prompt, str) and not messages: # Simple prompt
                 messages = [{"role": "user", "content": prompt}]
                 if kwargs.get("system_message"):
                     messages.insert(0, {"role": "system", "content": kwargs.get("system_message")})

            if messages: # Assuming chat_completion style
                response_text = await handler.chat_completion(
                    model_name_or_path=model_name_or_path,
                    messages=messages,
                    max_new_tokens=kwargs.get("max_new_tokens", 100),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    quantization_config=kwargs.get("quantization_config")
                )
                # Mimic Ollama's response structure for consistency if desired, or define own
                return {"model": model_name_or_path, "response": response_text, "done": True}
            else:
                raise InferenceError("For HuggingFace, provide 'messages' list or a simple string 'prompt'.")

        elif backend == "llamafile":
            # Llamafile uses OpenAI compatible API, requires server running
            return await handler.inference(
                prompt=prompt, # prompt is the user message content
                port=kwargs.get("port"), # Must be provided
                host=kwargs.get("host"),
                system_prompt=kwargs.get("system_message"),
                n_predict=kwargs.get("n_predict", kwargs.get("max_tokens", -1)),
                temperature=kwargs.get("temperature", 0.8),
                api_key=kwargs.get("api_key"),
                # Pass other OpenAI params
                **{k:v for k,v in kwargs.items() if k not in ["port", "host", "system_message", "n_predict", "temperature", "api_key", "max_tokens"]}
            )
        raise InferenceError(f"Inference not implemented for backend {backend} via this generic method or backend unknown.")

    async def start_server(self, backend: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        handler = self.get_handler(backend)
        self.logger.info(f"Starting server for backend {backend} with model {model_name or 'default'}")
        if backend == "ollama":
            # Ollama serve is general, model_name is for context/logging
            return await handler.serve_model(
                model_name=model_name or "any_ollama_model", # model_name is not strictly needed to start ollama serve
                port=kwargs.get("port"),
                host=kwargs.get("host")
            )
        elif backend == "llamafile":
            if not model_name:
                raise InferenceError("model_name (filename) is required for starting llamafile server.")
            return await handler.start_server(
                model_filename=model_name,
                server_args=kwargs.get("server_args") # Pass the whole dict of args
            )
        # HuggingFace doesn't typically "start a server" in the same way unless using specific serving tools
        # like Text Generation Inference, which is outside the scope of the provided transformers library usage.
        raise InferenceError(f"Server start not applicable or implemented for backend {backend} via this method.")

    async def stop_server(self, backend: str, **kwargs) -> str:
        handler = self.get_handler(backend)
        self.logger.info(f"Stopping server for backend {backend}")
        if backend == "ollama":
            return await handler.stop_server(pid=kwargs.get("pid"), port=kwargs.get("port"))
        elif backend == "llamafile":
            return await handler.stop_server(pid=kwargs.get("pid"), port=kwargs.get("port"))
        raise InferenceError(f"Server stop not applicable or implemented for backend {backend} via this method.")

    def cleanup_on_exit(self):
        """Call this on application shutdown to clean up managed resources, like llamafile servers."""
        self.logger.info("LLMInferenceManager performing cleanup_on_exit...")
        if self.llamafile:
            self.llamafile._cleanup_all_managed_servers() # This is synchronous
        # Add other cleanup if needed
        self.logger.info("LLMInferenceManager cleanup_on_exit complete.")

#
# End of LLMInferenceManager.py
########################################################################################################################
