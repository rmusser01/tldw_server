# Hugging_FaceHandler.py
# Description:
#
# Imports
import asyncio
from pathlib import Path
from typing import Any, Optional

#
# Third-party imports
from loguru import logger

from tldw_Server_API.app.core.Local_LLM import handler_utils
from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import (
    InferenceError,
    ModelDownloadError,
    ModelNotFoundError,
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import HuggingFaceConfig
from tldw_Server_API.app.core.Utils.torch_import_guard import (
    can_import_torch_safely,
    safe_import_torch,
)

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None
pipeline = None
_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
#
########################################################################################################################
#
# Functions:


class HuggingFaceHandler(BaseLLMHandler):
    def __init__(self, config: HuggingFaceConfig, global_app_config: dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: HuggingFaceConfig # For type hinting
        self.models_dir = Path(self.config.models_dir)
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[tuple, Any] = {} # Cache for loaded models and tokenizers

    @staticmethod
    def _ensure_hf_dependencies() -> tuple[Any, Any, Any, Any, Any]:
        """Load optional Hugging Face deps lazily so app startup doesn't hard-fail."""
        global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, _TRANSFORMERS_IMPORT_ERROR

        torch_ok, torch_reason = can_import_torch_safely()
        if not torch_ok:
            raise InferenceError(
                "HuggingFace backend unavailable: torch import preflight failed. "
                "Install/repair torch for this runtime."
            ) from ImportError(torch_reason)

        if any(dep is None for dep in (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)):
            if _TRANSFORMERS_IMPORT_ERROR is not None:
                raise InferenceError(
                    "HuggingFace backend unavailable: failed to import transformers dependencies. "
                    "Install/repair: transformers accelerate bitsandbytes."
                ) from _TRANSFORMERS_IMPORT_ERROR
            try:
                from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
                from transformers import AutoTokenizer as _AutoTokenizer
                from transformers import BitsAndBytesConfig as _BitsAndBytesConfig
                from transformers import pipeline as _pipeline
            except Exception as exc:  # pragma: no cover - optional dependency failure path
                raise InferenceError(
                    "HuggingFace backend unavailable: failed to import transformers dependencies. "
                    "Install/repair: transformers accelerate bitsandbytes."
                ) from exc
            AutoModelForCausalLM = _AutoModelForCausalLM
            AutoTokenizer = _AutoTokenizer
            BitsAndBytesConfig = _BitsAndBytesConfig
            pipeline = _pipeline
            _TRANSFORMERS_IMPORT_ERROR = None

        if torch is None:
            try:
                _torch = safe_import_torch()
            except Exception as exc:  # pragma: no cover - optional dependency failure path
                raise InferenceError(
                    "HuggingFace backend unavailable: failed to import torch. "
                    "Install/repair: torch torchvision torchaudio."
                ) from exc
            torch = _torch

        return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    def _is_path_allowed(self, p: Path) -> bool:
        base_dirs = handler_utils.build_allowed_paths(
            self.models_dir,
            getattr(self.config, "allowed_paths", None),
        )
        return handler_utils.is_path_allowed(p, base_dirs)

    def _freeze_config(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple((k, self._freeze_config(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_config(v) for v in value)
        return value

    def _cache_key(self, model_name_or_path: str, quantization_config: Optional[dict]) -> tuple:
        return (model_name_or_path, self._freeze_config(quantization_config))

    def _resolve_model_dir(self, model_name_or_path: str) -> Path | None:
        candidate = Path(model_name_or_path).expanduser()
        if candidate.is_dir():
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError, ValueError):
                return None
            if self._is_path_allowed(resolved) and (resolved / "config.json").exists():
                return resolved
            return None

        local_model_path = (self.models_dir / model_name_or_path).expanduser()
        try:
            resolved_local = local_model_path.resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if self._is_path_allowed(resolved_local) and (resolved_local / "config.json").exists():
            return resolved_local
        return None

    def _enforce_cache_limit(self) -> None:
        try:
            max_loaded = int(getattr(self.config, "max_loaded_models", 1) or 1)
        except (TypeError, ValueError):
            max_loaded = 1
        max_loaded = max(1, max_loaded)
        while len(self.loaded_models) > max_loaded:
            oldest_key = next(iter(self.loaded_models))
            del self.loaded_models[oldest_key]

    async def list_models(self) -> list[str]:
        """Lists locally available Hugging Face models (directories in models_dir)."""
        if not self.models_dir.exists():
            return []
        return await asyncio.to_thread(
            lambda: [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        )

    async def is_model_available(self, model_name: str) -> bool:
        """Checks if a model is available locally (either as a full path or in models_dir)."""
        return self._resolve_model_dir(model_name) is not None


    async def download_model(self, model_identifier: str, save_directory: Optional[str] = None) -> str:
        """
        Downloads a model and tokenizer from Hugging Face Hub.
        model_identifier: Hugging Face model ID (e.g., 'gpt2' or 'meta-llama/Meta-Llama-3-8B-Instruct')
        save_directory: Optional directory name (within self.models_dir) to save the model.
                        If None, uses the last part of model_identifier.
        """
        if save_directory:
            candidate = Path(save_directory)
            model_save_path = candidate if candidate.is_absolute() else (self.models_dir / candidate)
        else:
            model_save_path = self.models_dir / model_identifier.split('/')[-1]

        if not self._is_path_allowed(model_save_path):
            raise ModelDownloadError("Model save path must resolve under allowed directories.")

        if model_save_path.exists() and (model_save_path / "config.json").exists():
            self.logger.info(f"Model '{model_identifier}' already downloaded at {model_save_path}")
            return str(model_save_path)

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model_save_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading model '{model_identifier}' to {model_save_path}...")
        _, auto_model_cls, auto_tokenizer_cls, _, _ = self._ensure_hf_dependencies()

        try:
            # Running in a separate thread to avoid blocking asyncio event loop
            def _download():
                tokenizer = auto_tokenizer_cls.from_pretrained(model_identifier)
                model = auto_model_cls.from_pretrained(model_identifier) # Add quantization here if desired globally
                tokenizer.save_pretrained(model_save_path)
                model.save_pretrained(model_save_path)

            await asyncio.to_thread(_download)
            self.logger.info(f"Successfully downloaded model '{model_identifier}' to {model_save_path}")
            return str(model_save_path)
        except Exception as e:
            self.logger.exception(f"Failed to download model '{model_identifier}': {e}")
            if model_save_path.exists(): # Attempt to clean up partial download
                 try:
                    import shutil
                    await asyncio.to_thread(shutil.rmtree, model_save_path, ignore_errors=False)
                 except Exception as e_clean:
                    self.logger.exception(f"Failed to cleanup partial download at {model_save_path}: {e_clean}")
            raise ModelDownloadError(f"Failed to download model '{model_identifier}': {e}") from e

    def _get_torch_dtype(self, dtype_str: Optional[str], torch_module: Any = None):
        if torch_module is None:
            torch_module, _, _, _, _ = self._ensure_hf_dependencies()
        if not dtype_str:
            return None
        if dtype_str == "torch.bfloat16":
            return torch_module.bfloat16
        elif dtype_str == "torch.float16":
            return torch_module.float16
        elif dtype_str == "torch.float32":
            return torch_module.float32
        # Add more dtypes if needed
        self.logger.warning(f"Unsupported torch_dtype string: {dtype_str}. Returning None.")
        return None


    async def _load_model_and_tokenizer(self, model_name_or_path: str, quantization_config: Optional[dict] = None):
        """Loads model and tokenizer, applying quantization if specified."""
        torch_module, auto_model_cls, auto_tokenizer_cls, bits_and_bytes_config_cls, _ = self._ensure_hf_dependencies()
        cache_key = self._cache_key(model_name_or_path, quantization_config)
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        actual_path = self._resolve_model_dir(model_name_or_path)
        if actual_path is None:
            raise ModelNotFoundError("Model directory was not found under allowed HuggingFace model paths.")

        self.logger.info(f"Loading model and tokenizer from: {actual_path}")

        bnb_config = None
        if quantization_config:
            load_in_4bit = quantization_config.get("load_in_4bit", False)
            load_in_8bit = quantization_config.get("load_in_8bit", False)
            if load_in_4bit:
                bnb_config = bits_and_bytes_config_cls(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
                    bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_compute_dtype=self._get_torch_dtype(
                        quantization_config.get("bnb_4bit_compute_dtype", "torch.bfloat16"),
                        torch_module=torch_module,
                    ) or torch_module.bfloat16
                )
                self.logger.info("Applying 4-bit quantization.")
            elif load_in_8bit:
                bnb_config = bits_and_bytes_config_cls(load_in_8bit=True)
                self.logger.info("Applying 8-bit quantization.")


        def _load():
            tokenizer = auto_tokenizer_cls.from_pretrained(actual_path)
            model = auto_model_cls.from_pretrained(
                actual_path,
                device_map=self.config.default_device_map,
                torch_dtype=self._get_torch_dtype(
                    self.config.default_torch_dtype,
                    torch_module=torch_module,
                ),
                quantization_config=bnb_config,
                # low_cpu_mem_usage=True # Can be useful for large models
            )
            return model, tokenizer

        try:
            model, tokenizer = await asyncio.to_thread(_load)
            self.loaded_models[cache_key] = (model, tokenizer)
            self._enforce_cache_limit()
            self.logger.info(f"Model and tokenizer for '{model_name_or_path}' loaded successfully.")
            return model, tokenizer
        except Exception as e:
            self.logger.exception(f"Error loading model '{model_name_or_path}': {e}")
            raise InferenceError(f"Error loading model '{model_name_or_path}': {e}") from e

    async def unload_model(self, model_name_or_path: str):
        """Unloads a model from memory to free up resources."""
        keys_to_remove = [k for k in self.loaded_models if k[0] == model_name_or_path]
        if keys_to_remove:
            for key in keys_to_remove:
                del self.loaded_models[key]
            # Python's garbage collector should handle freeing GPU memory if model/tokenizer are no longer referenced.
            # For more explicit control, especially with CUDA:
            try:
                torch_module, _, _, _, _ = self._ensure_hf_dependencies()
            except InferenceError:
                torch_module = None
            if torch_module is not None and torch_module.cuda.is_available():
                await asyncio.to_thread(torch_module.cuda.empty_cache)
            self.logger.info(f"Model '{model_name_or_path}' unloaded from cache.")
        else:
            self.logger.info(f"Model '{model_name_or_path}' not found in loaded cache, no action taken.")


    async def chat_completion(self,
                              model_name_or_path: str,
                              messages: list[dict[str, str]], # e.g., [{"role": "user", "content": "Hello"}]
                              max_new_tokens: int = 100,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              quantization_config: Optional[dict] = None, # e.g. {"load_in_4bit": True}
                              **generation_kwargs) -> str:
        """
        Generates a chat completion using a Hugging Face model.
        Assumes model_name_or_path is a local path or a name of a model in self.models_dir.
        """
        if not messages:
            raise InferenceError("messages must be a non-empty list for chat_completion.")
        if not await self.is_model_available(model_name_or_path):
            self.logger.error(f"Model {model_name_or_path} not found locally. Please download it first.")
            raise ModelNotFoundError(f"Model {model_name_or_path} not found locally.")

        model, tokenizer = await self._load_model_and_tokenizer(model_name_or_path, quantization_config)

        def _generate():
            # Apply chat template
            try:
                formatted_chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                self.logger.warning(f"Could not apply chat template for {model_name_or_path} (possibly missing in tokenizer_config.json or not a chat model): {e}. Using raw concatenation.")
                # Fallback for models without a proper chat template (less ideal)
                formatted_chat = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                # Add a generic instruction prompt if system message is present
                if messages[0]['role'] == 'system':
                     formatted_chat += "\nassistant:" # Basic prompt for generation

            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

            # Generate
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if temperature > 0 else None, # Temp 0 can be problematic
                "top_p": top_p if temperature > 0 else None, # top_p ignored if temp is 0
                "do_sample": temperature > 0,
                **generation_kwargs # Allow overriding defaults
            }
            # Filter out None values from gen_kwargs
            gen_kwargs = {k:v for k,v in gen_kwargs.items() if v is not None}


            outputs = model.generate(**inputs, **gen_kwargs)
            decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return decoded_output

        try:
            response_text = await asyncio.to_thread(_generate)
            self.logger.debug(f"Hugging Face chat completion successful for {model_name_or_path}.")
            return response_text
        except Exception as e:
            self.logger.error(f"Error during Hugging Face chat completion for '{model_name_or_path}': {e}", exc_info=True)
            raise InferenceError(f"Error during Hugging Face chat completion for '{model_name_or_path}': {e}") from e

    async def text_generation_pipeline(self,
                                  model_name_or_path: str,
                                  prompt: str,
                                  max_length: int = 100,
                                  quantization_config: Optional[dict] = None,
                                  **pipeline_kwargs) -> str:
        """
        Uses the Hugging Face text-generation pipeline. Simpler for basic text generation.
        """
        if not await self.is_model_available(model_name_or_path):
            self.logger.error(f"Model {model_name_or_path} not found locally. Please download it first.")
            raise ModelNotFoundError(f"Model {model_name_or_path} not found locally.")

        # For pipeline, we usually pass the model and tokenizer names/paths directly.
        # But to use our cached/quantized versions if loaded:
        _, _, _, _, pipeline_fn = self._ensure_hf_dependencies()
        model, tokenizer = await self._load_model_and_tokenizer(model_name_or_path, quantization_config)

        def _generate_with_pipeline():
            # Determine device for pipeline
            device = model.device # Get device from the loaded model

            text_gen_pipeline = pipeline_fn(
                "text-generation",
                model=model, # Use pre-loaded model
                tokenizer=tokenizer, # Use pre-loaded tokenizer
                device=device # Specify device
            )
            # Default pipeline kwargs that can be overridden
            pipe_args = {
                "max_length": max_length,
                "num_return_sequences": 1,
                **pipeline_kwargs
            }
            result = text_gen_pipeline(prompt, **pipe_args)
            return result[0]['generated_text']

        try:
            generated_text = await asyncio.to_thread(_generate_with_pipeline)
            self.logger.debug(f"Hugging Face pipeline generation successful for {model_name_or_path}.")
            return generated_text
        except Exception as e:
            self.logger.error(f"Error during Hugging Face pipeline generation for '{model_name_or_path}': {e}", exc_info=True)
            raise InferenceError(f"Error during Hugging Face pipeline generation for '{model_name_or_path}': {e}") from e

#
# End of Hugging_FaceHandler.py
########################################################################################################################
