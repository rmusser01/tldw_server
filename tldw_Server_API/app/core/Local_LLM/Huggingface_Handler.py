# Hugging_FaceHandler.py
# Description:
#
# Imports
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
#
# Third-party imports
from loguru import logger

from tldw_Server_API.app.core.Local_LLM.LLM_Base_Handler import BaseLLMHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError, ModelDownloadError, \
    InferenceError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import HuggingFaceConfig

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
except ImportError:
    logger.error("transformers or torch not installed. Please install them: pip install transformers torch torchvision torchaudio accelerate bitsandbytes")
    # Raise or handle appropriately
    raise ImportError("HuggingFace handler requires 'transformers', 'torch', 'accelerate', 'bitsandbytes'.")
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# from .base_handler import BaseLLMHandler
# from .exceptions import ModelNotFoundError, ModelDownloadError, InferenceError
# from .utils_loader import logging, project_utils
# from .config_model import HuggingFaceConfig

class HuggingFaceHandler(BaseLLMHandler):
    def __init__(self, config: HuggingFaceConfig, global_app_config: Dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: HuggingFaceConfig # For type hinting
        self.models_dir = Path(self.config.models_dir)
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {} # Cache for loaded models and tokenizers

    async def list_models(self) -> List[str]:
        """Lists locally available Hugging Face models (directories in models_dir)."""
        if not self.models_dir.exists():
            return []
        return await asyncio.to_thread(
            lambda: [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        )

    async def is_model_available(self, model_name: str) -> bool:
        """Checks if a model is available locally (either as a full path or in models_dir)."""
        # Check if model_name is an absolute path to a model directory
        if Path(model_name).is_dir() and (Path(model_name)/"config.json").exists():
            return True
        # Check if it's a name in our local models_dir
        local_model_path = self.models_dir / model_name
        return local_model_path.is_dir() and (local_model_path / "config.json").exists()


    async def download_model(self, model_identifier: str, save_directory: Optional[str] = None) -> str:
        """
        Downloads a model and tokenizer from Hugging Face Hub.
        model_identifier: Hugging Face model ID (e.g., 'gpt2' or 'meta-llama/Meta-Llama-3-8B-Instruct')
        save_directory: Optional directory name (within self.models_dir) to save the model.
                        If None, uses the last part of model_identifier.
        """
        if save_directory:
            model_save_path = self.models_dir / save_directory
        else:
            model_save_path = self.models_dir / model_identifier.split('/')[-1]

        if model_save_path.exists() and (model_save_path / "config.json").exists():
            self.logger.info(f"Model '{model_identifier}' already downloaded at {model_save_path}")
            return str(model_save_path)

        model_save_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading model '{model_identifier}' to {model_save_path}...")

        try:
            # Running in a separate thread to avoid blocking asyncio event loop
            def _download():
                tokenizer = AutoTokenizer.from_pretrained(model_identifier)
                model = AutoModelForCausalLM.from_pretrained(model_identifier) # Add quantization here if desired globally
                tokenizer.save_pretrained(model_save_path)
                model.save_pretrained(model_save_path)

            await asyncio.to_thread(_download)
            self.logger.info(f"Successfully downloaded model '{model_identifier}' to {model_save_path}")
            return str(model_save_path)
        except Exception as e:
            self.logger.error(f"Failed to download model '{model_identifier}': {e}")
            if model_save_path.exists(): # Attempt to clean up partial download
                 try:
                    import shutil
                    await asyncio.to_thread(shutil.rmtree, model_save_path, ignore_errors=False)
                 except Exception as e_clean:
                    self.logger.error(f"Failed to cleanup partial download at {model_save_path}: {e_clean}")
            raise ModelDownloadError(f"Failed to download model '{model_identifier}': {e}")

    def _get_torch_dtype(self, dtype_str: Optional[str]):
        if not dtype_str:
            return None
        if dtype_str == "torch.bfloat16":
            return torch.bfloat16
        elif dtype_str == "torch.float16":
            return torch.float16
        elif dtype_str == "torch.float32":
            return torch.float32
        # Add more dtypes if needed
        self.logger.warning(f"Unsupported torch_dtype string: {dtype_str}. Returning None.")
        return None


    async def _load_model_and_tokenizer(self, model_name_or_path: str, quantization_config: Optional[Dict] = None):
        """Loads model and tokenizer, applying quantization if specified."""
        if model_name_or_path in self.loaded_models:
            return self.loaded_models[model_name_or_path]

        actual_path = model_name_or_path
        if not Path(actual_path).is_dir(): # If not a full path, assume it's in models_dir
            actual_path = str(self.models_dir / model_name_or_path)
            if not Path(actual_path).is_dir():
                raise ModelNotFoundError(f"Model directory not found at {actual_path} or {model_name_or_path}")

        self.logger.info(f"Loading model and tokenizer from: {actual_path}")

        bnb_config = None
        if quantization_config:
            load_in_4bit = quantization_config.get("load_in_4bit", False)
            load_in_8bit = quantization_config.get("load_in_8bit", False)
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
                    bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_compute_dtype=self._get_torch_dtype(quantization_config.get("bnb_4bit_compute_dtype", "torch.bfloat16")) or torch.bfloat16
                )
                self.logger.info("Applying 4-bit quantization.")
            elif load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.logger.info("Applying 8-bit quantization.")


        def _load():
            tokenizer = AutoTokenizer.from_pretrained(actual_path)
            model = AutoModelForCausalLM.from_pretrained(
                actual_path,
                device_map=self.config.default_device_map,
                torch_dtype=self._get_torch_dtype(self.config.default_torch_dtype),
                quantization_config=bnb_config,
                # low_cpu_mem_usage=True # Can be useful for large models
            )
            return model, tokenizer

        try:
            model, tokenizer = await asyncio.to_thread(_load)
            self.loaded_models[model_name_or_path] = (model, tokenizer)
            self.logger.info(f"Model and tokenizer for '{model_name_or_path}' loaded successfully.")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading model '{model_name_or_path}': {e}")
            raise InferenceError(f"Error loading model '{model_name_or_path}': {e}")

    async def unload_model(self, model_name_or_path: str):
        """Unloads a model from memory to free up resources."""
        if model_name_or_path in self.loaded_models:
            del self.loaded_models[model_name_or_path]
            # Python's garbage collector should handle freeing GPU memory if model/tokenizer are no longer referenced.
            # For more explicit control, especially with CUDA:
            if torch.cuda.is_available():
                await asyncio.to_thread(torch.cuda.empty_cache)
            self.logger.info(f"Model '{model_name_or_path}' unloaded from cache.")
        else:
            self.logger.info(f"Model '{model_name_or_path}' not found in loaded cache, no action taken.")


    async def chat_completion(self,
                              model_name_or_path: str,
                              messages: List[Dict[str, str]], # e.g., [{"role": "user", "content": "Hello"}]
                              max_new_tokens: int = 100,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              quantization_config: Optional[Dict] = None, # e.g. {"load_in_4bit": True}
                              **generation_kwargs) -> str:
        """
        Generates a chat completion using a Hugging Face model.
        Assumes model_name_or_path is a local path or a name of a model in self.models_dir.
        """
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
                "do_sample": True if temperature > 0 else False,
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
            raise InferenceError(f"Error during Hugging Face chat completion for '{model_name_or_path}': {e}")

    async def text_generation_pipeline(self,
                                  model_name_or_path: str,
                                  prompt: str,
                                  max_length: int = 100,
                                  quantization_config: Optional[Dict] = None,
                                  **pipeline_kwargs) -> str:
        """
        Uses the Hugging Face text-generation pipeline. Simpler for basic text generation.
        """
        if not await self.is_model_available(model_name_or_path):
            self.logger.error(f"Model {model_name_or_path} not found locally. Please download it first.")
            raise ModelNotFoundError(f"Model {model_name_or_path} not found locally.")

        # For pipeline, we usually pass the model and tokenizer names/paths directly.
        # But to use our cached/quantized versions if loaded:
        model, tokenizer = await self._load_model_and_tokenizer(model_name_or_path, quantization_config)

        def _generate_with_pipeline():
            # Determine device for pipeline
            device = model.device # Get device from the loaded model

            text_gen_pipeline = pipeline(
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
            raise InferenceError(f"Error during Hugging Face pipeline generation for '{model_name_or_path}': {e}")

#
# End of Hugging_FaceHandler.py
########################################################################################################################
