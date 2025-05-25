# LLM_Inference_Schemas.py
# Description:
#
# Imports
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, DirectoryPath, FilePath, HttpUrl
#
#########################################################################################################################
#
# Functions:

class BaseHandlerConfig(BaseModel):
    enabled: bool = True

class OllamaConfig(BaseHandlerConfig):
    models_dir: Optional[DirectoryPath] = None # Ollama manages its own models, but can be specified
    default_port: int = 11434

class HuggingFaceConfig(BaseHandlerConfig):
    models_dir: DirectoryPath = Path("models/huggingface_models") # Default path
    default_device_map: str = "auto"
    default_torch_dtype: str = "torch.bfloat16" # Store as string, convert later

    class Config:
        arbitrary_types_allowed = True


class LlamafileConfig(BaseHandlerConfig):
    llamafile_dir: DirectoryPath = Path("models/llamafile_exec") # Directory to store/find llamafile executable
    models_dir: DirectoryPath = Path("models/llamafile_models") # Directory to store llamafile models
    default_port: int = 8080
    default_host: str = "127.0.0.1"
    # Add other llamafile specific defaults if needed from the `start_llamafile` args

class LLMManagerConfig(BaseModel):
    ollama: Optional[OllamaConfig] = OllamaConfig()
    huggingface: Optional[HuggingFaceConfig] = HuggingFaceConfig()
    llamafile: Optional[LlamafileConfig] = LlamafileConfig()
    # Global settings for the library
    app_config: Dict[str, Any] = {} # To pass through parts of your project_config.settings

class LlamaCppConfig(BaseHandlerConfig):
    executable_path: FilePath = Path("vendor/llama.cpp/server") # Default path to llama.cpp server executable
    models_dir: DirectoryPath = Path("models/gguf_models")    # Directory for GGUF model files
    default_host: str = "127.0.0.1"
    default_port: int = 8080
    default_n_gpu_layers: int = 0  # Sensible default, user should override
    default_ctx_size: int = 2048
    default_threads: Optional[int] = None # Let llama.cpp decide by default
    # Add other common llama.cpp server arguments you want to default or control
    # e.g., default_main_gpu, default_tensor_split, etc.
    log_output_file: Optional[Path] = None # Optional: Path to save llama.cpp server logs

    class Config:
        arbitrary_types_allowed = True
        # If executable_path or models_dir might not exist at config load time,
        # you might need to remove FilePath/DirectoryPath validation temporarily
        # or ensure they are created before loading the config.

#
# End of LLM_Inference_Schemas.py
#######################################################################################################################