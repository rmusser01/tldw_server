# LLM_Inference_Schemas.py
# Description:
#
# Imports
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

#
#########################################################################################################################
#
# Functions:

class BaseHandlerConfig(BaseModel):
    enabled: bool = True

class OllamaConfig(BaseHandlerConfig):
    models_dir: Optional[Path] = None # Ollama manages its own models, but can be specified
    default_port: int = 11434
    max_pull_retries: int = 2  # Number of retries for model pull operations
    port_check_retries: int = 3  # Number of retries for port availability checks

class HuggingFaceConfig(BaseHandlerConfig):
    models_dir: Path = Path("models/huggingface_models") # Default path
    default_device_map: str = "auto"
    default_torch_dtype: str = "torch.bfloat16" # Store as string, convert later
    allowed_paths: Optional[list[Path]] = None
    max_loaded_models: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LlamafileConfig(BaseHandlerConfig):
    llamafile_dir: Path = Path("models/llamafile_exec") # Directory to store/find llamafile executable
    models_dir: Path = Path("models/llamafile_models") # Directory to store llamafile models
    default_port: int = 8080
    default_host: str = "127.0.0.1"
    allow_unvalidated_args: bool = False
    allow_cli_secrets: bool = False
    port_autoselect: bool = True
    port_probe_max: int = 10
    allowed_paths: Optional[list[Path]] = None
    allow_executable_auto_download: bool = False
    executable_sha256: Optional[str] = None
    http_timeout: float = 120.0  # HTTP request timeout in seconds
    readiness_timeout: float = 30.0  # Server readiness poll timeout in seconds
    stderr_read_timeout: float = 5.0  # Timeout for reading stderr during startup failures

class LlamaCppConfig(BaseHandlerConfig):
    executable_path: Path = Path("vendor/llama.cpp/server") # Default path to llama.cpp server executable
    models_dir: Path = Path("models/gguf_models")    # Directory for GGUF model files
    default_host: str = "127.0.0.1"
    default_port: int = 8080
    default_n_gpu_layers: int = 0  # Sensible default, user should override
    default_ctx_size: int = 2048
    default_threads: Optional[int] = None # Let llama.cpp decide by default
    allow_unvalidated_args: bool = False
    allow_cli_secrets: bool = False
    port_autoselect: bool = True
    port_probe_max: int = 10
    allowed_paths: Optional[list[Path]] = None
    http_timeout: float = 120.0  # HTTP request timeout in seconds
    readiness_timeout: float = 30.0  # Server readiness poll timeout in seconds
    stderr_read_timeout: float = 5.0  # Timeout for reading stderr during startup failures
    log_output_file: Optional[Path] = None # Optional: Path to save llama.cpp server logs

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # executable_path/models_dir are validated at runtime by handlers instead of on config creation.

class LLMManagerConfig(BaseModel):
    ollama: Optional[OllamaConfig] = Field(default_factory=OllamaConfig)
    huggingface: Optional[HuggingFaceConfig] = Field(default_factory=HuggingFaceConfig)
    llamafile: Optional[LlamafileConfig] = Field(default_factory=LlamafileConfig)
    llamacpp: Optional[LlamaCppConfig] = None
    # Global settings for the library
    app_config: dict[str, Any] = Field(default_factory=dict) # To pass through parts of your project_config.settings

#
# End of LLM_Inference_Schemas.py
#######################################################################################################################
