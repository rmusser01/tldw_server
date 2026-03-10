"""Managed vLLM instance persistence primitives."""

from .capabilities import derive_effective_capabilities, normalize_capabilities
from .command_builder import build_vllm_serve_argv
from .models import VLLMInstanceCreate, VLLMInstanceRecord
from .repository import VLLMInstanceRepository
from .resolver import ResolvedVLLMRoute, resolve_vllm_instance_for_request
from .sqlite_repo import SqliteVLLMInstanceRepository

__all__ = [
    "build_vllm_serve_argv",
    "derive_effective_capabilities",
    "normalize_capabilities",
    "ResolvedVLLMRoute",
    "SqliteVLLMInstanceRepository",
    "VLLMInstanceCreate",
    "VLLMInstanceRecord",
    "VLLMInstanceRepository",
    "resolve_vllm_instance_for_request",
]
