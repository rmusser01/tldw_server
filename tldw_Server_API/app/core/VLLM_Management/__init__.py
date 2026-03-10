"""Managed vLLM instance persistence primitives."""

from .capabilities import derive_effective_capabilities, normalize_capabilities
from .command_builder import build_vllm_serve_argv
from .models import VLLMInstanceCreate, VLLMInstanceRecord
from .request_capabilities import infer_chat_request_capabilities
from .repository import VLLMInstanceRepository
from .resolver import (
    ResolvedVLLMRoute,
    get_default_vllm_instance_repository,
    resolve_vllm_instance_for_request,
)
from .sqlite_repo import SqliteVLLMInstanceRepository

__all__ = [
    "build_vllm_serve_argv",
    "derive_effective_capabilities",
    "get_default_vllm_instance_repository",
    "normalize_capabilities",
    "ResolvedVLLMRoute",
    "SqliteVLLMInstanceRepository",
    "VLLMInstanceCreate",
    "VLLMInstanceRecord",
    "VLLMInstanceRepository",
    "infer_chat_request_capabilities",
    "resolve_vllm_instance_for_request",
]
