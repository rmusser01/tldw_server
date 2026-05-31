from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio import AudioConfig, load_audio_config
from .auth import AuthConfig, load_auth_config
from .chat import ChatConfig, load_chat_config
from .chunking import ChunkingConfig, load_chunking_config
from .database import DatabaseConfig, load_database_config
from .embeddings import EmbeddingsConfig, load_embeddings_config
from .jobs import JobsConfig, load_jobs_config
from .logging import LoggingConfig, load_logging_config
from .moderation import ModerationConfig, load_moderation_config
from .providers import ProvidersConfig, load_providers_config
from .rag import RAGConfig, load_rag_config
from .server import ServerConfig, load_server_config
from .stt import STTConfig, load_stt_config
from .types import ConfigParserLike


@dataclass(frozen=True)
class ConfigSections:
    auth: AuthConfig
    chat: ChatConfig
    chunking: ChunkingConfig
    database: DatabaseConfig
    embeddings: EmbeddingsConfig
    jobs: JobsConfig
    logging: LoggingConfig
    moderation: ModerationConfig
    rag: RAGConfig
    audio: AudioConfig
    providers: ProvidersConfig
    server: ServerConfig
    stt: STTConfig


def load_config_sections(config_parser: ConfigParserLike | None = None) -> ConfigSections:
    if config_parser is None:
        from tldw_Server_API.app.core import config as config_mod

        config_parser = config_mod.load_comprehensive_config()

    return ConfigSections(
        auth=load_auth_config(config_parser),
        chat=load_chat_config(config_parser),
        chunking=load_chunking_config(config_parser),
        database=load_database_config(config_parser),
        embeddings=load_embeddings_config(config_parser),
        jobs=load_jobs_config(config_parser),
        logging=load_logging_config(config_parser),
        moderation=load_moderation_config(config_parser),
        rag=load_rag_config(config_parser),
        audio=load_audio_config(config_parser),
        providers=load_providers_config(config_parser),
        server=load_server_config(config_parser),
        stt=load_stt_config(config_parser),
    )


__all__ = [
    "AudioConfig",
    "AuthConfig",
    "ChatConfig",
    "ChunkingConfig",
    "ConfigSections",
    "DatabaseConfig",
    "EmbeddingsConfig",
    "JobsConfig",
    "LoggingConfig",
    "ModerationConfig",
    "ProvidersConfig",
    "RAGConfig",
    "STTConfig",
    "ServerConfig",
    "load_audio_config",
    "load_auth_config",
    "load_chat_config",
    "load_chunking_config",
    "load_config_sections",
    "load_database_config",
    "load_embeddings_config",
    "load_jobs_config",
    "load_logging_config",
    "load_moderation_config",
    "load_providers_config",
    "load_rag_config",
    "load_server_config",
    "load_stt_config",
]
