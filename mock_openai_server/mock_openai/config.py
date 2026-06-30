"""
Configuration management for the mock OpenAI server.

Handles loading configuration from JSON/YAML files and environment variables.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import yaml


@dataclass
class ResponsePattern:
    """Pattern matching configuration for responses."""
    match: Dict[str, Any]
    response_file: str
    priority: int = 0

    def matches(self, request_data: Dict[str, Any]) -> bool:
        """Check if this pattern matches the given request data."""
        for key, pattern in self.match.items():
            if key == "model":
                if request_data.get("model") != pattern:
                    return False
            elif key == "content_regex":
                # Check message content against regex
                messages = request_data.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    content = last_message.get("content", "")
                    if isinstance(content, str):
                        if not re.search(pattern, content):
                            return False
                    else:
                        return False
                else:
                    return False
            elif key == "system_prompt":
                # Check for system message
                messages = request_data.get("messages", [])
                system_msg = next((m for m in messages if m.get("role") == "system"), None)
                if system_msg:
                    if pattern not in system_msg.get("content", ""):
                        return False
                else:
                    return False
            elif key in request_data:
                if request_data[key] != pattern:
                    return False
            else:
                return False
        return True


@dataclass
class StreamingConfig:
    """Streaming configuration."""
    enabled: bool = True
    chunk_delay_ms: int = 50
    words_per_chunk: int = 5  # More accurate name since we split by spaces


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8080
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    log_requests: bool = True
    simulate_errors: bool = False
    error_rate: float = 0.0  # Percentage of requests that should error
    require_auth: bool = True


@dataclass
class ResponseConfig:
    """Response configuration for each endpoint."""
    patterns: List[ResponsePattern] = field(default_factory=list)
    default: Optional[str] = None

    def find_matching_response(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Find the best matching response file for the request."""
        # Sort patterns by priority (higher first)
        sorted_patterns = sorted(self.patterns, key=lambda p: p.priority, reverse=True)

        for pattern in sorted_patterns:
            if pattern.matches(request_data):
                return pattern.response_file

        return self.default


@dataclass
class MockConfig:
    """Main configuration class."""
    server: ServerConfig = field(default_factory=ServerConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    responses: Dict[str, ResponseConfig] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)
    response_base_dir: Optional[str] = None
    scenario_failures: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "MockConfig":
        """Load configuration from a JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        config = cls.from_dict(data)
        if config.response_base_dir:
            response_base_dir = Path(config.response_base_dir)
            if not response_base_dir.is_absolute():
                config.response_base_dir = str(
                    (config_path.parent / response_base_dir).resolve()
                )
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockConfig":
        """Create configuration from a dictionary."""
        config = cls()

        # Parse server config
        if "server" in data:
            server_data = data["server"]
            config.server = ServerConfig(
                host=server_data.get("host", "0.0.0.0"),  # nosec B104
                port=server_data.get("port", 8080),
                cors_origins=server_data.get("cors_origins", ["*"]),
                log_requests=server_data.get("log_requests", True),
                simulate_errors=server_data.get("simulate_errors", False),
                error_rate=server_data.get("error_rate", 0.0),
                require_auth=server_data.get("require_auth", True),
            )

        # Parse streaming config
        if "streaming" in data:
            stream_data = data["streaming"]
            config.streaming = StreamingConfig(
                enabled=stream_data.get("enabled", True),
                chunk_delay_ms=stream_data.get("chunk_delay_ms", 50),
                words_per_chunk=stream_data.get("words_per_chunk", stream_data.get("tokens_per_chunk", 5))  # Support both names for backwards compatibility
            )

        # Parse response configs
        if "response_base_dir" in data:
            config.response_base_dir = data["response_base_dir"]

        if "responses" in data:
            for endpoint, resp_data in data["responses"].items():
                patterns = []
                if "patterns" in resp_data:
                    for pattern_data in resp_data["patterns"]:
                        patterns.append(ResponsePattern(
                            match=pattern_data["match"],
                            response_file=pattern_data["response_file"],
                            priority=pattern_data.get("priority", 0)
                        ))

                config.responses[endpoint] = ResponseConfig(
                    patterns=patterns,
                    default=resp_data.get("default")
                )

        # Parse models list
        if "models" in data:
            config.models = data["models"]

        if "scenario_failures" in data:
            config.scenario_failures = data["scenario_failures"]

        return config

    @classmethod
    def from_env(cls) -> "MockConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables
        if "MOCK_OPENAI_HOST" in os.environ:
            config.server.host = os.environ["MOCK_OPENAI_HOST"]

        if "MOCK_OPENAI_PORT" in os.environ:
            config.server.port = int(os.environ["MOCK_OPENAI_PORT"])

        if "MOCK_OPENAI_CORS_ORIGINS" in os.environ:
            config.server.cors_origins = os.environ["MOCK_OPENAI_CORS_ORIGINS"].split(",")

        if "MOCK_OPENAI_LOG_REQUESTS" in os.environ:
            config.server.log_requests = os.environ["MOCK_OPENAI_LOG_REQUESTS"].lower() == "true"

        if "MOCK_OPENAI_STREAMING_ENABLED" in os.environ:
            config.streaming.enabled = os.environ["MOCK_OPENAI_STREAMING_ENABLED"].lower() == "true"

        if "MOCK_OPENAI_CHUNK_DELAY" in os.environ:
            config.streaming.chunk_delay_ms = int(os.environ["MOCK_OPENAI_CHUNK_DELAY"])

        return config

    def merge_with_env(self) -> "MockConfig":
        """Merge current configuration with environment variables."""
        env_config = MockConfig.from_env()

        # Override specific fields from environment
        if os.environ.get("MOCK_OPENAI_HOST"):
            self.server.host = env_config.server.host

        if os.environ.get("MOCK_OPENAI_PORT"):
            self.server.port = env_config.server.port

        if os.environ.get("MOCK_OPENAI_CORS_ORIGINS"):
            self.server.cors_origins = env_config.server.cors_origins

        if os.environ.get("MOCK_OPENAI_LOG_REQUESTS"):
            self.server.log_requests = env_config.server.log_requests

        if os.environ.get("MOCK_OPENAI_STREAMING_ENABLED"):
            self.streaming.enabled = env_config.streaming.enabled

        if os.environ.get("MOCK_OPENAI_CHUNK_DELAY"):
            self.streaming.chunk_delay_ms = env_config.streaming.chunk_delay_ms

        return self


# Global configuration instance
_config: Optional[MockConfig] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> MockConfig:
    """Load and return the global configuration."""
    global _config

    if config_path:
        _config = MockConfig.from_file(config_path).merge_with_env()
    else:
        # Try to load from default locations
        default_paths = [
            Path("config.json"),
            Path("config.yaml"),
            Path("mock_config.json"),
            Path("mock_config.yaml"),
        ]

        for path in default_paths:
            if path.exists():
                _config = MockConfig.from_file(path).merge_with_env()
                break
        else:
            # No config file found, use defaults with env overrides
            _config = MockConfig().merge_with_env()

    return _config


def get_config() -> MockConfig:
    """Get the current configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
