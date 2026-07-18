"""Compatibility re-exports for the centralized chat exception family."""

from tldw_Server_API.app.core.exceptions import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
    ProviderCredentialTerminalError,
    SanitizedProviderStreamError,
)

__all__ = [
    "ChatAPIError",
    "ChatAuthenticationError",
    "ChatBadRequestError",
    "ChatConfigurationError",
    "ChatProviderError",
    "ChatRateLimitError",
    "ProviderCredentialTerminalError",
    "SanitizedProviderStreamError",
]
