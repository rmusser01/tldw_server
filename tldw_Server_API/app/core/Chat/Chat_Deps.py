# ---------------- Exceptions ----------------------------
class ChatAPIError(Exception):
    """Base exception for chat API call errors."""
    def __init__(self, message="An error occurred during the chat API call.", status_code=500, provider=None):
        self.message = message
        self.status_code = status_code # Suggested HTTP status code for the endpoint
        self.provider = provider
        super().__init__(self.message)

class ChatAuthenticationError(ChatAPIError):
    """Exception for authentication issues (e.g., invalid API key)."""
    def __init__(self, message="Authentication failed with the chat provider.", provider=None):
        super().__init__(message, status_code=401, provider=provider) # Default to 401

class ChatConfigurationError(ChatAPIError):
    """Exception for configuration issues (e.g., missing key, invalid model)."""
    def __init__(self, message="Chat provider configuration error.", provider=None):
        super().__init__(message, status_code=500, provider=provider) # Default to 500

class ChatBadRequestError(ChatAPIError):
    """Exception for bad requests sent to the chat provider (e.g., invalid params)."""
    def __init__(self, message="Invalid request sent to the chat provider.", provider=None):
        super().__init__(message, status_code=400, provider=provider) # Default to 400

class ChatRateLimitError(ChatAPIError):
    """Exception for rate limit errors from the chat provider."""
    def __init__(self, message="Rate limit exceeded with the chat provider.", provider=None):
        super().__init__(message, status_code=429, provider=provider) # Default to 429

class ChatProviderError(ChatAPIError):
    """Exception for general errors reported by the chat provider API."""
    def __init__(self, message="Error received from the chat provider API.", status_code=502, provider=None, details=None):
        # 502 Bad Gateway often suitable for upstream errors
        self.details = details # Store original error if available
        super().__init__(message, status_code=status_code, provider=provider)

# ---------------- End of Exceptions ----------------------------
