"""Request dependencies for the server-side provider credential policy."""

from fastapi import HTTPException, status

from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError


async def require_provider_override_snapshot() -> dict[str, llm_provider_overrides.LLMProviderOverride]:
    """Read one healthy snapshot on the request loop so recovery can be scheduled."""
    try:
        return llm_provider_overrides.get_llm_provider_overrides_snapshot()
    except ByokResolutionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider configuration unavailable",
        ) from None
