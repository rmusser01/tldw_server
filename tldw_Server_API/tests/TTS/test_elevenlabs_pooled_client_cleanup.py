"""Regression guard for TASK-13312.

`ElevenLabsAdapter._cleanup_resources` called `aclose()` on the HTTP client it
borrowed from `tts_resource_manager.ConnectionPool`. That pool caches by provider
name and hands the *same* object to every later borrower, and the only sanctioned
teardown is `close_pool`, which closes **and evicts**. Closing without evicting left
a dead client in `_pools`, so the documented admin operation
`POST /api/v1/audio/tts/providers/elevenlabs/unload` -- whose docstring says "so it
can be reloaded on demand" -- killed the provider until process restart with
`RuntimeError: Cannot send a request, as the client has been closed.`

`openai_adapter.py:_cleanup_resources` is the correct sibling and says why in a
comment: "HTTP clients are now managed by the resource manager / No need to manually
close them as they use connection pooling." It only clears the reference.

ADR-011 lists "permanent until process restart" as the *rejected* alternative to a
retry-after-cooldown, so this also restored an explicitly-rejected behaviour.
"""

import pytest

from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsAdapter
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit


class _PooledClient:
    """Stand-in for the shared pooled httpx client."""

    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _adapter(cls):
    # provider_name is a read-only property derived from the class name, so the
    # instance needs nothing but its borrowed client.
    inst = cls.__new__(cls)
    inst.client = _PooledClient()
    return inst


async def test_elevenlabs_cleanup_does_not_close_the_pooled_client() -> None:
    adapter = _adapter(ElevenLabsAdapter)
    client = adapter.client

    await adapter._cleanup_resources()

    assert client.aclose_calls == 0, (
        "cleanup closed the shared pooled client; the pool caches it by provider "
        "name and hands the same object to every later borrower, so the next "
        "request gets a closed client until process restart"
    )


async def test_elevenlabs_cleanup_still_clears_its_reference() -> None:
    """Releasing the reference is the correct half of the old behaviour."""
    adapter = _adapter(ElevenLabsAdapter)

    await adapter._cleanup_resources()

    assert adapter.client is None


async def test_elevenlabs_matches_its_openai_sibling() -> None:
    """Both remote adapters borrow from the same pool, so both must behave alike."""
    eleven = _adapter(ElevenLabsAdapter)
    openai = _adapter(OpenAIAdapter)
    eleven_client, openai_client = eleven.client, openai.client

    await eleven._cleanup_resources()
    await openai._cleanup_resources()

    assert eleven_client.aclose_calls == openai_client.aclose_calls == 0
    assert eleven.client is None and openai.client is None


async def test_owned_client_is_closed() -> None:
    """A client this adapter created itself must be closed, or it leaks.

    Five convenience methods lazily do `self.client = create_async_client()` when no
    pooled client exists. Never closing any client fixes the pool-corruption bug but
    leaks these. Ownership has to be tracked, not assumed.
    """
    adapter = _adapter(ElevenLabsAdapter)
    adapter._owns_client = True
    client = adapter.client

    await adapter._cleanup_resources()

    assert client.aclose_calls == 1, "a self-created client was leaked"
    assert adapter.client is None


async def test_pooled_client_is_not_closed() -> None:
    adapter = _adapter(ElevenLabsAdapter)
    adapter._owns_client = False
    client = adapter.client

    await adapter._cleanup_resources()

    assert client.aclose_calls == 0
    assert adapter.client is None


async def test_unknown_ownership_defaults_to_not_closing() -> None:
    """Fail towards the less damaging outcome.

    Wrongly closing a pooled client is a permanent provider outage until restart;
    wrongly leaving an owned client open is a bounded resource leak. With no
    ownership recorded, take the second.
    """
    adapter = _adapter(ElevenLabsAdapter)
    client = adapter.client  # no _owns_client attribute set at all

    await adapter._cleanup_resources()

    assert client.aclose_calls == 0
    assert adapter.client is None


async def test_cleanup_is_idempotent_when_client_already_released() -> None:
    adapter = _adapter(ElevenLabsAdapter)
    await adapter._cleanup_resources()

    # A second unload must not raise on the now-None reference.
    await adapter._cleanup_resources()
    assert adapter.client is None
