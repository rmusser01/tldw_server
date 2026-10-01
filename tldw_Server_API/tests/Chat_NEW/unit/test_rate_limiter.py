import asyncio

import pytest

from tldw_Server_API.app.core.Chat.rate_limiter import TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_concurrent_consume_does_not_over_consume():
    capacity = 5
    bucket = TokenBucket(capacity=capacity, refill_rate=capacity / 60.0)

    async def worker():
        return await bucket.consume(1)

    results = await asyncio.gather(*(worker() for _ in range(10)))

    successes = sum(1 for r in results if r)
    failures = len(results) - successes

    # At most `capacity` workers should be allowed to consume a token.
    assert successes <= capacity
    # Under concurrent load, we expect some workers to fail.
    assert failures >= 1


@pytest.mark.asyncio
async def test_chat_fixture_scopes_finite_limits_and_restores_cached_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Chat_NEW's finite limits inside its fixture's ownership."""
    import os

    from tldw_Server_API.app.core.Chat import rate_limiter
    from tldw_Server_API.tests.Chat_NEW.conftest import (
        _reset_chat_rate_limiter_between_tests,
    )

    monkeypatch.setenv("TEST_CHAT_PER_USER_RPM", "7")
    original = rate_limiter.get_rate_limiter()
    with pytest.MonkeyPatch.context() as scoped:
        reset = _reset_chat_rate_limiter_between_tests.__wrapped__(scoped)
        next(reset)
        try:
            current = rate_limiter.get_rate_limiter()
            assert (
                current.config.global_rpm,
                current.config.per_user_rpm,
                current.config.per_conversation_rpm,
                current.config.per_user_tokens_per_minute,
                current.config.burst_multiplier,
            ) == (10, 2, 2, 1000, 1.0)
            allowed = [
                (await current.check_rate_limit("fixture-owner", "fixture-chat", 1))[0]
                for _ in range(3)
            ]
            assert allowed == [True, True, False]
        finally:
            reset.close()
    assert rate_limiter.get_rate_limiter() is original
    assert os.environ["TEST_CHAT_PER_USER_RPM"] == "7"
