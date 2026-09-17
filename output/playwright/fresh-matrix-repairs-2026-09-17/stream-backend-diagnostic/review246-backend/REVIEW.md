# Independent UAT246 backend diagnostic review

**CLEAR as bounded diagnostic coverage. No blocking test changes requested. UAT246 remains open.**

Independent official SQLite/PostgreSQL run: **8 passed, zero skips**, four existing warnings, 9.19s. Test SHA256 `a39b84a8e91d0caacaab1db321fc7e6980f453674bb3cb46c9aa511fe1b7c22b`; test, endpoint and adapter hashes remained unchanged. Ruff, Bandit (only B101 excluded), compilation and formatting pass.

## What this proves

The eight cases cover both database backends, STREAMS_UNIFIED0/1, and completion/cancellation. They directly invoke the real complete-v2 endpoint, build its actual database-backed conversation context, and consume its body iterator through the real local adapter. Only provider dispatch/transport, credential runtime, strict model discovery and rate limiter are controlled.

The gated upstream establishes a meaningful ordering: a response object with status200 exists while the first iterator result is still pending; releasing the first upstream role frame makes that frame available before terminal content is released. The completion path then observes BEEP BOOP and exactly one DONE marker. Both paths verify one upstream request, the character instruction in the actual adapter request payload, resource cleanup before credential-runtime close, and no additional database message IDs. That last assertion matches the endpoint's existing stream-does-not-persist contract despite save_to_db=true.

## Limits and remaining acceptance

This exercises Python iterators, **not ASGI send, socket timing, Next proxy, browser rendering, real provider behavior or native first-byte latency**. Cancellation explicitly releases the fake upstream before requiring cleanup; it proves eventual cleanup for that cooperative boundary, not interruption of silent blocking I/O. No replay is proved only as one invocation in each tested flow, not retry/reconnect behavior. The test does not assert every possible error frame is absent or establish durable assistant persistence.

These are useful diagnostic controls, not a causal RED-to-GREEN repair. The original native45-second failure remains unexplained; an actual original-scenario completion and reload are still required. No production source, held runtime/profile, browser, task or git state was changed by this review.
