# UAT246 frontend transport qualification (no product timeout repair)

The added `qualifies real Character stream lifetime` cases call real useChatActions → streamCharacterChatCompletion → bgStream → a controlled fetch/ReadableStream. Synthetic configuration uses the normal device-credential shape. No native credentials, provider, browser, or held matrix database is accessed.

These controls show:

- HTTP200 headers with no body bytes reach the configured default 45-second idle deadline once; the acknowledged user identity survives and there is no replay or extra user write.
- An explicit 12-second configured idle budget remains honored.
- A role chunk after30seconds and reasoning after another30seconds reset inactivity, allowing a final answer after80seconds total. This qualifies transport activity, not a requirement for first visible answer within45seconds.
- Caller abort cancels one stream without replay.
- Caller abort together with snapshot invalidation preserves replacement private state and suppresses persistence.

They do **not** establish why original native TestBot produced no bytes. Parent owns the separate backend delivery8-control receipt. No timeout setting or production transport code changed here.

## Separately retained invalidation-only candidate

`authority-invalidation-only-candidate.test-snapshot.txt` preserves the exact intermediate test. The first attempt aborted only the injected snapshot's invalidation signal and then awaited a deliberately silent stream, causing a5-second test timeout. A second attempt explicitly delivered late bytes and closed that stream (`transport-lifetime-green.log`, misleading historical filename retained). It preserved Bob's local draft but invoked old-character persistence once.

This fixture mocks service-prompt snapshot acquisition; its invalidation controller is separate from the request signal. Real `service-prompts.ts` invalidation aborts its own lease controller as well. The action's actual authentication/context wiring has not been reproduced end-to-end by this candidate. It is **unproven actual-auth lifecycle risk**, not an accepted native finding and not a repaired requirement. The final cancellation test explicitly drives both signals and does not claim that it resolves the candidate. Parent requested preservation without expanding the product patch.

## Other fixture corrections retained

The first real-stream attempt seeded only legacy config.apiKey and never reached fetch; current credential resolution correctly requires manual/device metadata bound to origin. `real-transport-{initial,diagnostic}.log` preserve that fixture error. The corrected fixture reaches the real HTTP400 and successful SSE response, and its original-source replay reproduces the actual unavailable-model guidance failure.
