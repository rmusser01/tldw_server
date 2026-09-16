# UAT175 bounded diagnostic design

TASK13260.113. Exercise the real GeneratePanel / TanStack mutation / Flashcards service / background-proxy / request-core path with only Fetch, storage/auth and unrelated provider discovery boundaries stubbed. Install the exact locally installed Next Pages Router console/error handler in an isolated VM, recording its overlay dispatch while leaving the application promise chain and logged error intact. Observe process/window unhandled rejection separately.

First establish whether a rejected request actually escapes or whether the mutation onError's console.error(error) is classified as an unhandled runtime error by Next. Native capture alone cannot distinguish these. No production change before causal RED. Keep GeneratePanel and account/draft behavior unchanged; if the confirmed cause is severity classification, use the existing request diagnostic/reporting path and choose a minimal hook-local handling adjustment that preserves error propagation and unexpected errors. Root approval/coordination governs any scope change.

Controls: real failed save retains edited draft/retry, normal retry works, request HTTP status and owner scope are preserved, unexpected thrown errors remain visible, no duplicate dispatch or silent fulfillment. No browser/runtime/backend change.

## Confirmed cause and approved repair

The installed Next Pages Router console bridge directly sends the second Error argument to onUnhandledError. A real scoped POST/500 probe produced one overlay dispatch with the same Error caught by mutateAsync, zero Next rejection dispatches, and zero process unhandled rejections. The POST bypasses GET coalescing and has no discarded cleanup promise. This confirms a console-driven overlay, not an escaped rejection. Root approved covering both deck-create and card-create mutations.

No shared expected-HTTP-error classifier is suitable for this generic background-proxy Error shape (the separate TldwApiError class is not used here). A small hook-local reporter recognizes Error objects with integer HTTP statuses 400–599 and retains their warning diagnostics; all other errors retain console.error. Promise propagation, background request recording, and GeneratePanel catches remain unchanged. The additional existing scheduler test needed its pre-existing explicit undefined requestOptions argument reflected in the assertion; baseline replay independently reproduces that stale assertion.
