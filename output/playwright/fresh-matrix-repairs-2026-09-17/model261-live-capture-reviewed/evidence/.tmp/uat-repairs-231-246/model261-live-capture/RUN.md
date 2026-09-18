# UAT261 diagnostic wrapper interface

This directory is a diagnostic-only Python module. It does not modify product
source, provider settings, browser state, or runtime configuration.

## Launcher contract

The root-owned launcher may add this directory to `PYTHONPATH` and load:

```text
uat261_capture:create_app --factory
```

It must set `UAT261_CAPTURE_DIR` to a newly created, absolute, non-symlink
directory with mode `0700`. The factory rejects a missing, non-private, or
non-fresh directory before importing the application. It creates these fixed,
mode-`0600` artifacts only after that preflight:

- `uat261_capture.json`
- `uat261_capture_status.json`

The module imports the existing `tldw_Server_API.app.main:app`, replaces the
loaded `character_chat_sessions.perform_chat_api_call` binding, and returns
that same app object. It makes no source snapshot mutation. The binding is
restored after its second observed call and on app shutdown.

## Observation boundary

Each of the first two seam calls delegates once to the original callable. The
wrapper forwards the original return value or every original lazy-stream frame
unchanged. Its explicit sync/async iterator delegates expose `close`/`aclose`
that call the original provider stream even when the wrapper was never started
or was only partly consumed. A provider close exception is re-raised unchanged.
Provider exceptions and cancellation are also re-raised unchanged.

The delegates do not acquire `iter(source)` or `source.__aiter__()` in their
constructors. Acquisition occurs only when a consumer calls `__iter__` or
`__aiter__`, with a direct `next`/`__anext__` fallback for nonstandard callers.
That preserves the product stream helper's timeout and cancellation ownership.
On cleanup, an acquired iterator is closed first and a distinct original source
second, matching the product helper's resource ordering.

The allowlisted record contains only message fingerprint/version/count/known
role counts, a combined non-secret generation-controls hash, and terminal
`finish_reason`, numeric usage, and a hash of `system_fingerprint`. Terminal
fields missing from a provider frame remain `null`.

`capture_state` records whether a terminal frame was observed. The independent
`stream_observation_state` remains `incomplete` after malformed-frame parsing
even if a later terminal frame is available, so terminal metadata never hides a
parsing limitation. `input_projection_state` similarly identifies a safe-input
projection failure without retaining its exception or input.

The wrapper deliberately stores `final_answer: null`. A passive provider stream
may have inline `<think>` text in `delta.content`, while other adapters can use
`reasoning_content`; content presence cannot truthfully establish the native
final answer. The native protocol owns visible exact-output assessment and its
separate canonical final hash.

Parsing or diagnostic artifact write failures are isolated from the product
call. Parsing marks the capture record `incomplete`; artifact I/O only changes
the diagnostic status in memory and never changes forwarded frames or raised
exceptions.

## Verification performed

```text
source .venv/bin/activate && python -m pytest \
  .tmp/uat-repairs-231-246/model261-live-capture/test_live_capture.py -q --tb=short
```

The final isolated run passed **18 tests** (including the retained independent
cleanup control). It made no network or provider call.
Static command receipts and hashes are recorded in `REPORT.md`.
