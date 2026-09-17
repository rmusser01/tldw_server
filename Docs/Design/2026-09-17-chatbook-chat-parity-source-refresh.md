# Chatbook parity source refresh — 2026-09-17

Tracking: TASK-13261.1. This is a source delta review during H1 implementation, supplementing the [parity inventory](2026-09-16-chatbook-console-parity-matrix.md) and [review closure](2026-09-16-chatbook-chat-parity-review-closure.md). It does not award an implemented or Equivalent status.

Latest verified pins on this date: server `59049e094e0845a4611ea725ae19b7c1754ea709`, Chatbook `c97a64eba54d18f88cecc77bf6233e208df8bf24`. The two refreshes below preserve their own evidence and scope.

## First refresh — provider and settings behavior

Fresh `git ls-remote --heads` queries on 2026-09-17 returned:

| Repository | Current remote dev | Change since reviewed baseline |
|---|---|---|
| tldw_server | `59049e094e0845a4611ea725ae19b7c1754ea709` | Unchanged. |
| tldw_chatbook | `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6` | [PR 2703](https://github.com/rmusser01/tldw_chatbook/pull/2703), merged after `24094f23d59c7a9d3cfac964c19fd263bc0393b2`. |

The delta contains two implementation commits and their merge, touching 67 files. Inspection used immutable Git objects from `24094f23..1c0327b3`; the unrelated Chatbook working tree was neither changed nor treated as source evidence. The previous audit's test results remain tied to its original pin. No new Chatbook tests or live provider requests were run for this refresh.

## Required parity refinements

| Existing area | New source behavior | Acceptance addition |
|---|---|---|
| C08 session settings/defaults | Reapplying the current model or returning to a remembered exact provider/model retains its conversation snapshot, including fields hidden by a quick surface. A new target starts from its defaults and carries only supported dirty fields. Explicit Inherit resolves the current lower-precedence defaults. | Set several visible and hidden parameters, Apply repeatedly, switch away and back, and reopen. Verify retained values and provenance, explicit Inherit, and sparse default mutations. Unsupported provider fields must not leak across targets. |
| C07 endpoint discovery and provider identity | Custom endpoint discovery uses the exact entry's endpoint and credential; discovery evidence is fenced by entry, connection and credential revision. Late results cannot populate a changed or dismissed target. Immutable custom endpoint identity remains distinct from its provider family. | Use two entries sharing a provider family, change endpoint/credentials during a held discovery, dismiss/reopen, and verify stale evidence cannot authorize the new target. Discovery must not rewrite saved settings or establish generation success. |
| C07/C08 context capacity | Serving metadata takes precedence over model metadata, followed by a provider fallback or the 32,000-token system fallback. Resolution carries source and verification status; estimated capacity must not be labeled verified. The same resolved capacity reaches the prepared request. | Inspect capacity for two endpoints serving the same model with different limits, switch targets during lookup, and verify the final prepared request uses the captured target's capacity. Exercise missing/invalid/oversized metadata and fallback provenance. Do not copy 32,000 into every destination provider as a universal verified limit. |
| C07 readiness and setup | Subscription credential status is a bounded background UI snapshot (`pending`, `ready`, `expired`, `missing`); actual send still resolves the credential. Status refreshes without blocking the current view. An unchanged subscription setup preserves inactive API-key configuration; explicit replacement/clear remains a separate mutation. | Hold the credential reader, keep UI controls usable, switch targets, expire the cached result, and verify only current state updates. Saving an unchanged subscription selection must not delete unrelated configured credentials. Port observable behavior through the destination's credential owner; no browser access to OS credentials or new tldw-agent responsibility is implied. |

Primary implementation evidence:

- [Settings rebasing](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/console_chat_controller.py), [default mutation identity](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/console_settings_defaults.py), and [settings modal discovery](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Widgets/Console/console_settings_modal.py).
- [Bounded serving metadata](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/console_context_window.py), [capacity precedence](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Utils/token_counter.py), and [send-time capacity binding](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/console_provider_gateway.py).
- [Readiness projection](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/provider_readiness.py), [background subscription state](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/LLM_Calls/anthropic_subscription.py), and [sparse setup persistence](https://github.com/rmusser01/tldw_chatbook/blob/1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6/tldw_chatbook/Chat/provider_setup_persistence.py).

## Effect on the current implementation

The delta does not modify `console_chat_store.py`, `chat_persistence_service.py`, `console_context_repository.py`, or the inspected DB, Agents and Sync/Sync_Interop paths. The changed controller sections concern settings rebasing and context-capacity provenance. It therefore adds no identified change to H1's selected ancestry, immutable legacy projection, accepted-parent settlement, safe-copy or uncertain-outcome contracts.

H1 Task3.2 already binds the exact composed payload and resolved model/settings behind a captured connection lease. Preserve that requirement for any capacity or provider option the existing composer actually consumes; do not perform fresh mutable resolution after finalization. Porting the new capacity-discovery service, full settings retention/default behavior and readiness surfaces belongs to C07/C08 and the model/context delivery, with their own tests. H1 remains in progress; H2/H3/H4/F02 and the broader parity rows remain open.

Before integration, recheck both remote heads again. Future source changes need another explicit delta review rather than rewriting historical evidence pins.

## Second refresh — optional installation and batch transcription

A later read-only `git ls-remote origin refs/heads/dev` on 2026-09-17 found server dev unchanged and Chatbook advanced to `c97a64eba54d18f88cecc77bf6233e208df8bf24`, merging [PR 2705](https://github.com/rmusser01/tldw_chatbook/pull/2705). The delta from `1c0327b3` contains three implementation/documentation commits and the merge, touching 17 files (six production modules plus tests/docs/tracking). Immutable objects were already available locally; no fetch, checkout change, source test run or clipboard/provider operation was performed.

The six production changes were inspected directly. [Selected STT warnings](https://github.com/rmusser01/tldw_chatbook/blob/c97a64eba54d18f88cecc77bf6233e208df8bf24/tldw_chatbook/Library/ingest_capabilities.py) and [ingest state](https://github.com/rmusser01/tldw_chatbook/blob/c97a64eba54d18f88cecc77bf6233e208df8bf24/tldw_chatbook/Library/library_ingest_state.py) now project dependency warnings to the selected batch-transcription backend before display and consent forecasting, preserving the captured inventory for later provider changes. Invalid restored provider values remain visibly repairable and cannot silently render as Auto. Auto still selects faster-whisper; the new projection does not switch runtimes. Parakeet setup distinguishes runtime packages from model download.

[Install-command copying](https://github.com/rmusser01/tldw_chatbook/blob/c97a64eba54d18f88cecc77bf6233e208df8bf24/tldw_chatbook/Utils/install_clipboard.py), [the shared optional-feature dialog](https://github.com/rmusser01/tldw_chatbook/blob/c97a64eba54d18f88cecc77bf6233e208df8bf24/tldw_chatbook/Utils/widget_helpers.py) and [the Library canvas](https://github.com/rmusser01/tldw_chatbook/blob/c97a64eba54d18f88cecc77bf6233e208df8bf24/tldw_chatbook/Widgets/Library/library_ingest_canvas.py) distinguish acknowledged native copying from unconfirmed remote/client copying, retain the literal command for manual recovery, quote install extras and clean up cancelled clipboard subprocesses. The remaining module adds strict batch-provider validation to the existing input validators.

For later shared ingestion/setup work, retain the selected-backend warning/consent match, invalid-saved-choice recovery, separate runtime/model readiness and honest clipboard-result behavior. These are batch-ingestion and optional-feature improvements; this delta does not establish a changed dictation, realtime voice or chat-history contract. Browser implementations should use their own clipboard capability rather than copying the TUI's native subprocess mechanism. No new `tldw-agent` responsibility is implied.

The delta changes no Chat, DB, Agents, Sync or Sync_Interop source path and adds no identified H1 ancestry, fork ownership, controller/admission or uncertain-outcome requirement. Current H1 fixes and required browser qualification continue unchanged. Earlier source test evidence remains tied to its earlier commit; broader parity receives no completion credit from this source inspection.
