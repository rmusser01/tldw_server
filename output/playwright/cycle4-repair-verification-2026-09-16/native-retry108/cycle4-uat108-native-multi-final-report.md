# Native UAT108 targeted recheck — FAILED

## Scope and runtime

TASK13260 / task49, repair commit 7dcee3d72c. Parent intentionally restarted API18301 (PID95509) and verified health200. Next18381 remained running. This check used the existing preserved multi-user profile and Alice account, normal browser reload, then a NEW saved ordinary General chat. No mock responses, API seeding, runtime configuration edits, prior-record cleanup, or product edits. This is a targeted recheck, not a full UAT or signoff.

Root granted the exclusive configured Gemma inference lease. One real unavailable Ollama request and one real Gemma Retry were performed sequentially. The lease was released immediately after the final Retry answer, and no further generation occurred.

## Failure

The initial failed request contained exactly one user message. Before Retry the UI already displayed two copies of that user message. Clicking the actual Retry action sent both copies even though metadata.tldw_retry_failed_turn was true. The real Gemma response returned200 and produced AMBER RESTORED. Both user copies were canonical after Retry and remained after ordinary reload.

- Conversation: 0dff2eee-7bf3-4e92-9d1c-688318065fc1
- Exact synthetic request: Cycle4 fixed Retry108 control 20260916_0449: reply exactly AMBER RESTORED.
- Initial provider/model: ollama / gemma3:1b; actual502; one user in outbound body.
- Retry model: llama:../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf; actual200; two identical users in outbound body.
- Retry display error exclusion passed: no __tldw_error__ or error JSON entered the outbound messages.

## Canonical persistence evidence

Actual GET /api/v1/chats/0dff2eee-7bf3-4e92-9d1c-688318065fc1/messages returned200 both on opening the canonical URL after Retry and on an ordinary reload of that URL. Both responses contained these same four rows:

| Role | Canonical ID | Content |
|---|---|---|
| system | ff76a4a8-8f26-4da3-a5e7-f0c47130baea | You are a helpful AI assistant. |
| user | 301aa236-efb4-4fa2-a2c0-8094e3b69ee1 | Exact synthetic request above |
| user | 21418efb-afbe-4824-9b39-8dc0e8f16548 | Exact synthetic request above |
| assistant | 0014e8b3-33c7-4ca6-a44e-9331226f40f8 | AMBER RESTORED. |

The screenshot reloaded.png was visually inspected: two user bubbles, the retained error bubble, and the final answer are visible. UI shows five messages because an old local-only error remains alongside four canonical rows.

Read-only IndexedDB capture local-mirror-after-reload.txt identifies local history pa_d15d-f61f-d6b-4ff5. Local user pa_6e81-2e36-daf-b5e0 is ACKed to canonical21418efb-afbe-4824-9b39-8dc0e8f16548; a second server-materialized user ends in :server:301aa236-efb4-4fa2-a2c0-8094e3b69ee1. Final local assistant pa_052c-c826-6f0-3fd4 is ACKed to0014e8b3-33c7-4ca6-a44e-9331226f40f8 and points to that first local user. Local error pa_fcc5-e63a-ee7-40cf has no canonical ACK. This is evidence for diagnosis, not a proven attribution of the race.

## Evidence and limits

All files share prefix /private/tmp/cycle4-uat108-native-multi-:

- negative-request.txt: initial actual502 and one-user body.
- negative.txt: duplicated UI before Retry. No canonical GET was captured before Retry, so this does not establish pre-Retry server row count.
- requests.txt: actual request bodies and response statuses. The SSE observer could not read the response body; its acks:[] and done:false are capture limits, not a claim that generation failed. Final answer and canonical IDs are independently established by UI and real GET responses.
- canonical-before-reload.txt means canonical opening AFTER Retry, before ordinary reload; canonical-after-reload.txt confirms the same rows after ordinary reload.
- local-mirror-after-reload.txt uses a camelCase history field that is absent; do not infer missing history ownership from that omitted field. Canonical mapping is independently established by URL and real responses.
- reloaded.txt and reloaded.png: final normal reload state.
- console.txt contains earlier session history; it is preserved as raw evidence and is not treated as all newly caused by this check.
- Early new-chat.txt captures a transient hydration/loading state; no message was sent there. The new saved chat was selected again after hydration settled.

Optional Note/backlink was not attempted after the persistence failure. Existing conversations and all previous evidence were preserved; the only new product data from this check is this synthetic conversation and its rows. No source change was made. Parent and round2_chat received the evidence for read-only diagnosis. Credential scan and SHA256 manifest are separate sibling artifacts.
