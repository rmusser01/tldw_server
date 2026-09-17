# Independent native acceptance: UAT031 /173 /176 /193 /199 /200

## Disposition
| Finding / task | Bounded recommendation | Criteria and limits |
|---|---|---|
| UAT031 — TASK13260.7 AC2 | **Native criterion passes; close UAT031 after retaining this evidence.** | Retry retains the original conversation and canonical greeting/user IDs; both derived greeting save actions succeed and persist. Keep the broader TASK13260.7 In Progress for its other criteria. |
| UAT173 — TASK13260.110 AC3 | **Pass; completion is supported with prior checked AC1/2.** | Actual native global character creation 201 precedes the successful continuation/retry and greeting saves. Creation occurred04:35 on the earlier repaired application; do not call it a fresh06:01 creation on a130. No additional native workspace matrix is claimed or required by AC3. |
| UAT176 — TASK13260.112 AC3 | **Pass; completion is supported with prior checked AC1/2.** | The same actual creation advances through genuine world-book initialization, and the subsequent original Retry succeeds. Global/workspace/transaction automated controls are already retained; this is a bounded global native pass. |
| UAT199 — TASK13260.137 AC3 | **Pass; completion is supported.** | Prior independent 22 tests/zero skips are recorded. Native successful assistant persistence is followed by canonical conversation version 1→4, message_count 2→3 and last_modified06:01:16.174, proving actual conversation updates now persist. |
| UAT200 — TASK13260.138 AC3 | **Pass; completion is supported.** | The original actual memory-query 500 is retained; ordinary Retry completes 200 with the configured local model, persists 200 / saved=true and reloads the answer. Prior independent 44/zero-skip and security checks are recorded. |
| UAT193 — TASK13260.131 | **Native default→conversation boundary passes for Bob; do not close literal AC3 yet.** | Fresh Bob default 3 creates its chat 201, later Retry and greeting saves pass. AC3 explicitly asks for a controlled failed-provider Retry, which did not occur. This original failure was the genuine UAT200 memory query 500 before model dispatch. No Alice-specific new creation or provider-negative-control pass is claimed. AC1/2 are supported by the task's retained independent 242+5 review, but this audit does not amend their checkboxes. |

No source, runtime, browser, credentials/session store, task/tracker or git mutation was performed. Only this private audit packet was written.

## Exact native chain
Normal identity events at06:00:25.897 and06:00:26.177 show Bob/id3. Subsequent identity events during fresh Chat, Flashcards and Notes loads continue to show Bob 3. Authentication headers are not part of these receipts.

The original retained capture `.tmp/uat193-031-native-20260917/bob-events.txt` records:
- 04:35:17.785 POST/chats/ for character3, statein-progress, sourcewebui-character-chat, globalscope;201 at04:35:17.887 creates c0d85261-ecd0-4c78-af71-47c88dab9d1d.
- Greeting42959e8f-b87f-470d-a0b4-436ee0e07667 and user126aec81-f0e8-41a6-b932-373dc36cee17 are created201. Both belong to that conversation.
- Complete-v2 then returns500 at04:35:18.363, errorcodeinternal_server_error/request71df096e-92b6-4ff9-afba-7aba1118647e. `bob-response.txt` shows the actual error bubble and Retry same model action. The task200 prior diagnosis attributes this to archived boolean SQL and a subsequent aborted WorldBook transaction; no artificial provider fault occurred.

The new retained sequence records:
- `bob-retry-click.txt` executes the visible Retry same model button. At06:01:03.598 the browser posts complete-v2 to the **same** c0d85261… chat, provider llama, exact configured Gemma4-26B local GGUF path, include_character_context=true, save_to_db=true, stream=true.
- HTTP 200 is recorded06:01:03.928; body-read completion timestamp06:01:15.980. This monitor does not retain raw SSE data; no SSE event/finish-reason assertion is made.
- At06:01:16.037 persistence posts assistantpa_2350-1e7b-4e2-b300 to that same chat. Response06:01:16.101 is 200 with the same assistant ID and saved=true. No new chat-create, branch, greeting or user-message POST exists in the retained Retry window.
- The settled UI displays BOB CHARACTER VERIFIED. `bob-retry-reload.txt` explicitly calls page.reload(). Twelve subsequent canonical message responses contain exactly the same ordered three rows: original greeting42959e8f…, original user126aec81…, assistantpa_2350-1e7b-4e2-b300. Total 3; no duplicates or conversation fork. The canonical raw assistant contains a think block plus the visible final marker, so this audit does not certify exact-output/model-quality compliance.
- Canonical conversation GET changes version 1/message_count 2 before Retry to version 4/message_count 3 after it. Created_at stays04:35:17.832; last_modified advances from04:35:18.085 to06:01:16.174.

## Original greeting saves
Actual menu actions are scoped to Assistant message 1, the original greeting.

1. At06:03:14.133 POST/chat/knowledge/save carries conversationc0d85261… and message42959e8f…, exact original greeting text, make_flashcard=false. Response 201 at06:03:14.181 returns note4f8aeb6e-eeab-4a1e-9bbf-895f843d007e and repeats those same chat/message IDs. UI reports Saved to Notes.
2. The later semantic Save to Flashcards action, visible question entry and Save flashcard click post06:06:32.516 with the **same original greeting/chat IDs**, make_flashcard=true, front `Bob UAT031: What did the original Helpful greeting say?`, and original greeting as back. Response 201 at06:06:32.581 returns note8dd346a3-ac97-42aa-be86-df4fcc39ff3a plus card72fcc656-452a-4e80-adc6-4b66ddc42cb4. UI reports Saved to Flashcards.
3. A fresh full Flashcards page navigation records list 200 at06:07:28.554/.573 with card72fcc…, client_id=3, original greeting back, source_ref_typenote and source_ref_id8dd346a3…. The settled Manage snapshot shows its actual question. This is a fresh route load, not an invented second page.reload call.
4. A fresh Notes page navigation then normal notes-open-button-4f8a… click yields five successful exact note GETs from06:09:36.201 through06:11:06.258. Each has client_id=3, original content, conversationc0d85261… and message42959e8f…. The independently viewed `bob-owned-greeting-note.png` shows the saved text, Saved/All changes saved, and matching conversation/message backlink. The two notes are the expected products of two distinct Save actions (the Flashcard save also creates its source note), not duplicate canonical Chat messages.

One earlier `bob-greeting-flashcard-dialog.txt` attempt failed with a stale snapshot reference. It is retained as a harness attempt and is not claimed to have applied. Later semantic actions plus 201 responses establish the actual save.

## Attribution and limits
- Parent-supplied running backend attribution is a130b8e5507ff2f1ec6a930a7aa12183a44b5d44. The prestart source receipt is05:29:59.276 and hashes the complete backend tree; `source-attribution.json` extracts the relevant source hashes. The original creation 201 is the earlier04:35 run on repaired48f89447fc described in retained tracker/task199, while the successful Retry/persistence/saves are the later a130 run. Do not collapse those stages.
- The source receipt is backend-only. No exact immutable frontend bundle hash is available in this packet, and current frontend HMR activity is not represented as a clean source freeze. Actual native actions and wire receipts remain directly inspectable.
- UAT207 Persona profile500s, UAT208 visual identity, and UAT209 Notes ownership remain separate retained failures. In particular the Notes screenshot also shows the foreign Alice fixture row already associated with209. This own-note success does not establish Notes tenant isolation, clean console, clean overall Chat, or a full four-profile matrix.
- Transient ready snapshots briefly show sign-in/not-connected/loading before settled authenticated content; settled snapshots and fresh 200 bodies are used for acceptance, not those transients.
- At06:11:43.977 the separate metadata-only receipt shows20 idle sessions, no retained public ordinary/partitioned-table locks, and read-only verification. This is a point-in-time observation, not proof of all-time connection health, indexes/catalog locks, or a new181 closure.

## Audit reproducibility
`input-manifest.json` records 40 exact input hashes, including original creation/error, current action/event/screenshot receipts, task criteria snapshots obtained through the official CLI, tracker context, source receipt and metadata snapshot. `parsed-evidence.json` contains a minimal derived event selection plus canonical content hashes and successful assertions. Inputs are unchanged; model reasoning text is not duplicated into the derived summary. No raw credentials, headers or native private backend logs were copied. Source/test checks were not rerun for this acceptance audit; prior reviewed results are attributed to their retained tasks.
