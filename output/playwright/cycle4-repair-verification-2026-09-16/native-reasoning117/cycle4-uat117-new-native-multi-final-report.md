# Native UAT117: positive control; negative remains unverified

One fresh Cedar Character conversation was tested on the preserved multi-user runtime after commit4bd4e2dfda. Conversation:276cb69f-d61d-42f0-bdab-1442321213d8.

The visible Max Tokens input was filled with16 and Save was clicked. The actual configured Gemma complete-v2 request returned200 but omitted max_tokens. The model produced substantial reasoning followed by a valid final answer beginning “CEDAR GUIDE: First, finalize all technical infrastructure.” Therefore the new reasoning-only negative case remains UNVERIFIED. This is a successful reasoning-plus-answer control, not missing-answer recovery.

Reasoning was expanded in the actual UI. A normal reload returned canonical GET200 with the same three rows:

- Greeting4fb52d0b-7d3b-4dcd-9eba-7d6608198686.
- User e28a9e28-2cf0-4e82-824d-32611b662375.
- Assistant pa_8f3a-c8a3-717-fad4, containing a closed think section and the final answer.

## New product finding: UAT121

The visible token limit did not reach the Character request. actual-request.txt records request356, which contains no max_tokens. The caller in useChatActions.ts omits this setting even though the complete-v2 schema and provider call support it. After reload, the visible token input was blank (setting-value.txt). Subsequent source verification confirmed current-chat controls intentionally live only in the in-memory scoped model store; reload blank is expected session-setting lifetime, not a second persistence defect. No16 setting remained to restore; the dialog was closed without further edits.

## Evidence and limits

All artifacts share this report’s prefix. settings.txt shows the model-settings controls. actual-request.txt is the actual request body, captured without headers. persist-body.txt is the actual completion persistence request365. reasoning-expanded.txt shows reasoning and final answer separately. canonical-reload.txt records the actual reloaded rows. The CLI stream response-body capture was empty; no missing-answer inference is drawn from that capture limitation.

The first custom observer filtered messages/message/prompt and missed the context-only complete-v2 payload; supported CLI requests/request-body supplied the actual evidence. One real completion only, no mocked responses, seeds, resets, or product edits. The inference lease was released immediately after terminal completion. New data consists of this conversation and its three canonical rows. Existing records remain intact. No full UAT or new reasoning-only signoff is claimed.
